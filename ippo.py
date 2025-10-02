import os
import csv
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from ippo_agent import IPPOAgent
from buffer import TrajectoryBuffer
from lbf_wrapper import LBFParallelLike
from wafl import build_wafl_weights_policy, wafl_fuse_params_with_self, sample_calib_from_buffer
from metrics_utils import count_params, vector_nbytes, Timer, MetricsLogger

def make_env(env_name="lbf_gym", grid=(8, 8), players=3, foods=3, coop=False, max_episode_steps=50, seed=1):
    if env_name != "lbf_gym":
        raise ValueError(f"Unknown env: {env_name}")
    env = LBFParallelLike(
        grid=grid, players=players, foods=foods, coop=coop,
        max_episode_steps=max_episode_steps, seed=seed
    )
    env.reset(seed=seed)
    return env

def get_spaces(env, agent):
    obs_space = env.observation_space(agent)
    act_space = env.action_space(agent)
    if hasattr(act_space, "n"):
        act_dim = 1
    else:
        act_dim = act_space.shape[0]
    return obs_space, act_space, act_dim

def rollout_once(env, agents, buffers, steps_per_batch, device):
    agent_ids = list(agents.keys())
    for aid in agent_ids:
        agents[aid].reset_hidden()
    
    obs_dict, _ = env.reset()
    collected_steps = 0
    ep_returns = []
    ep_lengths = []

    running_return = 0.0
    running_len = 0

    active_in_ep = {aid: False for aid in agent_ids}

    while collected_steps < steps_per_batch:
        actions = {}
        logps = {}
        values = {}
        hprev = {}

        for aid in agent_ids:
            if agents[aid].use_gru:
                hprev[aid] = agents[aid].h
            a, logp, v = agents[aid].act(obs_dict[aid])
            actions[aid] = a
            logps[aid] = logp
            values[aid] = v

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        collected_steps += 1
        running_len += 1
        
        step_team_return = sum(float(r) for r in rewards.values())
        running_return += step_team_return

        for aid in agent_ids:
            if buffers[aid].ptr >= buffers[aid].max_size:
                continue
            rew = float(rewards.get(aid, 0.0))
            done_flag = float(terminations.get(aid, False) or truncations.get(aid, False))
            act = actions[aid]
            buffers[aid].store(
                obs=np.asarray(obs_dict[aid], dtype=np.float32),
                act=act,
                rew=rew,
                val=values[aid],
                logp=logps[aid],
                done=done_flag,
                h=hprev.get(aid, None)
            )
            active_in_ep[aid] = True

        ep_done = any(terminations.values()) or any(truncations.values())

        if ep_done or collected_steps >= steps_per_batch:
            with torch.no_grad():
                for aid in agent_ids:
                    if not active_in_ep[aid]:
                        continue
                    if ep_done:
                        last_val = 0.0  
                    else:
                        obs_last = torch.as_tensor(next_obs[aid], dtype=torch.float32, device=device)
                        if agents[aid].use_gru:
                            last_val = agents[aid].ac.value(obs_last, h=agents[aid].h)
                        else:
                            last_val = agents[aid].ac.critic(obs_last.unsqueeze(0)).squeeze(0).item()
                    buffers[aid].finish_path(last_val=last_val)

            if ep_done:
                ep_returns.append(running_return)
                ep_lengths.append(running_len)

            if ep_done and collected_steps < steps_per_batch:
                obs_dict, _ = env.reset()
                for aid in agent_ids:
                    agents[aid].reset_hidden()
                running_return = 0.0
                running_len = 0
                active_in_ep = {aid: False for aid in agent_ids}
            else:
                obs_dict = next_obs  
                break

        obs_dict = next_obs

    return ep_returns, ep_lengths

def linear_anneal(cur_steps: int, total_decay_steps: int, start: float, end: float) -> float:
    if total_decay_steps <= 0:
        return end
    ratio = min(max(cur_steps / float(total_decay_steps), 0.0), 1.0)
    return start + (end - start) * ratio

def evaluate_policy(make_env_fn, agents_dict, players=3, foods=3, episodes=100, device="cpu"):
    """
    return: mean_norm_return, std_norm_return
    normalized caliber: each round normalized_return = (#foods_collected) / foods
    """
    env = make_env_fn()
    agent_ids = list(getattr(env, "possible_agents", [f"agent_{i}" for i in range(players)]))
    assert len(agent_ids) == players
    scores = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        foods_collected = 0.0
        ep_raw = 0.0
        reward_unit = None
        
        while True:
            if len(agent_ids) == 0:
                break

            actions = {}
            for aid in agent_ids:
                if aid not in obs:
                    continue
                obs_t = torch.as_tensor(obs[aid], dtype=torch.float32, device=device)
                actions[aid] = agents_dict[aid].act_deterministic(obs_t)
            
            if len(actions) == 0:
                break

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            step_team = float(sum(rewards.values()))
            ep_raw += step_team

            if step_team > 0.0:
                if (reward_unit is None) or (0.0 < step_team < reward_unit):
                    reward_unit = step_team
                foods_collected += step_team / (reward_unit if reward_unit else step_team)

            obs = next_obs
            if any(terminations.values()) or any(truncations.values()):
                break

        foods_in_env = getattr(env, "foods", 1)
        norm = float(np.clip(foods_collected / float(foods_in_env), 0.0, 1.0))
        scores.append(norm)

    env.close()
    return float(np.mean(scores)), float(np.std(scores))

def train_ippo(
    env_name="lbf_gym",
    total_steps=1000000,            
    steps_per_agent_batch=2048,      
    grid=(8, 8),
    players=3,
    foods=3,
    coop=False,
    max_episode_steps=50,
    hidden=(128, 128),
    wafl_enable=False,
    wafl_period_batches=2,
    wafl_alpha=0.3,
    wafl_tau=2.0,
    wafl_topk=2,
    wafl_eps_fuse=0.01,
    use_gru=False,
    lr=3e-4,
    clip_ratio=0.2,
    vf_coef=0.5,
    ent_coef_start=0.05,     
    ent_coef_end=0.01,       
    ent_coef_decay_ratio=0.1,  
    train_iters=4,
    minibatch_size=256,
    seed=1,
    device="cpu",
    log_dir="./ippo_logs",
    log_csv_name="ippo_wafl_train_log.csv",
    eval_every_steps=50000,
    eval_episodes=100,
    eval_csv_name="ippo_wafl_eval.csv"
):
    os.makedirs(log_dir, exist_ok=True)
    log_csv_path = os.path.join(log_dir, f"{env_name}_{log_csv_name}")
    with open(log_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "ep_return", "ep_len"])
    eval_csv_path = os.path.join(log_dir, f"{env_name}_{eval_csv_name}")
    with open(eval_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "normalized_return_mean", "normalized_return_std"])

    env = make_env(env_name, grid=grid, players=players, foods=foods, coop=coop, max_episode_steps=max_episode_steps, seed=seed)
    agent_ids = env.possible_agents

    agents = {}
    buffers = {}
    for aid in agent_ids:
        obs_space, act_space, act_dim = get_spaces(env, aid)
        agents[aid] = IPPOAgent(
            obs_space, act_space, hidden=hidden, use_gru=use_gru, lr=lr, clip_ratio=clip_ratio,
            vf_coef=vf_coef, ent_coef=ent_coef_start, train_iters=train_iters,
            batch_size=steps_per_agent_batch, minibatch_size=minibatch_size, device=device
        )
        
        H = getattr(agents[aid], "hidden_dim", None) if use_gru else None
        buffers[aid] = TrajectoryBuffer(
            obs_dim=(obs_space.shape[0],),
            act_dim=(1 if hasattr(act_space, "n") else act_space.shape[0]),
            size=steps_per_agent_batch,
            gamma=0.99, lam=0.95, device=device, hidden_dim=H
        )

    if wafl_enable:
        for aid in agent_ids:
            if hasattr(agents[aid], "snapshot_params"):
                agents[aid].snapshot_params()

    timer = Timer()
    timer.start()
    metrics = MetricsLogger()
    comm_bytes_total = 0
    comm_rounds = 0
    algo_name = "IPPO_WAFL" if wafl_enable else "IPPO"

    sample_aid = agent_ids[0]
    actor_params_per = count_params(agents[sample_aid].ac)
    actor_vec_bytes = vector_nbytes(agents[sample_aid].ac, dtype_bytes=4)
    critic_params_shared = 0

    steps_collected = 0
    batches_since_comm = 0
    pbar = tqdm(total=total_steps, desc="Training (IPPO, LBF)", unit="step", dynamic_ncols=True, smoothing=0.1, leave=True)

    while steps_collected < total_steps:
        # 1) Rollout
        ep_returns, ep_lengths = rollout_once(env, agents, buffers, steps_per_agent_batch, device)
        batch_steps = steps_per_agent_batch * len(agent_ids)
        steps_collected += steps_per_agent_batch * len(agent_ids)

        # 2) Update per-agent
        decay_steps = int(ent_coef_decay_ratio * total_steps)
        current_ent = linear_anneal(cur_steps=steps_collected, total_decay_steps=decay_steps, start=ent_coef_start, end=ent_coef_end)
        for aid in agent_ids:
            agents[aid].ent_coef = float(current_ent)
            data = buffers[aid].get(adv_norm=True)
            if device.startswith("cuda"):
                for k, v in data.items():
                    if isinstance(v, torch.Tensor) and v.device.type == "cpu":
                        data[k] = v.pin_memory()               
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.to(device, non_blocking=True)
            else:
                for k in data:
                    data[k].to(device)
            agents[aid].update(data)

        # 3) (Optional) WAFL P2P Communication
        batches_since_comm += 1
        if wafl_enable and (batches_since_comm % wafl_period_batches == 0):
            # a) small calibration sets sampled from their buffers
            calib_obs_by_aid = {aid: sample_calib_from_buffer(buffers[aid], n=512) for aid in agent_ids}
            # b) build WAFL weights based on policy
            weights = build_wafl_weights_policy(agents, calib_obs_by_aid, tau=wafl_tau, topk=wafl_topk, self_weight=0.0)
            # c) compute neighbor weight parameter vector
            trunk_vec = {aid: agents[aid].trunk_vector() for aid in agent_ids}
            target_vec = wafl_fuse_params_with_self(trunk_vec, weights)
            # d) personalized KL fusion and refresh snapshot
            for aid in agent_ids:
                if len(weights[aid][0]) == 0:
                    continue
                agents[aid].trust_region_fuse_trunk(
                    target_trunk_vec_cpu=target_vec[aid], 
                    calib_obs=calib_obs_by_aid[aid],
                    eps_fuse=wafl_eps_fuse, alpha_max=wafl_alpha, steps=10,
                )
                agents[aid].clear_optimizer_state()
                agents[aid].snapshot_params()
            # e) calculate communication volume
            avg_deg = 0.0
            for (neigh_ids, _, _) in weights.values():
                avg_deg += len(neigh_ids)
            avg_deg = avg_deg / max(1, len(agent_ids))
            comm_bytes = int(len(agent_ids) * avg_deg * actor_vec_bytes)
            comm_bytes_total += comm_bytes
            comm_rounds += 1

        # 3) Log
        ep_returns_avg = float(np.mean(ep_returns)) if len(ep_returns) else float("nan")
        ep_lengths_avg = float(np.mean(ep_lengths)) if len(ep_lengths) else float("nan")
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([steps_collected, ep_returns_avg, ep_lengths_avg])

        # 4) Evaluate
        if (steps_collected // eval_every_steps) != ((steps_collected - batch_steps) // eval_every_steps):
            def _eval_env_factory():
                return make_env(
                    env_name=env_name, grid=grid, players=players, foods=foods, coop=coop,
                    max_episode_steps=max_episode_steps, seed=seed
                )
            m, s = evaluate_policy(_eval_env_factory, agents, players=players, foods=foods, episodes=eval_episodes, device=device)
            with open(eval_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([steps_collected, m, s])
            pbar.set_postfix_str(f"batch={batch_steps}, total={steps_collected}, eval_norm={m:.3f}")

        pbar.set_postfix({"ep_return": ep_returns_avg})
        pbar.update(steps_per_agent_batch * len(agent_ids))

    timer.stop()
    num_batches = steps_collected // int(sum(buffers[aid].max_size for aid in agent_ids))
    metrics.append(
        algo=algo_name, 
        env_name=env_name, 
        total_steps=steps_collected,
        wall_seconds=round(timer.elapsed, 2),
        n_agents=len(agent_ids),
        actor_params_per_agent=actor_params_per,
        critic_params_shared=critic_params_shared,
        steps_per_agent_batch=steps_per_agent_batch,
        train_iters=train_iters, 
        minibatch_size=minibatch_size,
        num_batches=num_batches,
        comm_rounds=comm_rounds,
        comm_bytes_total=comm_bytes_total
    )
    print(f"[METRICS] {algo_name}: wall={timer.elapsed:.2f}s, comm={comm_bytes_total/1e6:.2f} MB over {comm_rounds} rounds")

    env.close()
    pbar.close()
    print(f"[Done] logs saved to {log_csv_path}")

if __name__ == "__main__":
    train_ippo(
        env_name="lbf_gym",
        total_steps=2000000,
        steps_per_agent_batch=4096,
        grid=(8, 8),
        players=3,
        foods=3,
        coop=True,
        max_episode_steps=150,  
        hidden=(128, 128),
        wafl_enable=True,
        wafl_period_batches=2,
        wafl_alpha=0.5,
        wafl_tau=2.0,
        wafl_topk=2,
        wafl_eps_fuse=0.01,
        use_gru=False,
        lr=3e-4,
        clip_ratio=0.2,
        vf_coef=0.5,
        ent_coef_start=0.02,     
        ent_coef_end=0.005,       
        ent_coef_decay_ratio=0.4, 
        train_iters=6,
        minibatch_size=512,
        seed=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="./ippo_logs",
        log_csv_name="ippo_wafl2_train_log.csv",
        eval_every_steps=50000,
        eval_episodes=100,
        eval_csv_name="ippo_wafl2_eval.csv"
    )