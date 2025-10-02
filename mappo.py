import os
import csv
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from lbf_wrapper import LBFParallelLike
from ppo_fc import CentralCritic
from mappo_agent import MAPPOAgent, update_central_critic
from buffer import MAPPOBuffer
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
    return obs_space, act_space

def build_joint_obs(agent_ids_fixed, obs_dict, obs_template):
    vecs = []
    for aid in agent_ids_fixed:
        if aid in obs_dict:
            vecs.append(np.asarray(obs_dict[aid], dtype=np.float32))
        else:
            vecs.append(np.zeros_like(obs_template[aid], dtype=np.float32))
    return np.concatenate(vecs, axis=-1).astype(np.float32)

def critic_forward(central_critic, joint_obs_t, agent_idx, device):
    if not torch.is_tensor(agent_idx):
        agent_id_t = torch.tensor([agent_idx], dtype=torch.long, device=device)
    else:
        agent_id_t = agent_idx.view(1).to(device=device, dtype=torch.long)
    return central_critic(joint_obs_t, agent_id_t)

def rollout_once(env, agent_ids, agents, central_critic, buffers, steps_per_batch, device):
    obs_dict, _ = env.reset()
    ep_returns = []
    ep_lengths = []

    agent_ids_fixed = list(agent_ids)
    idx_of = {aid: i for i, aid in enumerate(agent_ids_fixed)}
    
    obs_template = {aid: np.asarray(obs_dict[aid], dtype=np.float32) for aid in agent_ids_fixed}
    env_steps = 0
    running_return = 0.0
    running_len = 0
    active_in_ep = {aid: False for aid in agent_ids_fixed}

    while env_steps < steps_per_batch:
        joint_obs_np = build_joint_obs(agent_ids_fixed, obs_dict, obs_template)
        joint_obs_t = torch.as_tensor(joint_obs_np, dtype=torch.float32, device=device).unsqueeze(0)

        actions = {}
        logps = {}
        values = {}

        for aid in agent_ids_fixed:
            # actor
            obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=device)
            a, logp = agents[aid].act(obs_t)
            actions[aid] = a
            logps[aid] = logp
            # centralized critic: V(s, i)
            with torch.no_grad():
                v_t = critic_forward(central_critic, joint_obs_t, idx_of[aid], device).item()
            values[aid] = v_t

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        env_steps += 1
        running_len += 1
        running_return += sum(float(r) for r in rewards.values())
        dones = {aid: float(terminations.get(aid, False) or truncations.get(aid, False)) for aid in agent_ids_fixed}

        for aid in agent_ids_fixed:
            if buffers[aid].ptr < buffers[aid].max_size:
                rew = float(rewards.get(aid, 0.0))
                buffers[aid].store(
                    obs=np.array(obs_dict[aid], dtype=np.float32),
                    joint_obs=joint_obs_np,
                    agent_id=idx_of[aid],
                    act=actions[aid],
                    rew=rew,
                    val=values[aid],
                    logp=logps[aid],
                    done=dones.get(aid, 1.0)
                )
                active_in_ep[aid] = True

        ep_done = any(terminations.values()) or any(truncations.values())

        if ep_done or env_steps >= steps_per_batch:
            if not ep_done:
                joint_next_np = build_joint_obs(agent_ids_fixed, next_obs, obs_template)
                joint_next_t = torch.as_tensor(joint_next_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                for aid in agent_ids_fixed:
                    if not active_in_ep[aid]:
                        continue  
                    if ep_done:
                        last_val = 0.0
                    else:
                        last_val = critic_forward(central_critic, joint_next_t, idx_of[aid], device).item()
                    buffers[aid].finish_path(last_val=last_val)

            if ep_done:
                ep_returns.append(running_return)
                ep_lengths.append(running_len)

            if ep_done and env_steps < steps_per_batch:
                obs_dict, _ = env.reset()
                running_return, running_len = 0.0, 0
                active_in_ep = {aid: False for aid in agent_ids_fixed}
            else:
                obs_dict = next_obs
                break

        obs_dict = next_obs
        
    return ep_returns, ep_lengths, env_steps

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
    agent_ids = list(env.possible_agents)
    assert len(agent_ids) == players
    scores = []
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done_flags = {aid: False for aid in agent_ids}
        foods_collected = 0
        while True:
            actions = {}
            for aid in agent_ids:
                if done_flags[aid] or (aid not in obs):
                    continue
                obs_t = torch.as_tensor(obs[aid], dtype=torch.float32, device=device)
                a = agents_dict[aid].act_deterministic(obs_t)
                actions[aid] = a
            
            if len(actions) == 0:
                obs, _ = env.reset()
                done_flags = {aid: False for aid in agent_ids}
                continue

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            total_step_reward = sum(float(r) for r in rewards.values())
            if total_step_reward > 0.0:
                foods_collected += 1

            dones = {aid: bool(terminations.get(aid, True) or truncations.get(aid, True)) for aid in actions}
            for aid in agent_ids:
                if dones.get(aid, False) or (aid not in next_obs):
                    done_flags[aid] = True
            
            obs = next_obs
            if all(done_flags.values()):
                break

        scores.append(min(1.0, foods_collected / float(foods)))

    env.close()
    return float(np.mean(scores)), float(np.std(scores))

def train_mappo(
    env_name="lbf_gym",
    total_steps=2000000,
    steps_per_agent_batch=4096,
    grid=(8, 8), 
    players=3, 
    foods=3, 
    coop=False, 
    max_episode_steps=50,
    hidden_actor=(128, 128), 
    hidden_critic=(256, 256),
    lr_actor=3e-4, 
    lr_critic=3e-4,
    clip_ratio=0.2, 
    vf_coef=0.5, 
    ent_coef_start=0.05,     
    ent_coef_end=0.01,       
    ent_coef_decay_ratio=0.1, 
    train_iters=6, 
    minibatch_size=512,
    seed=1, 
    device=None, 
    log_dir="./mappo_logs",
    log_csv_name="mappo_train_log.csv",
    eval_every_steps=50000,
    eval_episodes=100,
    eval_csv_name="mappo_eval.csv"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(log_dir, exist_ok=True)
    log_csv_path = os.path.join(log_dir, f"{env_name}_{log_csv_name}")
    with open(log_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "ep_return", "ep_len"])
    eval_csv_path = os.path.join(log_dir, f"{env_name}_{eval_csv_name}")
    with open(eval_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "normalized_return_mean", "normalized_return_std"])

    env = make_env(env_name, grid, players, foods, coop, max_episode_steps, seed)
    agent_ids = env.possible_agents

    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    joint_obs_dim = obs_dim * len(agent_ids)

    central_critic = CentralCritic(joint_obs_dim, n_agents=len(agent_ids), hidden=hidden_critic).to(device)
    critic_optim = torch.optim.Adam(central_critic.parameters(), lr=lr_critic)

    agents = {}
    buffers = {}
    for idx, aid in enumerate(agent_ids):
        obs_space, act_space = get_spaces(env, aid)
        agents[aid] = MAPPOAgent(
            obs_space, act_space, hidden=hidden_actor,
            lr_actor=lr_actor, clip_ratio=clip_ratio, ent_coef=ent_coef_start,
            train_iters=train_iters, minibatch_size=minibatch_size, device=device
        )
        buffers[aid] = MAPPOBuffer(
            obs_dim=obs_dim, act_is_discrete=True, act_dim=1,
            joint_obs_dim=joint_obs_dim, n_agents=len(agent_ids),
            size=steps_per_agent_batch, gamma=0.99, lam=0.95, device=device
        )

    timer = Timer()
    timer.start()
    metrics = MetricsLogger()
    algo_name = "MAPPO"

    sample_aid = agent_ids[0]
    actor_params_per = count_params(agents[sample_aid].actor)
    critic_params_shared = count_params(central_critic)

    comm_bytes_total = 0
    bytes_per_obs = 4
    per_env_step_comm = len(agent_ids) * obs_dim * bytes_per_obs

    steps_collected = 0
    pbar = tqdm(total=total_steps, desc="Training (MAPPO, LBF)", unit="steps", dynamic_ncols=True, leave=True)

    while steps_collected < total_steps:
        # 1) rollout
        ep_returns, ep_lengths, env_steps = rollout_once(env, agent_ids, agents, central_critic, buffers, steps_per_agent_batch, device)
        batch_steps = steps_per_agent_batch * len(agent_ids)
        steps_collected += batch_steps
        comm_bytes_total += int(env_steps * per_env_step_comm)

        batch_data = {}
        for aid in agent_ids:
            batch_data[aid] = buffers[aid].get(adv_norm=True)

        # 2) update critic (use joint_obs/agent_id/ret)
        decay_steps = int(ent_coef_decay_ratio * total_steps)
        current_ent = linear_anneal(cur_steps=steps_collected, total_decay_steps=decay_steps, start=ent_coef_start, end=ent_coef_end)
        for aid in agent_ids:
            agents[aid].ent_coef = float(current_ent)
        for aid in agent_ids:
            update_central_critic(central_critic, critic_optim, batch_data[aid],
                                  vf_coef=vf_coef, train_iters=train_iters, 
                                  minibatch_size=minibatch_size)
            
        # 3) update actors (respectively, use obs/act/adv/logp)
        def _dummy_values(joint_obs, agent_id): # Interface placeholder
            return None
        for aid in agent_ids:
            agents[aid].update_actor(batch_data[aid], _dummy_values)

        # 4) log
        ep_returns_avg = float(np.mean(ep_returns)) if len(ep_returns) else float("nan")
        ep_lengths_avg = float(np.mean(ep_lengths)) if len(ep_lengths) else float("nan")
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([steps_collected, ep_returns_avg, ep_lengths_avg])

        # 5) Evaluate
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

        pbar.set_postfix_str(f"ep_return={ep_returns_avg}")
        pbar.update(batch_steps)

    timer.stop()
    num_batches = steps_collected // int(steps_per_agent_batch * len(agent_ids))
    metrics.append(
        algo=algo_name, 
        env_name="lbf_gym", 
        total_steps=steps_collected,
        wall_seconds=round(timer.elapsed, 2),
        n_agents=len(agent_ids),
        actor_params_per_agent=actor_params_per,
        critic_params_shared=critic_params_shared,
        steps_per_agent_batch=steps_per_agent_batch,
        train_iters=train_iters, 
        minibatch_size=minibatch_size,
        num_batches=num_batches,
        comm_rounds=0, 
        comm_bytes_total=comm_bytes_total
    )
    print(f"[METRICS] {algo_name}: wall={timer.elapsed:.2f}s, joint_obs upload = {comm_bytes_total / 1e6:.2f} MB")

    pbar.close()
    env.close()
    print(f"[Done] logs saved to {log_csv_path}")
            

if __name__ == "__main__":
    train_mappo(
        env_name="lbf_gym",
        total_steps=2000000,
        steps_per_agent_batch=4096,
        grid=(8, 8), 
        players=3, 
        foods=3, 
        coop=False, 
        max_episode_steps=50,
        hidden_actor=(128, 128), 
        hidden_critic=(256, 256),
        lr_actor=3e-4, 
        lr_critic=3e-4,
        clip_ratio=0.2, 
        vf_coef=0.5, 
        ent_coef_start=0.02,     
        ent_coef_end=0.005,       
        ent_coef_decay_ratio=0.4, 
        train_iters=6, 
        minibatch_size=512,
        seed=1, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="./mappo_logs",
        log_csv_name="mappo_train_log.csv",
        eval_every_steps=50000,
        eval_episodes=100,
        eval_csv_name="mappo_eval.csv"
    )
