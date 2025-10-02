import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import BatchSampler, SubsetRandomSampler

from ppo_fc import DiscreteActor

class MAPPOAgent:
    def __init__(self, obs_space, act_space, hidden=(128, 128),
                 lr_actor=3e-4, clip_ratio=0.2, ent_coef=0.01,
                 train_iters=4, minibatch_size=256, device="cpu"):
        self.device = device
        self.discrete = hasattr(act_space, "n")
        self.n_act = act_space.n if self.discrete else act_space.shape[0]
        obs_dim = obs_space.shape[0]
        self.actor = DiscreteActor(obs_dim, self.n_act, hidden).to(device)
        self.optim = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.train_iters = train_iters
        self.minibatch_size = minibatch_size

    @torch.no_grad()
    def act(self, obs_t):
        pi = self.actor(obs_t.unsqueeze(0) if obs_t.dim() == 1 else obs_t)
        a = pi.sample()
        logp = pi.log_prob(a)
        if a.dim() > 1: 
            a = a.squeeze(0)
        if logp.dim() > 1: 
            logp = logp.squeeze(0)
        return a.item(), logp.item()
    
    @torch.no_grad()
    def act_deterministic(self, obs_t: torch.Tensor) -> int:
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        pi = self.actor(obs_t)
        return int(torch.argmax(pi.logits, dim=-1).item())
    
    def update_actor(self, data, get_values_fn):
        """
        data: dict with obs, act, adv, logp; value targets via get_values_fn
        get_values_fn: callable(joint_obs, agent_id) -> v
        """
        obs = data["obs"]
        act = data["act"]
        adv = data["adv"]
        logp_old = data["logp"]

        n = obs.shape[0]
        idxs = list(range(n))

        for _ in range(self.train_iters):
            for mb_idx in BatchSampler(SubsetRandomSampler(idxs), self.minibatch_size, drop_last=False):
                pi = self.actor(obs[mb_idx])
                new_logp = pi.log_prob(act[mb_idx])
                ratio = torch.exp(new_logp - logp_old[mb_idx])
                surr1 = ratio * adv[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy = pi.entropy().mean()
                loss = policy_loss - self.ent_coef * entropy
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optim.step()

def update_central_critic(central_critic, critic_optim, data, vf_coef=0.5, train_iters=4, minibatch_size=256):
    joint_obs = data["joint_obs"]
    agent_id = data["agent_id"]
    ret = data["ret"]
    n = joint_obs.shape[0]
    idxs = list(range(n))
    for _ in range(train_iters):
        for mb_idx in BatchSampler(SubsetRandomSampler(idxs), minibatch_size, drop_last=False):
            v = central_critic(joint_obs[mb_idx], agent_id[mb_idx])
            value_loss = nn.MSELoss()(v, ret[mb_idx])
            critic_optim.zero_grad(set_to_none=True)
            (vf_coef * value_loss).backward()
            nn.utils.clip_grad_norm_(central_critic.parameters(), 0.5)
            critic_optim.step()