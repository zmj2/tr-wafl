import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)

class DiscreteActor(nn.Module):
    def __init__(self, obs_dim, n_act, hidden=(128, 128)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_act])

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)
    
class ContinuousActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128)):
        super().__init__()
        self.mu_net = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=(128,128)):
        super().__init__()
        self.v_net = mlp([obs_dim, *hidden, 1])

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1)
    
class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space, hidden=(128, 128), device="cpu"):
        super().__init__()
        self.device = device
        if len(obs_space.shape) == 1:
            self.obs_dim = obs_space.shape[0]
        else:
            raise NotImplementedError("Only 1D obs supported for LBF vector obs.")
        if hasattr(act_space, "n"):
            self.discrete = True
            self.n_act = act_space.n
            self.actor = DiscreteActor(self.obs_dim, self.n_act, hidden)
        else:
            self.discrete = False
            self.act_dim = act_space.shape[0]
            self.low = torch.as_tensor(act_space.low, device=device)
            self.high = torch.as_tensor(act_space.high, device=device, dtype=torch.float32)
            self.actor = ContinuousActor(self.obs_dim, self.act_dim, hidden)
        self.critic = Critic(self.obs_dim, hidden)
        self.to(device)

    @torch.no_grad()
    def act(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        pi = self.actor(obs)
        if self.discrete:
            a = pi.sample()
            logp = pi.log_prob(a)
            v = self.critic(obs)
            return a.squeeze(0).item(), logp.squeeze(0).item(), v.squeeze(0).item()
        else:
            a = pi.sample()
            logp = pi.log_prob(a).sum(-1)
            v = self.critic(obs)
            a_tanh = torch.tanh(a)
            act_scaled = (self.high - self.low) / 2 * (a_tanh + 1) + self.low
            return act_scaled.squeeze(0).cpu().numpy(), logp.squeeze(0).item(), v.squeeze(0).item()
        
    def evaluate_actions(self, obs, act):
        pi = self.actor(obs)
        v = self.critic(obs)
        if self.discrete:
            logp = pi.log_prob(act)
            entropy = pi.entropy().mean()
        else:
            logp = pi.log_prob(act).sum(-1)
            entropy = pi.entropy().sum(-1).mean()
        return logp, v, entropy
    
class CentralCritic(nn.Module):
    """
    V(s, i) --- input joint_obs (concat all agents' obs)
    output the state value of the agent
    share one critic, shared by all agents
    """
    def __init__(self, joint_obs_dim, n_agents, hidden=(256, 256), use_agent_id=True):
        super().__init__()
        self.use_agent_id = use_agent_id
        in_dim = joint_obs_dim + (n_agents if use_agent_id else 0)
        self.v_net = mlp([in_dim, *hidden, 1])

    def forward(self, joint_obs, agent_id=None):
        """
        joint_obs: [B, joint_obs_dim]
        agent_id: [B]'s integer id, or one-hot [B, n_agents]        
        """
        if self.use_agent_id:
            if agent_id is None:
                raise ValueError("CentralCritic expects agent_id when use_agent_id=True")
            if agent_id.dim() == 1:
                B = agent_id.shape[0]
                n_agents = self.v_net[0].in_features - joint_obs.shape[1]
                onehot = torch.zeros(B, n_agents, device=joint_obs.device)
                onehot.scatter_(1, agent_id.view(-1, 1), 1.0)
            else:
                onehot = agent_id
            x = torch.cat([joint_obs, onehot], dim=-1)
        else:
            x = joint_obs
        return self.v_net(x).squeeze(-1)
    

