import torch
import torch.nn as nn
from torch.distributions import Categorical

class GRUEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x, h=None):
        # x: [B, T, obs_dim] or [B, obs_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, 1, obs]
        z = self.fc(x) # [B, T, H]
        z, h_n = self.gru(z, h) # h_n: [1, B, H]
        return z, h_n
    
class DiscreteActorHead(nn.Module):
    def __init__(self, hidden_dim, n_act):
        super().__init__()
        self.head = nn.Linear(hidden_dim, n_act)

    def forward(self, z): # [B, T, H]
        logits = self.head(z) # [B, T, A]
        dist = Categorical(logits=logits.squeeze(1)) # when T == 1 -> [B, A]
        return dist
    
class ValueHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, z): # [B, T, H]
        v = self.v(z).squeeze(-1).squeeze(1) # when T == 1 -> [B]
        return v
    
class ActorCriticGRU(nn.Module):
    def __init__(self, obs_space, act_space, hidden_dim=128, device="cpu"):
        super().__init__()
        assert len(obs_space.shape) == 1, "only 1D obs"
        self.obs_dim = obs_space.shape[0]
        assert hasattr(act_space, "n"), "LBF uses discrete actions"
        self.n_act = act_space.n
        self.enc = GRUEncoder(self.obs_dim, hidden_dim)
        self.actor = DiscreteActorHead(hidden_dim, self.n_act)
        self.critic = ValueHead(hidden_dim)
        self.device = device
        self.to(device)

    @torch.no_grad()
    def act(self, obs, h=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z, h_next = self.enc(obs, h)
        dist = self.actor(z)
        a = dist.sample()
        logp = dist.log_prob(a)
        v = self.critic(z)
        return int(a.item()), float(logp.item()), float(v.item()), h_next
    
    @torch.no_grad()
    def value(self, obs, h=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        z, _ = self.enc(obs, h)
        v = self.critic(z)
        if v.numel() == 1:
            return float(v.item())
        return v
    
    def evaluate_actions(self, obs_seq, act_seq, h0=None):
        x = obs_seq.unsqueeze(0)
        z, h_last = self.enc(x, h0)
        logits = self.actor.head(z).squeeze(0)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_seq)
        ent = dist.entropy().mean()
        v = self.critic.v(z).squeeze(-1).squeeze(0)
        return logp, v, ent, h_last