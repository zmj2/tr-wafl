import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ppo_fc import ActorCritic
from ppo_rnn import ActorCriticGRU

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import BatchSampler, SubsetRandomSampler


def to_hidden_dim(hidden, default=128):
    if hidden is None:
        return default
    if isinstance(hidden, int):
        return hidden
    if isinstance(hidden, (tuple, list)) and len(hidden) > 0:
        return int(hidden[0])
    return default

class IPPOAgent:
    def __init__(self, obs_space, act_space, hidden=(128, 128), use_gru=False,
                 lr=3e-4, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01,
                 train_iters=4, batch_size=2048, minibatch_size=256, device="cpu"):
        if use_gru:
            H = to_hidden_dim(hidden, 128)
            self.ac = ActorCriticGRU(obs_space, act_space, hidden_dim=H, device=device)
            self.hidden_dim = H
        else:
            self.ac = ActorCritic(obs_space, act_space, hidden)
        self.h = None
        self.use_gru = use_gru
        self.optim = optim.Adam(self.ac.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device

    @torch.no_grad()
    def act(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if self.use_gru:
            a, logp, v, h_next = self.ac.act(obs, self.h)
            self.h = h_next
            return a, logp, v
        else:
            return self.ac.act(obs)

    def reset_hidden(self):
        if self.use_gru:
            self.h = None
    
    @torch.no_grad()
    def act_deterministic(self, obs_t):
        if not torch.is_tensor(obs_t):
            obs_t = torch.as_tensor(obs_t, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if self.use_gru:
            z, _ = self.ac.enc(obs_t, self.h)
            logits = self.ac.actor.head(z).squeeze(0)
            a = torch.argmax(logits, dim=-1)
            return int(a.item())
        else:
            pi = self.ac.actor(obs_t)                    
            if getattr(self.ac, "discrete", False):
                a = torch.argmax(pi.probs, dim=-1)
                return int(a.item())
            else:
                a = torch.tanh(pi.mean)
                low, high = self.ac.low, self.ac.high
                a = (high - low) / 2 * (a + 1) + low
                return a.squeeze(0).cpu().numpy()

    def update(self, data):
        # data: dict(obs, act, ret, adv, logp) — tensor
        device = self.device
        obs = data["obs"]
        act = data["act"]
        ret = data["ret"]
        adv = data["adv"]
        logp_old = data["logp"]

        if self.use_gru:
            done = data["done"]
            has_h = ("h" in data) and (data["h"] is not None)

            N = obs.shape[0]
            segs = []
            i = 0
            while i < N:
                j = i
                while j < N - 1 and done[j].item() == 0.0:
                    j += 1
                segs.append((i, j))
                i = j + 1
            lengths = [e - s + 1 for s, e in segs]
            order = np.argsort(lengths)[::-1]
            segs = [segs[k] for k in order]
            lengths = [lengths[k] for k in order]
            B = len(segs)
            T_max = lengths[0]
            # pad
            obs_pad = torch.zeros((B, T_max,obs.shape[-1]), device=device)
            act_pad = torch.zeros((B, T_max), dtype=torch.long, device=device)
            ret_pad = torch.zeros((B, T_max), device=device)
            adv_pad = torch.zeros((B, T_max), device=device)
            lpo_pad = torch.zeros((B, T_max), device=device)
            h0_batch = None
            if has_h:
                H = data["h"].shape[-1]
                h0_batch = torch.zeros((1, B, H), device=device)
            for b, (s, e) in enumerate(segs):
                T = e - s + 1
                obs_pad[b,: T] = obs[s: e + 1]
                act_pad[b,: T] = act[s: e + 1]
                ret_pad[b,: T] = ret[s: e + 1]; 
                adv_pad[b,: T] = adv[s: e + 1]
                lpo_pad[b,: T] = logp_old[s: e + 1]
                if has_h:
                    h0_batch[:,b: b + 1,:] = data["h"][s].to(device)
            # pack
            packed = pack_padded_sequence(obs_pad, lengths, batch_first=True, enforce_sorted=True)
            z_mlp = self.ac.enc.fc(packed.data)
            z_mlp_packed = torch.nn.utils.rnn.PackedSequence(
                data=z_mlp, batch_sizes=packed.batch_sizes,
                sorted_indices=packed.sorted_indices, unsorted_indices=packed.unsorted_indices
            )
            z_packed, _ = self.ac.enc.gru(z_mlp_packed, h0_batch)
            z, _ = pad_packed_sequence(z_packed, batch_first=True)  # [B,T,H]
            logits = self.ac.actor.head(z)                          # [B,T,A]
            dist = torch.distributions.Categorical(logits=logits)
            mask = torch.arange(T_max, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
            logp_new = dist.log_prob(act_pad)                       # [B,T]
            ent = (dist.entropy() * mask).sum() / mask.sum()
            v_pred = self.ac.critic.v(z).squeeze(-1)                # [B,T]

            ratio = torch.exp(logp_new - lpo_pad)
            surr1 = ratio * adv_pad
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_pad
            policy_loss = - torch.where(mask, torch.min(surr1, surr2), torch.zeros_like(surr1)).sum() / mask.sum()
            value_loss = torch.where(mask, (v_pred - ret_pad) ** 2, torch.zeros_like(ret_pad)).sum() / mask.sum()
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * ent

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.optim.step()

        else:
            n = obs.shape[0] 
            idxs = list(range(n)) 
            for _ in range(self.train_iters): 
                for mb_idx in BatchSampler(SubsetRandomSampler(idxs), self.minibatch_size, drop_last=False): 
                    mb_obs = obs[mb_idx] 
                    mb_act = act[mb_idx] 
                    mb_ret = ret[mb_idx] 
                    mb_adv = adv[mb_idx] 
                    mb_logp_old = logp_old[mb_idx] 
                    new_logp, v, entropy = self.ac.evaluate_actions(mb_obs, mb_act) 
                    ratio = torch.exp(new_logp - mb_logp_old) 
                    surr1 = ratio * mb_adv 
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv 
                    policy_loss = - torch.min(surr1, surr2).mean() 
                    value_loss = nn.MSELoss()(v, mb_ret) 
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy 
                    self.optim.zero_grad(set_to_none=True) 
                    loss.backward() 
                    nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5) 
                    self.optim.step()


    def _flatten_params(self):
        vecs = []
        for p in self.ac.parameters():
            vecs.append(p.data.view(-1))
        return torch.cat(vecs)
    
    def _assign_from_vector(self, vec):
        offset = 0
        with torch.no_grad():
            for p in self.ac.parameters():
                num = p.numel()
                p.copy_(vec[offset: offset + num].view_as(p))
                offset += num

    def snapshot_params(self):
        self._last_sync_vec = self._flatten_params().detach().cpu()

    def get_param_delta(self):
        cur = self._flatten_params().detach().cpu()
        last = getattr(self, "_last_sync_vec", None)
        if last is None:
            return cur.clone() * 0.0
        return cur - last
    
    def apply_personalized_fusion(self, fused_vec, alpha=0.3):
        """
        WAFL: θ_i ← (1-α)θ_i + α * fused_vec
        which fused_vec means neighbor weighted model vector
        """
        cur = self._flatten_params().detach().cpu()
        new = (1 - alpha) * cur + alpha * fused_vec.cpu()
        self._assign_from_vector(new.to(self.device))

    @staticmethod
    @torch.no_grad()
    def params_to_vec(params):
        return torch.cat([p.data.view(-1) for p in params])
    
    @staticmethod
    @torch.no_grad()
    def assign_from_vec(params, vec):
        off = 0
        for p in params:
            n = p.numel()
            p.copy_(vec[off:off+n].view_as(p))
            off += n

    def split_trunk_head(self):
        """
        return: (trunk_params, head_params)
        - GRU: trunk = enc.{fc, gru}, head = actor.head
        - FC:  trunk = actor.net 
        """
        ac = self.ac
        if hasattr(ac, "enc") and hasattr(ac, "actor") and hasattr(ac.actor, "head"):
            trunk, head = [], []
            if hasattr(ac.enc, "fc"):  
                trunk += list(ac.enc.fc.parameters())
            if hasattr(ac.enc, "gru"): 
                trunk += list(ac.enc.gru.parameters())
            head = list(ac.actor.head.parameters())
            if len(trunk) == 0:
                trunk = list(ac.parameters())
                head = []
            return trunk, head

        if hasattr(ac, "actor") and hasattr(ac.actor, "net"):
            seq = ac.actor.net
            linears = [m for m in seq.modules() if isinstance(m, nn.Linear)]
            if len(linears) >= 1:
                last = linears[-1]
                head = [last.weight] + ([last.bias] if last.bias is not None else [])
                head_ids = set(id(p) for p in head)
                trunk = [p for p in seq.parameters() if id(p) not in head_ids]
                if len(trunk) == 0:
                    trunk = list(seq.parameters())
                    head = []
                return trunk, head

        return list(ac.parameters()), []
    
    @torch.no_grad()
    def policy_dist(self, obs_batch):
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        if getattr(self, "use_gru", False):
            z, _ = self.ac.enc(obs_t, None)         # z: [B,1,H] or [B,H]
            if z.dim() == 3 and z.size(1) == 1:
                z = z.squeeze(1)
            logits = self.ac.actor.head(z)          # [B,A]
            return torch.distributions.Categorical(logits=logits)
        else:
            return self.ac.actor(obs_t)    
            
    @torch.no_grad()
    def trunk_vector(self):
        trunk, _ = self.split_trunk_head()
        return self.params_to_vec(trunk).detach().cpu()
    
    @torch.no_grad()
    def assign_trunk_from_vec(self, vec_cpu):
        trunk, _ = self.split_trunk_head()
        self.assign_from_vec(trunk, vec_cpu.to(self.device))

    def trust_region_fuse_trunk(self, target_trunk_vec_cpu, calib_obs, eps_fuse=0.01, alpha_max=0.5, steps=10):
        trunk, _ = self.split_trunk_head()
        cur = self.params_to_vec(trunk).detach().clone()
        direction = target_trunk_vec_cpu.to(cur.device) - cur.cpu()
        direction = direction.to(self.device)

        pi_old = self.policy_dist(calib_obs)

        low = 0.0
        high = float(alpha_max)
        for _ in range(steps):
            alpha = 0.5 * (low + high)
            cand = cur.to(self.device) + alpha * direction
            self.assign_from_vec(trunk, cand)
            pi_new = self.policy_dist(calib_obs)
            kl = torch.distributions.kl_divergence(pi_old, pi_new).mean().item()
            if kl <= eps_fuse:
                low = alpha
            else:
                high = alpha
        
        final = cur.to(self.device) + low * direction
        self.assign_from_vec(trunk, final)
        return low
    
    def clear_optimizer_state(self):
        if hasattr(self, "optim") and hasattr(self.optim, "state"):
            self.optim.state.clear()