import numpy as np
import torch

class TrajectoryBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device="cpu", hidden_dim=None):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32) if act_dim > 1 else np.zeros((size,), dtype=np.int64)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.hidden_dim = hidden_dim
        if hidden_dim is not None:
            self.h_buf = np.zeros((size, 1, 1, hidden_dim), dtype=np.float32)
        else:
            self.h_buf = None

        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start = 0
        self.max_size = size
        self.device = device

    def store(self, obs, act, rew, val, logp, done, h=None):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done   

        if self.h_buf is not None:
            if h is None:
                self.h_buf[self.ptr] = 0.0
            else:
                if isinstance(h, torch.Tensor):
                    h_np = h.detach().to("cpu").numpy()
                else:
                    h_np = np.asarray(h, dtype=np.float32)
                if h_np.ndim == 1:
                    h_np = h_np.reshape(1, 1, -1)
                elif h_np.ndim == 2:
                    h_np = h_np.reshape(1, 1, h_np.shape[-1])
                self.h_buf[self.ptr] = h_np

        self.ptr += 1

    def finish_path(self, last_val=0.0):
        path_slice = slice(self.path_start, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        dones = np.append(self.done_buf[path_slice], 1.0)

        # delta_t = r_t + γ V(s_{t+1}) - V(s_t)
        deltas = rews[:-1] + self.gamma * vals[1:] * (1.0 - dones[:-1]) - vals[:-1]
        adv = np.zeros_like(deltas)
        gae = 0.0

        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * (1.0 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals[:-1]
        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = ret
        self.path_start = self.ptr

    def get(self, adv_norm=True):
        assert self.ptr == self.max_size, "Buffer not full when get() called"
        obs = torch.as_tensor(self.obs_buf, device=self.device, dtype=torch.float32)
        if len(self.act_buf.shape) == 1:
            act = torch.as_tensor(self.act_buf, device=self.device, dtype=torch.long)
        else:
            act = torch.as_tensor(self.act_buf, device=self.device, dtype=torch.float32)
        ret = torch.as_tensor(self.ret_buf, device=self.device, dtype=torch.float32)
        adv = torch.as_tensor(self.adv_buf, device=self.device, dtype=torch.float32)
        logp = torch.as_tensor(self.logp_buf, device=self.device, dtype=torch.float32)
        done = torch.as_tensor(self.done_buf, device=self.device, dtype=torch.float32)

        if adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        data = dict(obs=obs, act=act, ret=ret, adv=adv, logp=logp, done=done)

        if self.h_buf is not None:
            h = torch.as_tensor(self.h_buf, device=self.device, dtype=torch.float32)  # [N,1,1,H]
            data["h"] = h
        
        self.ptr = 0
        self.path_start = 0
        return data
    
class MAPPOBuffer:
    def __init__(self, obs_dim, act_is_discrete, act_dim, joint_obs_dim, n_agents, 
                 size, gamma=0.99, lam=0.95, device="cpu"):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.joint_obs_buf = np.zeros((size, joint_obs_dim), dtype=np.float32)
        self.agent_id_buf = np.zeros((size,), dtype=np.int64)
        self.act_buf = np.zeros((size, ), dtype=np.int64) if act_is_discrete else np.zeros((size, act_dim), dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)

        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start = 0
        self.max_size = size
        self.device = device

    def store(self, obs, joint_obs, agent_id, act, rew, val, logp, done):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.joint_obs_buf[self.ptr] = joint_obs
        self.agent_id_buf[self.ptr] = agent_id
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        path_slice = slice(self.path_start, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        dones = np.append(self.done_buf[path_slice], 1.0)
        deltas = rews[:-1] + self.gamma * vals[1:] * (1.0 - dones[:-1]) - vals[:-1]
        adv = np.zeros_like(deltas, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lam * (1.0 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals[:-1]
        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = ret
        self.path_start = self.ptr

    def get(self, adv_norm=True):
        assert self.ptr == self.max_size, "Buffer not full when get() called"
        to_tensor = lambda x, dt=torch.float32: torch.as_tensor(x, device=self.device)
        obs = to_tensor(self.obs_buf)
        joint_obs = to_tensor(self.joint_obs_buf)
        agent_id = torch.as_tensor(self.agent_id_buf, device=self.device, dtype=torch.long)

        if len(self.act_buf.shape) == 1:
            act = torch.as_tensor(self.act_buf, device=self.device, dtype=torch.long)
        else:
            act = to_tensor(self.act_buf)

        ret = to_tensor(self.ret_buf)
        adv = to_tensor(self.adv_buf)
        logp = to_tensor(self.logp_buf)
        if adv_norm:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = dict(obs=obs, joint_obs=joint_obs, agent_id=agent_id, act=act, ret=ret, adv=adv, logp=logp)
        self.ptr = 0
        self.path_start = 0
        return data
