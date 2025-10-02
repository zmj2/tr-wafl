import torch
import numpy as np
import torch.nn as nn

def _cosine_sim(a: torch.Tensor, b: torch.Tensor):
    an = a.norm() + 1e-8
    bn = b.norm() + 1e-8
    return (a @ b) / (an * bn)

def _neg_euclid(a: torch.Tensor, b: torch.Tensor):
    return -torch.norm(a - b, p=2)

@torch.no_grad()
def get_policy_dist(agent, obs_batch):
    ac = agent.ac
    obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=agent.device)
    if hasattr(agent, "use_gru") and agent.use_gru:
        z, _ = ac.enc(obs_t, None)
        if z.dim() == 3 and z.size(1) == 1:
            z = z.squeeze(1)
        logits = ac.actor.head(z) 
        return torch.distributions.Categorical(logits=logits)
    else:
        return ac.actor(obs_t)  
    
@torch.no_grad()
def _policy_kl_agents(agent_a, agent_b, obs_batch):
    pi_a = get_policy_dist(agent_a, obs_batch)
    pi_b = get_policy_dist(agent_b, obs_batch)
    return float(torch.distributions.kl_divergence(pi_a, pi_b).mean().cpu())

def build_wafl_weights(deltas: dict, metric="cosine", topk=None, self_weight=0.0):
    """
    deltas: {aid: Δθ_i (1D tensor on CPU)}
    return: {aid: (neigh_ids, neigh_weights)} --- for each i give a set of neighbors and normalized weights
    """
    aids = list(deltas.keys())
    S = np.zeros((len(aids), len(aids)), dtype=np.float64)
    for i, ai in enumerate(aids):
        for j, aj in enumerate(aids):
            if i == j:
                S[i, j] = 0.0
            else:
                if metric == "cosine":
                    S[i, j] = float(_cosine_sim(deltas[ai], deltas[aj]))
                else:
                    S[i, j] = float(_neg_euclid(deltas[ai], deltas[aj]))

    # for each i choose top-k neighbors and do softmax
    out = {}
    k = min(topk if topk is not None else len(aids) - 1, len(aids) - 1)
    for i, ai in enumerate(aids):
        idx = np.argsort(-S[i])
        idx = [j for j in idx if j != i][:k]
        scores = np.array([S[i, j] for j in idx], dtype=np.float64)
        if scores.size == 0:
            neigh_w = np.array([], dtype=np.float64)
        else:
            m = scores.max()
            exps = np.exp(scores - m)
            neigh_w = exps / (exps.sum() + 1e-12)

        neigh_w = (1.0 - self_weight) * neigh_w
        out[ai] = ([aids[j] for j in idx], neigh_w)
    return out

def wafl_fuse_params(param_vectors: dict, weights_dict: dict):
    """
    params_vectors: {aid: θ_j vector (1D tensor, CPU)}
    weights_dict: {aid: (neigh_ids, neigh_weights)}
    return: {aid: fused_vec_i = Σ_j w_ij θ_j}
    """
    fused = {}
    for ai, (neigh_ids, neigh_w) in weights_dict.items():
        if len(neigh_ids) == 0:
            fused[ai] = param_vectors[ai].clone()
        else:
            acc = torch.zeros_like(param_vectors[ai])
            for nid, w in zip(neigh_ids, neigh_w):
                acc += float(w) * param_vectors[nid]
            fused[ai] = acc
    return fused

def build_wafl_weights_policy(agents, calib_obs_by_aid, *, tau=2.0, topk=None, self_weight=0.0, device="cpu"):
    aids = list(agents.keys())
    n = len(aids)
    S = np.zeros((n, n), dtype=np.float64)
    for i, ai in enumerate(aids):
        for j, aj in enumerate(aids):
            if i == j:
                S[i, j] = 0.0
            else:
                obs_i = calib_obs_by_aid[ai]
                k1 = _policy_kl_agents(agents[ai], agents[aj], obs_i)
                k2 = _policy_kl_agents(agents[aj], agents[ai], obs_i)
                S[i, j] = - 0.5 * (k1 + k2)
    
    k = min(topk if topk is not None else n - 1, n - 1)
    out = {}
    for i, ai in enumerate(aids):
        idx_sorted = np.argsort(-S[i])
        neigh_idx = [j for j in idx_sorted if j != i][:k]
        scores = np.array([S[i, j] for j in neigh_idx], dtype=np.float64)
        if scores.size == 0:
            neigh_w = np.array([], dtype=np.float64)
        else:
            exps = np.exp(scores / max(1e-9, float(tau)))
            neigh_w = exps / (exps.sum() + 1e-12)
        self_w = float(np.clip(self_weight, 0.0, 1.0))
        neigh_w = (1.0 - self_w) * neigh_w
        out[ai] = ([aids[j] for j in neigh_idx], neigh_w, self_w)
    return out

def wafl_fuse_params_with_self(param_vectors: dict, weights_dict: dict):
    # fused_i = self_w*θ_i + Σ_j w_ij θ_j
    fused = {}
    for ai, (neigh_ids, neigh_w, self_w) in weights_dict.items():
        base = self_w * param_vectors[ai]
        for nid, w in zip(neigh_ids, neigh_w):
            base += float(w) * param_vectors[nid]
        fused[ai] = base
    return fused

def sample_calib_from_buffer(buf, n=512):
    m = min(n, int(buf.ptr)) if hasattr(buf, "ptr") else n
    if m <= 0:
        return np.zeros((32, buf.obs_buf.shape[-1]), dtype=np.float32)
    idx = np.random.choice(int(buf.ptr), size=m, replace=False)
    return buf.obs_buf[idx]
