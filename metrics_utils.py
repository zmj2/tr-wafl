import time, csv, os
import torch

def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def vector_numel(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

def vector_nbytes(module: torch.nn.Module, dtype_bytes: int = 4) -> int:
    return vector_numel(module) * dtype_bytes

class Timer:
    def __init__(self):
        self.t0 = None
        self.elapsed = 0.0
    def start(self):
        self.t0 = time.perf_counter()
    def stop(self):
        self.elapsed += time.perf_counter() - self.t0
        self.t0 = None

class MetricsLogger:
    header = ["algo", "env_name", "total_steps", "wall_seconds",
              "n_agents", "actor_params_per_agent", "critic_params_shared",
              "steps_per_agent_batch", "train_iters", "minibatch_size",
              "num_batches", "comm_rounds", "comm_bytes_total"]
    def __init__(self, log_dir="./metrics_logs", fname="metrics.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.fpath = os.path.join(log_dir, fname)
        if not os.path.exists(self.fpath):
            with open(self.fpath, "w", newline="") as f:
                csv.writer(f).writerow(self.header)
    def append(self, **kw):
        row = [kw.get(k, "") for k in self.header]
        with open(self.fpath, "a", newline="") as f:
            csv.writer(f).writerow(row)