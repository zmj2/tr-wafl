import csv, os
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def to_float(x, default=0.0):
    try: return float(x)
    except: return default

def main(metrics_path="./metrics_logs/metrics.csv"):
    rows = load_metrics(metrics_path)
    algos = ["IPPO", "IPPO_WAFL", "MAPPO"]

    agg = {a: {"wall": [], "comm_mb": [], "params": []} for a in algos}
    for r in rows:
        a = r["algo"]
        if a not in agg: continue
        wall = to_float(r["wall_seconds"])
        comm_mb = to_float(r["comm_bytes_total"]) / 1e6

        actor_per = int(float(r["actor_params_per_agent"]))
        n_agents  = int(float(r["n_agents"]))
        critic_sh = int(float(r["critic_params_shared"]))
        params_total = actor_per * n_agents + critic_sh
        agg[a]["wall"].append(wall)
        agg[a]["comm_mb"].append(comm_mb)
        agg[a]["params"].append(params_total)

    def avg(lst): return np.mean(lst) if lst else 0.0
    wall = [avg(agg[a]["wall"]) for a in algos]
    comm = [avg(agg[a]["comm_mb"]) for a in algos]
    params = [avg(agg[a]["params"]) for a in algos]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Training Time
    axes[0].bar(algos, wall, color=colors)
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_title("Training Time")

    # Communication Volume
    axes[1].bar(algos, comm, color=colors)
    axes[1].set_ylabel("Total Communication (MB)")
    axes[1].set_title("Communication Volume")

    # Model Size
    axes[2].bar(algos, np.array(params)/1e6, color=colors)
    axes[2].set_ylabel("Trainable Parameters (Millions)")
    axes[2].set_title("Model Size (Proxy for Complexity)")

    plt.tight_layout()
    plt.savefig("./metrics_logs/efficiency_comparison.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()

