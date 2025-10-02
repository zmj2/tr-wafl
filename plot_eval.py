import csv
import numpy as np
import matplotlib.pyplot as plt

def load_eval_csv(path):
    steps, means, stds = [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            steps.append(int(r["step"]))
            means.append(float(r["normalized_return_mean"]))
            stds.append(float(r["normalized_return_std"]))
    return np.array(steps), np.array(means), np.array(stds)

def ema(y, alpha=0.2):
    if len(y) == 0: return y
    out = np.zeros_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

if __name__ == "__main__":
    ippo_eval = "./ippo_logs/lbf_gym_ippo_eval.csv"
    wafl_eval = "./ippo_logs/lbf_gym_ippo_wafl_eval.csv"
    mappo_eval = "./mappo_logs/lbf_gym_mappo_eval.csv"

    curves = []
    eval_list = [(ippo_eval, "IPPO"), (wafl_eval, "IPPO + WAFL"), (mappo_eval, "MAPPO")]
    # eval_list = [(wafl_eval, "IPPO + WAFL"), (mappo_eval, "MAPPO")]
    for p, label in eval_list:
        try:
            s, m, sd = load_eval_csv(p)
            curves.append((label, s, m, sd))
        except Exception:
            pass

    plt.figure(figsize=(8, 5))
    for label, s, m, sd in curves:
        m_smooth = ema(m, alpha=0.25)
        plt.plot(s, m_smooth, label=label)
        plt.fill_between(s, m - sd, m + sd, alpha=0.15)

    # plt.ylim(0, 1.0)
    plt.xlabel("Env Steps")
    plt.ylabel("Normalized Return (0-1)")
    plt.title("Evaluation: Normalized Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lbf_gym_eval_results.png", dpi=150)
    plt.show()
