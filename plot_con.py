import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_curve(csv_path):
    df = pd.read_csv(csv_path)
    ret = df["ep_return"]
    steps = df["step"]
    return steps, ret

fig_path = "lbf_gym_results.png"
ippo_csv = "./ippo_logs/lbf_gym_ippo_train_log.csv"
ippo_wafl_csv = "./ippo_logs/lbf_gym_ippo_wafl_train_log.csv"
mappo_csv = "./mappo_logs/lbf_gym_mappo_train_log.csv"
window = 5

steps_ippo, ippo = load_curve(ippo_csv)
steps_ippo_wafl, ippo_wafl = load_curve(ippo_wafl_csv)
steps_mappo, mappo = load_curve(mappo_csv)

ippo_roll = ippo.rolling(window=window, min_periods=1).mean()
ippo_wafl_roll = ippo_wafl.rolling(window=window, min_periods=1).mean()
mappo_roll = mappo.rolling(window=window, min_periods=1).mean()

plt.figure(figsize=(9, 5))
plt.plot(steps_ippo, ippo_roll, label="IPPO", color="blue")
plt.plot(steps_ippo_wafl, ippo_wafl_roll, label="IPPO+WAFL", color="green")
plt.plot(steps_mappo, mappo_roll, label="MAPPO", color="red")
plt.xlabel("Env Steps")
plt.ylabel("Mean Episode Return")
plt.title("IPPO vs MAPPO vs IPPO+WAFL on LBF")
plt.legend()
plt.tight_layout()
plt.savefig(fig_path, dpi=150)
print(f"Saved figure to {fig_path}")
plt.show()

