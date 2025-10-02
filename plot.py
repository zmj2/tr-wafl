import os
import pandas as pd
import matplotlib.pyplot as plt

def main(log_dir="./ippo_logs", env_name="lbf_gym", algo="ippo_wafl", window=5, save_fig=True):
    csv_path = os.path.join(log_dir, f"{env_name}_{algo}_train_log.csv")
    df = pd.read_csv(csv_path)
    x = df["step"]
    y = df["ep_return"]
    plt.figure(figsize=(9, 5))    
    plt.plot(x, y, label="ep_return")
    y_roll = y.rolling(window=window, min_periods=1).mean()
    plt.plot(x, y_roll, label=f"rolling {window}")
    plt.xlabel("Env Steps")
    plt.ylabel("Episode Return")
    plt.title(f"{algo.upper()} on {env_name}")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        fig_path = os.path.join(log_dir, f"{env_name}_{algo}_curve.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Saved figure to {fig_path}")
    plt.show()

if __name__ == "__main__":
    main(log_dir="./ippo_logs", algo="ippo_wafl")
