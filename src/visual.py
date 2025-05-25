# src/visual.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_entropy(entropy_track):
    plt.figure(figsize=(7,4))
    sns.lineplot(x=range(len(entropy_track)), y=entropy_track)
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.title("Entropy over Time (Guess 1)")
    plt.tight_layout()

def plot_delta_vs_time(delta_norm_track):
    plt.figure(figsize=(7,4))
    sns.lineplot(x=range(len(delta_norm_track)), y=delta_norm_track)
    plt.xlabel("Step")
    plt.ylabel("Mean |ΔS|")
    plt.title("Mean State Change Magnitude")
    plt.tight_layout()

def scatter_entropy_vs_delta(entropy_track, delta_norm_track):
    df = pd.DataFrame({
        "entropy": entropy_track,
        "delta": delta_norm_track
    })
    plt.figure(figsize=(5,5))
    sns.scatterplot(data=df, x="delta", y="entropy")
    plt.xlabel("|ΔS| (proxy for perceived time)")
    plt.ylabel("Entropy")
    plt.title("Guess-2 Correlation Plot")
    plt.tight_layout()

def _finalize(filename: str):
    import pathlib, os
    pathlib.Path("outputs").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", filename), dpi=150)
    plt.close()