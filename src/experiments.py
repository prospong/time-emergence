# src/experiments.py
from .config import SimConfig
from .simulator import TimeSimulator
from .metrics import pearson_corr
from .visual import (
    plot_entropy,
    plot_delta_vs_time,
    scatter_entropy_vs_delta,
)                       # 如果 visual.py 里实现了 _finalize 就把它也加进来
import matplotlib.pyplot as plt
import pathlib, os

# --- 可选：简单保存 PNG 的小工具 ---
def _finalize(fname: str):
    pathlib.Path("outputs").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", f"{fname}.png"), dpi=150)
    plt.close()

def run_single(seed=42, **kwargs):
    cfg = SimConfig(seed=seed, **kwargs)
    sim = TimeSimulator(cfg)
    entropy, delta, _ = sim.run()   # <-- 这里就拿到两个 ndarray

    # 画图 + 保存
    plot_entropy(entropy)
    _finalize("entropy_curve")

    plot_delta_vs_time(delta)
    _finalize("delta_curve")

    scatter_entropy_vs_delta(entropy, delta)
    _finalize("entropy_vs_delta")

    # Pearson 相关
    corr = pearson_corr(delta, entropy)
    print(f"Pearson r = {corr['pearson_r']:.3f}  (p={corr['p_value']:.4g})")

if __name__ == "__main__":
    run_single()

