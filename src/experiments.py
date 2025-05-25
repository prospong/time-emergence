# src/experiments.py
from .config import SimConfig
from .simulator import TimeSimulator
from .metrics import pearson_corr
from .visual import (
    plot_entropy,
    plot_delta_vs_time,
    scatter_entropy_vs_delta,
)
from .utils import new_run_dir, save_config, save_metrics

import matplotlib.pyplot as plt
import os, pathlib

# ---------- 帮助函数：保存并关闭图 ----------
def _save_fig(run_dir, fname):
    plt.tight_layout()
    plt.savefig(run_dir / "figs" / f"{fname}.png", dpi=150)
    plt.close()

def run_single(seed=42, **kwargs):
    # 1) 构建配置 & 生成 run 目录
    cfg = SimConfig(seed=seed, **kwargs)
    run_dir = new_run_dir()
    save_config(cfg, run_dir)

    # 2) 运行仿真
    sim = TimeSimulator(cfg)
    entropy, delta, _ = sim.run()

    # 3) 绘图
    plot_entropy(entropy)             ; _save_fig(run_dir, "entropy")
    plot_delta_vs_time(delta)         ; _save_fig(run_dir, "delta")
    scatter_entropy_vs_delta(entropy, delta)
    _save_fig(run_dir, "corr")

    # 4) 统计 + 保存
    corr = pearson_corr(delta, entropy)
    save_metrics(entropy, delta, corr, run_dir)

    # 5) 控制台简报
    print(f"[RUN SAVED] {run_dir}")
    print(f"Pearson r = {corr['pearson_r']:.3f}  (p={corr['p_value']:.4g})")

if __name__ == "__main__":
    run_single()
