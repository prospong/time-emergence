# experiment_extend_2.py – advanced batch runner + dynamic graph + torch attention
# -----------------------------------------------------------------------------
# 修订版 2025-05-25 🛠️
#   • 新增 --seed、--evolve_interval CLI 选项 + EXT2_SEED 环境变量支持
#   • 单文件批跑（--n_runs）或单轮（--seed）均可
#   • 记录 entropy_slope / entropy_range / spearman_r
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse, json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch, networkx as nx, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import entropy as shannon_entropy

from .utils import new_run_dir, save_config, save_metrics
from .visual import plot_entropy, plot_delta_vs_time, scatter_entropy_vs_delta
from .metrics import pearson_corr, spearman_corr, entropy_slope

# ------------------------------- Config --------------------------------------
@dataclass
class TorchConfig:
    num_agents: int = 200
    dims: int = 16
    steps: int = 1000
    seed: int = 0

    A: float = 0.3
    omega: float = 0.05
    sigma: float = 0.15

    attn_scale: float = 0.2
    layernorm_eps: float = 1e-5

    graph_type: str = "watts_strogatz"
    k_neighbors: int = 8
    rewiring_p: float = 0.1

    evolve_interval: int = 0              # 0 = 不演化
    evolve_prob_add: float = 0.02
    evolve_prob_remove: float = 0.02

    bins: int = 40
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Simulator ---------------------------------------------
class TorchSimulator:
    def __init__(self, cfg: TorchConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
        self.device = torch.device(cfg.device)

        self.G = self._build_graph(cfg)
        self.n = cfg.num_agents
        self.states = torch.rand((self.n, cfg.dims), device=self.device)
        self.phase = torch.rand(self.n, device=self.device) * 2 * np.pi
        self.ln = torch.nn.LayerNorm(cfg.dims, eps=cfg.layernorm_eps).to(self.device)

        self.entropy_track: List[float] = []
        self.delta_track: List[float] = []

    # ---------- graph ----------
    def _build_graph(self, cfg: TorchConfig):
        if cfg.graph_type == "watts_strogatz":
            return nx.watts_strogatz_graph(cfg.num_agents, cfg.k_neighbors, cfg.rewiring_p)
        if cfg.graph_type == "barabasi":
            return nx.barabasi_albert_graph(cfg.num_agents, max(1, cfg.k_neighbors // 2))
        p = cfg.k_neighbors / (cfg.num_agents - 1)
        return nx.erdos_renyi_graph(cfg.num_agents, p)

    # ---------- one step ----------
    def step(self, t: int):
        c = self.cfg
        osc = c.A * torch.sin(c.omega * t + self.phase)[:, None]
        noise = torch.randn_like(self.states) * c.sigma
        delta = osc + noise + c.attn_scale * self._neighbor_attention()
        self.states += delta

        self.delta_track.append(delta.norm(dim=1).mean().item())
        self.entropy_track.append(self._entropy_cpu())

        if c.evolve_interval and t and t % c.evolve_interval == 0:
            self._evolve_graph()

    # ---------- attention ----------
    def _neighbor_attention(self):
        c = self.cfg
        out = torch.zeros_like(self.states)
        q = self.states / (self.states.norm(dim=1, keepdim=True) + 1e-8)
        for i in range(self.n):
            nbrs = list(self.G.neighbors(i))
            if not nbrs: continue
            k = q[nbrs]
            scores = (k @ q[i]).flatten()
            w = torch.softmax(scores * np.sqrt(c.dims), dim=0)
            agg = (w[:, None] * (self.states[nbrs] - self.states[i])).sum(0)
            out[i] = self.ln(agg)
        return out

    # ---------- entropy ----------
    def _entropy_cpu(self):
        c = self.cfg
        x = self.states.detach().cpu().numpy()
        if c.dims <= 6:
            hist, _ = np.histogramdd(x, bins=min(c.bins, 8))
            p = hist.ravel(); p = p[p > 0] / p.sum()
            return float(shannon_entropy(p, base=np.e))
        ent = 0.0
        for d in range(c.dims):
            h, _ = np.histogram(x[:, d], bins=c.bins)
            p = h[h > 0] / h.sum(); ent += float(shannon_entropy(p, base=np.e))
        return ent / c.dims

    # ---------- evolve ----------
    def _evolve_graph(self):
        c = self.cfg
        for u, v in list(self.G.edges()):
            if random.random() < c.evolve_prob_remove:
                self.G.remove_edge(u, v)
        while random.random() < c.evolve_prob_add:
            u, v = random.sample(range(self.n), 2)
            self.G.add_edge(u, v)

    # ---------- run ----------
    def run(self):
        for t in range(self.cfg.steps):
            self.step(t)
        return np.array(self.entropy_track), np.array(self.delta_track)

# ---------------- helpers ----------------
def save_fig(fig_dir: Path, name: str, func, *data):
    fig_dir.mkdir(parents=True, exist_ok=True)
    func(*data)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=150)
    plt.close()

# ---------------- single run -------------
def run_once(cfg: TorchConfig, root: Path):
    """运行一次模拟 → 保存 run 目录 → 返回主要统计量字典"""
    run_dir = new_run_dir(root)
    fig_dir = run_dir / "figs"
    save_config(cfg, run_dir)

    entropy, delta = TorchSimulator(cfg).run()

    # 计算统计量
    metrics = {
        **pearson_corr(delta, entropy),
        **spearman_corr(delta, entropy),
        "entropy_mean":  float(entropy.mean()),
        "entropy_std":   float(entropy.std()),
        "delta_mean":    float(delta.mean()),
        "delta_std":     float(delta.std()),
        "entropy_slope": entropy_slope(entropy),
        "entropy_range": float(entropy.max() - entropy.min()),
    }
    save_metrics(entropy, delta, metrics, run_dir)

    # 画图
    save_fig(fig_dir, "entropy", plot_entropy, entropy)
    save_fig(fig_dir, "delta",   plot_delta_vs_time, delta)
    save_fig(fig_dir, "corr",    scatter_entropy_vs_delta, entropy, delta)

    return metrics | {"run_dir": str(run_dir)}

# ---------------- main --------------------
def main():
    p = argparse.ArgumentParser("Torch Time-Emergence Experiments")
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--out", default="torch_runs")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dims", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--attn_scale", type=float, default=0.2)
    p.add_argument("--evolve_interval", type=int, default=0)   # 新增
    p.add_argument("--seed", type=int)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(exist_ok=True)

    # seeds 列表
    if args.seed is not None:
        seeds = [args.seed]
    elif os.getenv("EXT2_SEED") is not None:
        seeds = [int(os.getenv("EXT2_SEED"))]
    else:
        seeds = list(range(args.n_runs))

    rows = []
    for i, sd in enumerate(seeds):
        cfg = TorchConfig(
            seed=sd,
            steps=args.steps,
            dims=args.dims,
            sigma=args.sigma,
            attn_scale=args.attn_scale,
            evolve_interval=args.evolve_interval,
        )
        res = run_once(cfg, out)
        print(f"RUN {i}: seed={sd}, r={res['pearson_r']:.3f}, p={res['p_value']:.3g}")
        rows.append(res)

    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False)
    agg = df.mean(numeric_only=True).to_dict()
    with open(out / "aggregate.json", "w") as f:
        json.dump({k: float(v) for k, v in agg.items()}, f, indent=2)
    print("Aggregate Pearson r = {:.3f}".format(agg.get("pearson_r", float('nan'))))

if __name__ == "__main__":
    main()
