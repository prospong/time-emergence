# experiment_extent.py – extended experiment runner with graph-based agents + attention
# -----------------------------------------------------------------------------
# USAGE (inside repo root):
#   python -m src.experiment_extent  # identical pattern to previous runner
# -----------------------------------------------------------------------------
"""Run an extended simulation where each agent is a node in a complex network
and state updates are influenced by an attention-weighted aggregation of 
neighbor states in addition to deterministic oscillation and random noise.

Key differences vs. the basic model:
1. Agents live on a NetworkX graph (default: Watts–Strogatz small‑world).
2. Each agent attends to its neighbors – we compute attention scores as a
   softmax over cosine similarity between current state vectors.
3. The aggregated neighbor signal is added to the ΔS update.
4. All outputs are saved in a dedicated run folder (reuse utils.py helpers).

The file is self‑contained – no changes are required in the existing core files
except having NetworkX in the environment (add to requirements.txt).
"""

from __future__ import annotations
from dataclasses import dataclass


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from .config import SimConfig
from .utils import new_run_dir, save_config, save_metrics
from .metrics import pearson_corr
from .visual import (
    plot_entropy,
    plot_delta_vs_time,
    scatter_entropy_vs_delta,
)
from scipy.stats import entropy as shannon_entropy

# --------------------------- EXTENDED CONFIG ---------------------------------
@dataclass
class ExtConfig(SimConfig):
    """Add graph + attention parameters on top of SimConfig."""

    # Graph topology
    graph_type: str = "watts_strogatz"  # or "barabasi" / "erdos_renyi"
    k_neighbors: int = 6                 # Watts‑Strogatz: each node connected to k nearest neighbours
    rewiring_p: float = 0.1              # WS rewiring probability

    # Attention mechanism
    attn_temperature: float = 0.5        # softmax temperature (smaller → sharper weights)
    attn_scale: float = 0.2              # weight of neighbor influence in ΔS

# ------------------------- CORE SIMULATOR ------------------------------------
class GraphTimeSimulator:
    """Simulator with complex network + neighbor attention."""

    def __init__(self, cfg: ExtConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)

        # Create graph
        self.G = self._build_graph(cfg)
        self.n = cfg.num_agents  # shorthand

        # Initial states: n × D
        self.states = np.random.rand(self.n, cfg.dims)
        if cfg.phase is None:
            cfg.phase = np.random.uniform(0, 2 * np.pi, self.n)

        self.entropy_track = []
        self.delta_track = []  # mean ‖ΔS‖ per step

    # --------------------- graph helpers ---------------------
    def _build_graph(self, cfg: ExtConfig):
        if cfg.graph_type == "watts_strogatz":
            return nx.watts_strogatz_graph(cfg.num_agents, cfg.k_neighbors, cfg.rewiring_p)
        elif cfg.graph_type == "barabasi":
            return nx.barabasi_albert_graph(cfg.num_agents, max(1, cfg.k_neighbors // 2))
        elif cfg.graph_type == "erdos_renyi":
            p = cfg.k_neighbors / (cfg.num_agents - 1)
            return nx.erdos_renyi_graph(cfg.num_agents, p)
        else:
            raise ValueError(f"Unknown graph_type: {cfg.graph_type}")

    # -------------------- simulation step --------------------
    def step(self, t: int):
        cfg = self.cfg

        # Base ΔS: deterministic oscillation + Gaussian noise
        osc = cfg.A * np.sin(cfg.omega * t + cfg.phase).reshape(-1, 1)
        noise = np.random.normal(0, cfg.sigma, size=self.states.shape)
        delta = osc + noise

        # Attention‑weighted neighbor influence
        attn_delta = self._neighbor_attention()
        delta += cfg.attn_scale * attn_delta

        # Update states
        self.states += delta

        # Logging
        self.delta_track.append(np.linalg.norm(delta, axis=1).mean())
        self.entropy_track.append(self._compute_entropy())

    # ---------------- attention -----------------
    def _neighbor_attention(self):
        cfg = self.cfg
        new_delta = np.zeros_like(self.states)

        # Pre‑compute norms for cosine similarity
        norms = np.linalg.norm(self.states, axis=1, keepdims=True) + 1e-8
        norm_states = self.states / norms

        for i in range(self.n):
            nbrs = list(self.G.neighbors(i))
            if not nbrs:
                continue
            # cosine similarity with neighbors
            sims = norm_states[nbrs] @ norm_states[i]  # (len(nbrs), )
            # softmax attention
            weights = np.exp(sims / cfg.attn_temperature)
            weights /= weights.sum()
            # aggregate neighbor state difference
            aggregated = (weights[:, None] * (self.states[nbrs] - self.states[i])).sum(axis=0)
            new_delta[i] = aggregated
        return new_delta

    # ------------- entropy (same as basic) -------------
    def _compute_entropy(self):
        cfg = self.cfg
        hist, _ = np.histogramdd(self.states, bins=cfg.bins)
        p = hist.flatten() / hist.sum()
        p = p[p > 0]
        return shannon_entropy(p, base=np.e)

    # ---------------- run full sim ----------------
    def run(self):
        for t in range(self.cfg.steps):
            self.step(t)
        return (
            np.array(self.entropy_track),
            np.array(self.delta_track),
            self.states.copy(),
        )

# ----------------------- run + archive -----------------------

def save_fig(fig_func, data_tuple, run_dir, name):
    fig_func(*data_tuple) if isinstance(data_tuple, tuple) else fig_func(data_tuple)
    plt.tight_layout()
    plt.savefig(run_dir / "figs" / f"{name}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extended time‑emergence experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--A", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=0.05)
    # graph params
    parser.add_argument("--graph_type", choices=["watts_strogatz","barabasi","erdos_renyi"], default="watts_strogatz")
    parser.add_argument("--k_neighbors", type=int, default=6)
    parser.add_argument("--rewiring_p", type=float, default=0.1)
    # attention params
    parser.add_argument("--attn_temperature", type=float, default=0.5)
    parser.add_argument("--attn_scale", type=float, default=0.2)

    args = parser.parse_args()

    # Build config dataclass
    cfg = ExtConfig(**vars(args))

    # Create run directory & save config
    run_dir = new_run_dir()
    save_config(cfg, run_dir)

    # Run simulation
    sim = GraphTimeSimulator(cfg)
    entropy, delta, _ = sim.run()

    # Correlation
    corr = pearson_corr(delta, entropy)
    save_metrics(entropy, delta, corr, run_dir)

    # Figures
    save_fig(plot_entropy, entropy, run_dir, "entropy")
    save_fig(plot_delta_vs_time, delta, run_dir, "delta")
    save_fig(scatter_entropy_vs_delta, (entropy, delta), run_dir, "corr")

    print(f"[EXT RUN SAVED] {run_dir}")
    print(f"Pearson r = {corr['pearson_r']:.3f}  (p={corr['p_value']:.4g})")


if __name__ == "__main__":
    main()
