# src/simulator.py
import numpy as np
from scipy.stats import entropy as shannon_entropy
from .config import SimConfig

class TimeSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)

        # 初始状态：随机 0~1
        self.states = np.random.rand(cfg.num_agents, cfg.dims)

        # 给每个智能体分配一个随机初相位
        if cfg.phase is None:
            cfg.phase = np.random.uniform(0, 2*np.pi, cfg.num_agents)

        # 记录
        self.entropy_track = []
        self.delta_norm_track = []

    # ===== 单步更新 ===== #
    def step(self, t: int):
        cfg = self.cfg
        # 确定性振荡 ΔS_det
        osc = cfg.A * np.sin(cfg.omega * t + cfg.phase).reshape(-1,1)

        # 随机高斯噪声 ΔS_rand
        noise = np.random.normal(0, cfg.sigma, size=self.states.shape)

        delta = osc + noise     # 总 ΔS
        self.states += delta

        # ------------- 记录指标 ------------- #
        self.delta_norm_track.append(np.linalg.norm(delta, axis=1).mean())
        self.entropy_track.append(self._compute_entropy())

    # ===== 计算熵：把连续状态离散化再求 Shannon 熵 ===== #
    def _compute_entropy(self):
        cfg = self.cfg
        # 对每个维度分箱
        hist, _ = np.histogramdd(self.states, bins=cfg.bins)
        probs = hist.flatten() / hist.sum()
        # 去掉 0 概率
        probs = probs[probs > 0]
        return shannon_entropy(probs, base=np.e)

    # ===== 运行完整仿真 ===== #
    def run(self):
        for t in range(self.cfg.steps):
            self.step(t)
        return (
            np.array(self.entropy_track),
            np.array(self.delta_norm_track),
            self.states.copy()
        )
