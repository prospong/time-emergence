# src/config.py
import numpy as np
from dataclasses import dataclass, field

@dataclass
class SimConfig:
    # 基本
    num_agents: int = 100
    dims: int = 3
    steps: int = 500
    seed: int = 42

    # 确定性部分（振荡）
    A: float = 1.0          # 振幅
    omega: float = 0.05     # 角频率
    phase: np.ndarray = field(default_factory=lambda: None)  # 每个智能体自己的初相位

    # 随机部分（高斯噪声）
    sigma: float = 0.1      # 标准差

    # 离散化熵
    bins: int = 30          # 每个维度分桶数量
