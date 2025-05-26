# src/metrics.py
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


def pearson_corr(x: np.ndarray, y: np.ndarray):
    r, p = stats.pearsonr(x, y)
    return {"pearson_r": float(r), "p_value": float(p)}


def spearman_corr(x: np.ndarray, y: np.ndarray):
    r, p = stats.spearmanr(x, y)
    return {"spearman_r": float(r), "spearman_p": float(p)}


def entropy_slope(ent: np.ndarray, take_ratio: float = 0.7):
    """
    对前 take_ratio 区段做线性拟合，返回斜率
    """
    n = int(len(ent) * take_ratio)
    X = np.arange(n).reshape(-1, 1)
    y = ent[:n]
    model = LinearRegression().fit(X, y)
    return float(model.coef_[0])
