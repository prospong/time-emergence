# src/metrics.py
import scipy.stats as stats
import numpy as np

def pearson_corr(x: np.ndarray, y: np.ndarray):
    r, p = stats.pearsonr(x, y)
    return {"pearson_r": r, "p_value": p}
