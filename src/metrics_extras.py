# metrics_extras.py – additional entropy & mutual‑information helpers
# -----------------------------------------------------------------------------
# Usage:
#   from .metrics_extras import renyi_entropy, tsallis_entropy, mi_delta_entropy
# -----------------------------------------------------------------------------
import numpy as np
from scipy.stats import entropy as shannon_entropy
from sklearn.metrics import mutual_info_score

# -------- Rényi entropy (order α) --------

def renyi_entropy(p: np.ndarray, alpha: float = 2.0) -> float:
    """p must be probability vector (sum==1). α>0, α!=1"""
    p = p[p > 0]
    if alpha == 1.0:
        return float(shannon_entropy(p, base=np.e))
    return float(1.0 / (1 - alpha) * np.log(np.power(p, alpha).sum()))

# -------- Tsallis entropy (order q) --------

def tsallis_entropy(p: np.ndarray, q: float = 1.5) -> float:
    p = p[p > 0]
    if q == 1.0:
        return float(shannon_entropy(p, base=np.e))
    return float((1 - np.power(p, q).sum()) / (q - 1))

# -------- Mutual Information between |ΔS| and entropy --------

def mi_delta_entropy(delta: np.ndarray, entropy: np.ndarray, bins: int = 30) -> float:
    """Compute discrete mutual information by binning both variables"""
    # discretize
    d_bins = np.digitize(delta, np.histogram(delta, bins=bins)[1])
    e_bins = np.digitize(entropy, np.histogram(entropy, bins=bins)[1])
    return float(mutual_info_score(d_bins, e_bins))
