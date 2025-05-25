# src/utils.py
import uuid, datetime, pathlib, json, yaml, numpy as np

def new_run_dir(root="runs"):
    """返回 runs/<timestamp>-<uuid4short>/ 绝对路径"""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = str(uuid.uuid4())[:8]
    run_dir = pathlib.Path(root) / f"{ts}-{uid}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "figs").mkdir()
    return run_dir

def save_config(cfg, run_dir):
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg.__dict__, f, sort_keys=False)

def save_metrics(entropy, delta, corr, run_dir):
    # 统计信息
    metrics = {
        "entropy_mean": float(entropy.mean()),
        "entropy_std":  float(entropy.std()),
        "delta_mean":   float(delta.mean()),
        "delta_std":    float(delta.std()),
        "pearson_r":    float(corr["pearson_r"]),
        "p_value":      float(corr["p_value"]),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 原始 ndarray 也可保存
    np.save(run_dir / "raw_entropy.npy", entropy)
    np.save(run_dir / "raw_delta.npy",   delta)
