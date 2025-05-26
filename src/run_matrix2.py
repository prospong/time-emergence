# run_matrix2.py – batch driver for 6 configs × 5 seeds using experiment_extend_4
# ---------------------------------------------------------------------------
import subprocess, sys, pathlib, json, pandas as pd

PY       = sys.executable
EXT      = "src.experiment_extend_4"   # 模块路径
common   = [('--steps', '1200'),       # 调小步数 / 维度可防 OOM
            ('--dims',  '8')]

# tag, evolve_interval, sigma, attn_scale, rewiring_p
MATRIX = [
    ("A1", 0,   0.2, 0.3, 0.05),
    ("A2", 0,   0.5, 0.3, 0.05),
    ("B1", 100, 0.2, 0.3, 0.05),
    ("B2", 100, 0.5, 0.3, 0.05),
    ("B3", 100, 0.5, 0.6, 0.05),
    ("C1", 400, 0.2, 0.3, 0.05),
]

for tag, interval, sigma, scale, rew in MATRIX:
    out_dir = pathlib.Path(f"runs_{tag}")
    out_dir.mkdir(exist_ok=True)

    # ----- 5 seeds -----
    for seed in range(5):
        cmd = [
            PY, "-m", EXT,
            "--out", str(out_dir),
            "--evolve_interval", str(interval),
            "--sigma", str(sigma),
            "--attn_scale", str(scale),
            "--rewiring_p", str(rew),
            "--seed", str(seed),
        ]
        for k, v in common:
            cmd.extend([k, v])

        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # ----- 汇总 -----
    metrics_files = list(out_dir.glob("*/metrics.json"))  # <— 改成任意子目录
    rows = [json.load(open(f)) for f in metrics_files]
    if not rows:
        print(f"[WARN] no metrics found for {tag}")
        continue

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_all.csv", index=False)
    agg = df.mean(numeric_only=True).to_dict()
    with open(out_dir / "aggregate_all.json", "w") as g:
        json.dump({k: float(v) for k, v in agg.items()}, g, indent=2)

    pr = agg.get("pearson_r", float("nan"))
    sr = agg.get("spearman_r", float("nan"))
    def fmt(x): return "nan" if pd.isna(x) else f"{x:.3f}"
    print(f"[{tag}] μ pearson_r={fmt(pr)}   μ spearman_r={fmt(sr)}")

