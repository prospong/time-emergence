# run_matrix.py – batch driver for six‑config × five‑seed experiments
# -----------------------------------------------------------------------------
# * 每轮调用 experiment_extend_2.py 生成 run-<stamp> 子文件夹。
# * 汇总每个配置目录 (runs_TAG) 下所有 run-*/metrics.json，
#   输出 summary_all.csv 与 aggregate_all.json。
# -----------------------------------------------------------------------------

import subprocess, sys, pathlib, json, pandas as pd

PY = sys.executable
EXT2 = "src.experiment_extend_2"

matrix = [
    # tag, evolve_interval, sigma, attn_scale
    ("A1", 0,   0.2, 0.3),
    ("A2", 0,   0.5, 0.3),
    ("B1", 100, 0.2, 0.3),
    ("B2", 100, 0.5, 0.3),
    ("B3", 100, 0.5, 0.6),
    ("C1", 400, 0.2, 0.3),
]

common_args = ["--steps", "1000", "--dims", "12"]  # 可自行调小避免 OOM

for tag, interval, sigma, scale in matrix:
    out_dir = pathlib.Path(f"runs_{tag}")
    out_dir.mkdir(exist_ok=True)

    for seed in range(5):
        cmd = [
            PY, "-m", EXT2,
            "--n_runs", "1",
            "--seed", str(seed),
            "--evolve_interval", str(interval),
            "--sigma", str(sigma),
            "--attn_scale", str(scale),
            "--out", str(out_dir),
            *common_args,
        ]
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # -------- 汇总每个 TAG 的 run 结果 --------
    metrics_files = list(out_dir.glob("run-*/metrics.json"))
    rows = []
    for mf in metrics_files:
        with open(mf) as f:
            data = json.load(f)
            rows.append(data)
    if not rows:
        print(f"[WARN] No metrics.json found for {tag}")
        continue

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_all.csv", index=False)
    agg = df.mean(numeric_only=True).to_dict()
    with open(out_dir / "aggregate_all.json", "w") as g:
        json.dump({k: float(v) for k, v in agg.items()}, g, indent=2)

    print(f"[{tag}] μ pearson_r={agg['pearson_r']:.3f}   μ spearman_r={agg['spearman_r']:.3f}")
