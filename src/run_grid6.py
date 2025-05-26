# run_grid6.py – batch grid for experiment_extend_6 (Rényi/Tsallis/MI)
# ----------------------------------------------------------------------------
# 参数网格：
#   graph ∈ {ws, ba, er}
#   sigma ∈ {0.1, 0.3, 0.6}
#   dims  ∈ {6, 12}
#   steps ∈ {5000, 10000}
# 每组合跑 seeds=0..9 (10 次)，保存 metrics.json
# 汇总 mean & 95% CI 并写 summary.json
# ----------------------------------------------------------------------------
import itertools, subprocess, sys, pathlib, json, math, pandas as pd, numpy as np

PY = sys.executable
MOD = "src.experiment_extend_6"
BASE = pathlib.Path("grid6_runs"); BASE.mkdir(exist_ok=True)
SEEDS = list(range(10))

GRAPHS = ["ws", "ba", "er"]
SIGMAS = [0.1, 0.3, 0.6]
DIMS   = [6, 12]
STEPS  = [5000, 10000]

# ---------------------------------------------------------------
for graph, sigma, dims, steps in itertools.product(GRAPHS, SIGMAS, DIMS, STEPS):
    tag = f"{graph}_s{sigma}_d{dims}_t{steps}"
    out_dir = BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== RUN", tag, "===")

    for sd in SEEDS:
        cmd = [
            PY, "-m", MOD,
            "--out", str(out_dir),
            "--seed", str(sd),
            "--graph", graph,
            "--sigma", str(sigma),
            "--dims", str(dims),
            "--steps", str(steps),
            "--stride", "50"   # 采样步长，减内存
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE)

    # -------- 汇总 --------
    metas = [json.load(open(p)) for p in out_dir.glob("*/metrics.json")]
    df = pd.DataFrame(metas)
    mu = df.mean(numeric_only=True)
    std = df.std(numeric_only=True)
    ci95 = 1.96 * std / math.sqrt(len(df))

    summary = {"mean": mu.to_dict(), "ci95": ci95.to_dict()}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=2)
    print(f"   pearson μ={mu['pearson_r']:.3f}  ±{ci95['pearson_r']:.3f}")
