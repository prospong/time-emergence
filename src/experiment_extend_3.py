# experiment_extend_3.py – "streaming" batch runner
# -----------------------------------------------------------------------------
# 场景：Codespaces / 小内存环境经常因一次跑多轮而被 OOM 杀掉。
# 解决：每轮实验启动 *独立子进程*，跑完即退出 → OS 释放全部内存。
# -----------------------------------------------------------------------------
# 依赖：python>=3.8。同一仓库已存在 experiment_extend_2.py。
# 逻辑：
#   1. 逐个 seed 循环；
#   2. 用 subprocess 调用 `python -m src.experiment_extend_2 --n_runs 1 ...`；
#   3. 将所有单轮 run_dir 保存在同一个 --out 目录；
#   4. 主进程汇总 summary.csv / aggregate.json。
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

SCRIPT = "src.experiment_extend_2"


def run_subprocess(seed: int, args):
    """调用 experiment_extend_2，每次 n_runs=1，指定 seed。"""
    cmd = [
        sys.executable,
        "-m", SCRIPT,
        "--n_runs", "1",
        "--out", args.out,
        "--steps", str(args.steps),
        "--dims", str(args.dims),
        "--sigma", str(args.sigma),
        "--attn_scale", str(args.attn_scale),
    ]
    # experiment_extend_2 默认把 seed=i 写入 TorchConfig(seed=i)
    # 我们通过环境变量传递 SEED
    env = dict(**os.environ, EXT2_SEED=str(seed))
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser("Streaming batch runner (1 subprocess per run)")
    parser.add_argument("--total_runs", type=int, default=10)
    parser.add_argument("--out", type=str, default="torch_runs_stream")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dims", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--attn_scale", type=float, default=0.2)
    args = parser.parse_args()

    # 确保输出目录存在
    Path(args.out).mkdir(exist_ok=True)

    for seed in range(args.total_runs):
        run_subprocess(seed, args)
        # 子进程结束后，系统已回收全部显存/内存

    # -------- 汇总已有 summary.csv 文件 --------
    # experiment_extend_2 会在 args.out 目录追加 /summary.csv；
    # 我们需要合并所有 CSV。
    csv_files = list(Path(args.out).glob("run-*/summary.csv"))
    if not csv_files:  # fallback：如果 extend_2 直接写到 out/summary.csv
        csv_files = [Path(args.out) / "summary.csv"]

    dfs = [pd.read_csv(f) for f in csv_files if f.exists()]
    if dfs:
        concat = pd.concat(dfs, ignore_index=True)
        concat.to_csv(Path(args.out) / "summary_all.csv", index=False)
        agg = concat.mean().to_dict()
        with open(Path(args.out) / "aggregate_all.json", "w") as f:
            json.dump({k: float(v) for k, v in agg.items()}, f, indent=2)
        print("\n=== Aggregate over", len(dfs), "runs ===")
        print(json.dumps(agg, indent=2))
    else:
        print("No summary.csv files found – check sub‑runs.")
