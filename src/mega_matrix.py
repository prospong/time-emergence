# mega_matrix.py – exhaustive grid search for falsification
# ---------------------------------------------------------------------------
# 目标：覆盖 3 拓扑 × 3 σ × 2 dims × 2 steps = 36 组，每组 10 seeds。
#        共 360 runs ，并在每组计算 95% 置信区间；若 CI 跨 0 → 视为“无显著相关”。
# 实验参数：
#    graph: ws / ba / er
#    sigma: 0.1, 0.3, 0.6
#    dims : 6, 12
#    steps: 1500, 3000
# ---------------------------------------------------------------------------
import itertools, subprocess, sys, pathlib, json, numpy as np, pandas as pd, math

PY=sys.executable; EXT='src.experiment_extend_5'
SEEDS=list(range(10))  # 10 seeds / group
BASE_OUT=pathlib.Path('mega_runs');BASE_OUT.mkdir(exist_ok=True)

graphs=dict(ws={}, ba={'graph':'ba'}, er={'graph':'er'})
sigmas=[0.1,0.3,0.6]; dims=[6,12]; steps=[1500,3000]

# helper
def run_single(cfg:dict,out_dir: pathlib.Path,seed:int):
    args=[PY,'-m',EXT,'--out',str(out_dir),'--seed',str(seed)]
    for k,v in cfg.items(): args.extend([f'--{k}',str(v)])
    subprocess.run(args,check=True,stdout=subprocess.PIPE)

for g, sigma, dim, stp in itertools.product(graphs.keys(), sigmas, dims, steps):
    tag=f"{g}_s{sigma}_d{dim}_t{stp}"
    cfg=dict(num=120,dims=dim,steps=stp,sigma=sigma,**graphs[g])
    group_dir=BASE_OUT/tag; group_dir.mkdir(parents=True,exist_ok=True)
    print('RUN GROUP',tag)
    for sd in SEEDS:
        run_single(cfg,group_dir,sd)
    # collect metrics
    mets=[json.load(open(p)) for p in group_dir.glob('*/metrics.json')]
    df=pd.DataFrame(mets)
    mu=df.mean(numeric_only=True); std=df.std(numeric_only=True)
    ci95=1.96*std/ math.sqrt(len(df))
    summary={'mean':mu.to_dict(),'ci95':ci95.to_dict()}
    json.dump(summary,open(group_dir/'summary.json','w'),indent=2)
    print(f"  pearson_r={mu['pearson_r']:.3f} ±{ci95['pearson_r']:.3f}")
