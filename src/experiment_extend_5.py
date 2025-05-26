# experiment_extend_5.py – universal single‑run simulator for mega‑grid
# -----------------------------------------------------------------------------
# 与 v4 差异：
#   ✓ 支持三种拓扑: watts_strogatz / barabasi / erdos_renyi (CLI --graph)
#   ✓ 参数全部 CLI 化：--num --dims --steps 等
#   ✓ 支持 --hist_stride：熵采样步长，减轻内存
#   ✓ 默认不画图，除非 --plot
#   ✓ 仅保存 metrics.json 便于大规模收集 (save_raw 可选)
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse, json, os, random, math
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np, torch, networkx as nx
from scipy.stats import entropy as shannon_entropy
from .utils import new_run_dir, save_metrics
from .metrics import pearson_corr, spearman_corr, entropy_slope

# ---------------- config ----------------
@dataclass
class Cfg:
    num:int=120; dims:int=8; steps:int=2000; seed:int=0
    A:float=0.3; omega:float=0.05; sigma:float=0.2
    attn_scale:float=0.3
    graph:str="ws"   # ws / ba / er
    k:int=8; p:float=0.05   # ws:p=rewiring, ba:k, er:p
    evolve:int=0            # 0=static
    hist_stride:int=10      # 采样间隔
    device:str="cpu"; save_raw:bool=False

# --------------- sim --------------------
class Sim:
    def __init__(self,c:Cfg):
        self.c=c; torch.manual_seed(c.seed); np.random.seed(c.seed);
        self.dev=torch.device(c.device)
        self.G=self._make_g()
        self.n=c.num; self.d=c.dims
        self.x=torch.rand((self.n,self.d),device=self.dev)
        self.phase=torch.rand(self.n,device=self.dev)*2*math.pi
        self.entropy=[]; self.delta=[]
    def _make_g(self):
        c=self.c
        if c.graph=='ws':
            return nx.watts_strogatz_graph(c.num,c.k,c.p)
        if c.graph=='ba':
            return nx.barabasi_albert_graph(c.num,max(1,c.k//2))
        return nx.erdos_renyi_graph(c.num,c.p)
    def _attn(self):
        q=self.x/ (self.x.norm(dim=1,keepdim=True)+1e-8); out=torch.zeros_like(self.x)
        for i in range(self.n):
            nbr=list(self.G.neighbors(i));
            if not nbr: continue
            w=torch.softmax((q[nbr] @ q[i])*math.sqrt(self.d),0)
            out[i]=(w[:,None]*(self.x[nbr]-self.x[i])).sum(0)
        return out*self.c.attn_scale
    def _entropy(self):
        x=self.x.detach().cpu().numpy(); bins=min(6,self.d)
        # 如果维度过高或者 n_bin^d 太大，则改为逐维熵平均
        max_cells = 2_000_000          # 约占 16 MB 内存
        n_cells   = bins ** self.d
        if n_cells > max_cells:
            e = 0.0
            for d in range(self.d):
                h,_ = np.histogram(x[:, d], bins=bins)
                p = h[h > 0] / h.sum()
                e += float(shannon_entropy(p))
            return e / self.d

        hist,_ = np.histogramdd(x, bins=bins)
        p = hist.ravel(); p = p[p > 0] / p.sum()
        return float(shannon_entropy(p))

    def run(self):
        c=self.c
        for t in range(c.steps):
            self.x+=c.A*torch.sin(c.omega*t+self.phase)[:,None]+torch.randn_like(self.x)*c.sigma+self._attn()
            if t% c.hist_stride==0:
                self.entropy.append(self._entropy())
                self.delta.append(float(self.x.norm(dim=1).mean()))
        return np.array(self.entropy),np.array(self.delta)

# ---------------- main ------------------
if __name__=='__main__':
    pa=argparse.ArgumentParser();
    pa.add_argument('--out',required=True); pa.add_argument('--seed',type=int,required=True)
    pa.add_argument('--num',type=int,default=120); pa.add_argument('--dims',type=int,default=8); pa.add_argument('--steps',type=int,default=2000)
    pa.add_argument('--sigma',type=float,default=0.2); pa.add_argument('--attn_scale',type=float,default=0.3);
    pa.add_argument('--graph',choices=['ws','ba','er'],default='ws'); pa.add_argument('--k',type=int,default=8); pa.add_argument('--p',type=float,default=0.05)
    pa.add_argument('--evolve',type=int,default=0); pa.add_argument('--hist_stride',type=int,default=10)
    pa.add_argument('--save_raw',action='store_true'); args=pa.parse_args()

    cfg=Cfg(num=args.num,dims=args.dims,steps=args.steps,seed=args.seed,sigma=args.sigma,attn_scale=args.attn_scale,
            graph=args.graph,k=args.k,p=args.p,evolve=args.evolve,hist_stride=args.hist_stride,save_raw=args.save_raw)
    run_dir=new_run_dir(Path(args.out));
    sim=Sim(cfg); ent,dn=sim.run()
    stats={**pearson_corr(dn,ent),**spearman_corr(dn,ent),
           'entropy_slope':entropy_slope(ent),'entropy_range':float(ent.max()-ent.min())}
    if cfg.save_raw:
        np.save(run_dir/'entropy.npy',ent); np.save(run_dir/'delta.npy',dn)
    save_metrics(ent,dn,stats,run_dir)
    print(json.dumps(stats))
