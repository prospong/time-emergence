# experiment_extend_6.py – same as v5 but with extra entropy metrics + MI + longer steps
# -----------------------------------------------------------------------------
from __future__ import annotations
import argparse, json, pathlib, numpy as np, torch, networkx as nx, random, math, os
from dataclasses import dataclass
from typing import List
from scipy.stats import entropy as sh_entropy
from .utils import new_run_dir, save_metrics
from .metrics import pearson_corr, spearman_corr, entropy_slope
from .metrics_extras import renyi_entropy, tsallis_entropy, mi_delta_entropy

@dataclass
class Cfg:
    num:int=120; dims:int=8; steps:int=10000; seed:int=0
    A:float=0.3; omega:float=0.05; sigma:float=0.3
    attn:float=0.3; graph:str='ws'; k:int=8; p:float=0.05
    evolve:int=0; stride:int=20; device:str='cpu'

def make_graph(c:Cfg):
    if c.graph=='ws':
        return nx.watts_strogatz_graph(c.num,c.k,c.p)
    if c.graph=='ba':
        return nx.barabasi_albert_graph(c.num,max(1,c.k//2))
    return nx.erdos_renyi_graph(c.num,c.p)

def entropy_vector(x: np.ndarray, bins: int = 6):
    d = x.shape[1]
    max_cells = 2_000_000          # ≈16 MB
    if bins ** d > max_cells:
        # 高维：用逐维熵平均
        ps = []
        for k in range(d):
            h, _ = np.histogram(x[:, k], bins=bins)
            p = h[h > 0] / h.sum()
            ps.append(p)
        # 返回拼接概率数组，仅供后续 renyi/tsallis 计算
        return np.concatenate(ps)
    hist, _ = np.histogramdd(x, bins=bins)
    p = hist.ravel(); p = p[p > 0] / p.sum()
    return p

class Sim:
    def __init__(self,c:Cfg):
        self.c=c; torch.manual_seed(c.seed); np.random.seed(c.seed); random.seed(c.seed)
        self.dev=torch.device(c.device); self.d=c.dims; self.n=c.num
        self.G=make_graph(c)
        self.x=torch.rand((self.n,self.d),device=self.dev); self.phase=torch.rand(self.n,device=self.dev)*2*math.pi
        self.ent=[]; self.ent_renyi=[]; self.ent_tsallis=[]; self.mi=[]; self.dn=[]
    def _attn(self):
        q=self.x/(self.x.norm(dim=1,keepdim=True)+1e-8); out=torch.zeros_like(self.x)
        for i in range(self.n):
            nbr=list(self.G.neighbors(i));
            if not nbr: continue
            w=torch.softmax((q[nbr]@q[i])*math.sqrt(self.d),0)
            out[i]=(w[:,None]*(self.x[nbr]-self.x[i])).sum(0)
        return out*self.c.attn
    def _entropy_metrics(self):
        p=entropy_vector(self.x.detach().cpu().numpy())
        return (float(sh_entropy(p)), renyi_entropy(p,2.0), tsallis_entropy(p,1.5))
    def run(self):
        c=self.c; delta_last=None
        for t in range(c.steps):
            delta = c.A*torch.sin(c.omega*t+self.phase)[:,None] + torch.randn_like(self.x)*c.sigma + self._attn()
            self.x += delta
            if t % c.stride == 0:
                e,r,tq = self._entropy_metrics()
                self.ent.append(e); self.ent_renyi.append(r); self.ent_tsallis.append(tq)
                d_mean=float(delta.norm(dim=1).mean())
                self.dn.append(d_mean)
                if delta_last is not None:
                    self.mi.append(mi_delta_entropy(np.array([d_mean]), np.array([e])))
                delta_last=delta
        return np.array(self.ent),np.array(self.ent_renyi),np.array(self.ent_tsallis),np.array(self.dn[1:]),np.array(self.mi)

def run_once(cfg:Cfg,out: pathlib.Path):
    rd=new_run_dir(out)
    sim=Sim(cfg)
    ent,ren,tsal,delta,mi=sim.run()
    # --- ensure two arrays have same length ---
    L = min(len(ent), len(delta))
    ent_aligned = ent[:L]
    delta_aligned = delta[:L]

    metrics = {**pearson_corr(delta_aligned, ent_aligned),
               **spearman_corr(delta_aligned, ent_aligned),
               'renyi_mean':float(ren.mean()),'tsallis_mean':float(tsal.mean()),
               'mi_mean':float(mi.mean()) if len(mi)>0 else float('nan'),
               'entropy_slope':entropy_slope(ent),'steps':cfg.steps,'dims':cfg.dims,'sigma':cfg.sigma,'graph':cfg.graph}
    save_metrics(ent,delta,metrics,rd)
    print(json.dumps(metrics))

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--out',required=True); pa.add_argument('--seed',type=int,required=True)
    pa.add_argument('--graph',choices=['ws','ba','er'],default='ws'); pa.add_argument('--sigma',type=float,default=0.3)
    pa.add_argument('--dims',type=int,default=8); pa.add_argument('--steps',type=int,default=10000)
    pa.add_argument('--stride',type=int,default=20)
    args=pa.parse_args()
    cfg=Cfg(dims=args.dims,steps=args.steps,seed=args.seed,sigma=args.sigma,graph=args.graph,stride=args.stride)
    run_once(cfg,pathlib.Path(args.out))
