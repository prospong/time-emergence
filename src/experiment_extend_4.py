# experiment_extend_4.py – flexible simulator for new matrix
# -----------------------------------------------------------------------------
# 变更点 vs extend_2:
#   • rewiring_p CLI 可调 (影响初始随机性)
#   • allow --save_raw：保存 entropy.npy / delta.npy 便于后续深入画图
#   • 默认 steps 提升到 3000，但 CLI 可改
#   • 输出图新增 jointplot (seaborn) 展示 ΔS–Entropy 分布
# -----------------------------------------------------------------------------

from __future__ import annotations
import argparse, json, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import torch, networkx as nx, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import entropy as shannon_entropy
from .utils import new_run_dir, save_config, save_metrics
from .metrics import pearson_corr, spearman_corr, entropy_slope

# ---------------- dataclass -----------------------------------
@dataclass
class TorchCfg:
    num_agents:int = 120; dims:int=12; steps:int=3000; seed:int=0
    A:float=0.3; omega:float=0.05; sigma:float=0.2
    attn_scale:float=0.3; layernorm_eps:float=1e-5
    graph_type:str="watts_strogatz"; k_neighbors:int=8; rewiring_p:float=0.1
    evolve_interval:int=0; evolve_prob_add:float=0.02; evolve_prob_remove:float=0.02
    bins:int=40; device:str="cuda" if torch.cuda.is_available() else "cpu"
    save_raw:bool=False

# -------------- Simulator (同 extend_2, 略微简化) ------------
class Sim:
    def __init__(self,c:TorchCfg):
        self.c=c; torch.manual_seed(c.seed); np.random.seed(c.seed); random.seed(c.seed)
        self.d=c.dims; self.device=torch.device(c.device)
        self.G=self._g(c); self.n=c.num_agents
        self.x=torch.rand((self.n,self.d),device=self.device); self.phase=torch.rand(self.n,device=self.device)*2*np.pi
        self.ln=torch.nn.LayerNorm(self.d,eps=c.layernorm_eps).to(self.device)
        self.e,self.dn=[],[]
    def _g(self,c):
        if c.graph_type=="watts_strogatz":
            return nx.watts_strogatz_graph(c.num_agents,c.k_neighbors,c.rewiring_p)
        if c.graph_type=="barabasi":
            return nx.barabasi_albert_graph(c.num_agents,max(1,c.k_neighbors//2))
        return nx.erdos_renyi_graph(c.num_agents,c.k_neighbors/(c.num_agents-1))
    def _attn(self):
        q=self.x/ (self.x.norm(dim=1,keepdim=True)+1e-8); out=torch.zeros_like(self.x)
        for i in range(self.n):
            nbr=list(self.G.neighbors(i));
            if not nbr: continue
            k=q[nbr]; w=torch.softmax((k@q[i])*np.sqrt(self.d),0)
            out[i]=self.ln(((w[:,None]*(self.x[nbr]-self.x[i])).sum(0)))
        return out
    def _entropy(self):
        c=self.c; x = self.x.detach().cpu().numpy();
        if self.d<=6:
            hist,_=np.histogramdd(x,bins=min(c.bins,8)); p=hist.ravel();p=p[p>0]/p.sum();return float(shannon_entropy(p))
        e=0.;
        for d in range(self.d):
            h,_=np.histogram(x[:,d],bins=c.bins); p=h[h>0]/h.sum(); e+=float(shannon_entropy(p))
        return e/self.d
    def _evolve(self):
        c=self.c
        for u,v in list(self.G.edges()):
            if random.random()<c.evolve_prob_remove: self.G.remove_edge(u,v)
        while random.random()<c.evolve_prob_add:
            u,v=random.sample(range(self.n),2); self.G.add_edge(u,v)
    def run(self):
        c=self.c
        for t in range(c.steps):
            delta=c.A*torch.sin(c.omega*t+self.phase)[:,None]+torch.randn_like(self.x)*c.sigma+c.attn_scale*self._attn()
            self.x+=delta
            if t%10==0:
                self.e.append(self._entropy()); self.dn.append(delta.norm(dim=1).mean().item())
            if c.evolve_interval and t and t% c.evolve_interval==0: self._evolve()
        return np.array(self.e),np.array(self.dn)

# -------------- plot util ------------------------------------

def sns_joint(ent,delta,save_path):
    df=pd.DataFrame({'entropy':ent,'delta':delta})
    g=sns.jointplot(data=df,x='delta',y='entropy',kind='hex');g.fig.tight_layout();g.fig.savefig(save_path);plt.close()

# -------------- run_once ------------------------------------

def run_once(cfg:TorchCfg,out_root:Path):
    run_dir = new_run_dir(out_root)
    fig_dir = run_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)   # <- add this flag

    save_config(cfg,run_dir)
    sim=Sim(cfg); entropy,delta=sim.run()
    stats={**pearson_corr(delta,entropy),**spearman_corr(delta,entropy),
           'entropy_slope':entropy_slope(entropy),'entropy_range':float(entropy.max()-entropy.min()),
           'entropy_mean':float(entropy.mean()),'delta_mean':float(delta.mean())}
    save_metrics(entropy,delta,stats,run_dir)
    # figs
    sns.set_theme();
    plt.figure();sns.lineplot(x=range(len(entropy)),y=entropy);plt.title('Entropy');plt.tight_layout();plt.savefig(run_dir/'figs/entropy.png');plt.close()
    plt.figure();sns.lineplot(x=range(len(delta)),y=delta);plt.title('|ΔS|');plt.tight_layout();plt.savefig(run_dir/'figs/delta.png');plt.close()
    sns_joint(entropy,delta,run_dir/'figs/joint.png')
    if cfg.save_raw:
        np.save(run_dir/'entropy.npy',entropy);np.save(run_dir/'delta.npy',delta)
    return stats

# -------------- CLI-------------------------------------------
if __name__=='__main__':
    pa=argparse.ArgumentParser('Flexible time‑emergence experiment')
    pa.add_argument('--out',default='runs')
    pa.add_argument('--steps',type=int,default=3000)
    pa.add_argument('--dims',type=int,default=12)
    pa.add_argument('--sigma',type=float,default=0.2)
    pa.add_argument('--attn_scale',type=float,default=0.3)
    pa.add_argument('--rewiring_p',type=float,default=0.1)
    pa.add_argument('--evolve_interval',type=int,default=0)
    pa.add_argument('--seed',type=int,default=0)
    pa.add_argument('--save_raw',action='store_true')
    args=pa.parse_args()
    cfg=TorchCfg(steps=args.steps,dims=args.dims,sigma=args.sigma,attn_scale=args.attn_scale,rewiring_p=args.rewiring_p,evolve_interval=args.evolve_interval,seed=args.seed,save_raw=args.save_raw)
    res=run_once(cfg,Path(args.out))
    print(json.dumps(res,indent=2))
