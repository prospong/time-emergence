# analyze_grid6.py – load summary.json and produce tables & plots
# ---------------------------------------------------------------------------
import glob, json, pathlib, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, math

sns.set_theme()
rows=[]
for f in glob.glob("grid6_runs/*/summary.json"):
    tag = pathlib.Path(f).parent.name
    g,s,d,t = tag.split('_')[0], tag.split('_')[1], tag.split('_')[2], tag.split('_')[3]
    data=json.load(open(f))
    rows.append({"tag":tag,"graph":g,"sigma":float(s[1:]),"dims":int(d[1:]),"steps":int(t[1:]),
                **{f"mu_{k}":v for k,v in data['mean'].items()},
                **{f"ci_{k}":v for k,v in data['ci95'].items()}})

df=pd.DataFrame(rows)
# --------- 表格 ---------
print(df[["tag","mu_pearson_r","ci_pearson_r","mu_spearman_r","ci_spearman_r","mu_mi_mean","ci_mi_mean"]].to_markdown(index=False))

# --------- 梯形图：pearson with CI --------
plt.figure(figsize=(12,4))
order=df.sort_values('mu_pearson_r')['tag']
plt.errorbar(x=np.arange(len(df)), y=df.loc[order].mu_pearson_r, yerr=df.loc[order].ci_pearson_r,
             fmt='o', ecolor='gray', capsize=3)
plt.axhline(0, color='red', ls='--')
plt.xticks(np.arange(len(df)), order, rotation=90)
plt.ylabel('Pearson r')
plt.tight_layout()
plt.savefig('grid6_pearson_ci.png')
plt.close()

# --------- 热图：互信息 ---------
heat=df.pivot_table(index='sigma',columns='dims',values='mu_mi_mean',aggfunc='mean')
plt.figure(figsize=(4,4))
sns.heatmap(heat,annot=True,cmap='crest')
plt.title('Mean Mutual Information by σ & dims')
plt.tight_layout();plt.savefig('grid6_mi_heat.png');plt.close()

print("Plots saved: grid6_pearson_ci.png, grid6_mi_heat.png")
