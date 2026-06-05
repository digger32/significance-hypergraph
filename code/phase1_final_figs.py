"""Generate CD diagram + repro-check table from per-seed JSON."""
import os, json, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
from node_acc import T1, A5, DATASETS as PAPER_DS

# load per-seed test acc
data={}
for fp in glob.glob("results_phase1/node_cls/*/*.json"):
    j=json.load(open(fp)); ds=j["dataset"]; m=j["method"]
    mk=next(iter(j["metrics"]))
    seeds=np.array([row[-1] for row in j["metrics"][mk]],float)
    data.setdefault(ds,{})[m]=seeds

# build mean matrix (methods x datasets)
all_methods=sorted({m for d in data.values() for m in d})
all_ds=sorted(data)
M=pd.DataFrame(index=all_methods, columns=all_ds, dtype=float)
for ds in all_ds:
    for m in data[ds]: M.loc[m,ds]=data[ds][m].mean()

# CD diagram across 17 HNNs over the 9 complete datasets
hnns=[m for m in all_methods if m not in ("MLP","CEGCN","CEGAT")]
complete=[c for c in all_ds if M.loc[hnns,c].notna().all() and c!="trivago"]
B=M.loc[hnns,complete]
k,n=B.shape; mat=B.values
R=np.apply_along_axis(lambda c: stats.rankdata(-c),0,mat)
ranks=pd.Series(R.mean(1),index=B.index).sort_values()
nem=sp.posthoc_nemenyi_friedman(B.T.values); nem.index=nem.columns=B.index
plt.figure(figsize=(10,3.6))
sp.critical_difference_diagram(ranks,nem)
plt.title(f"Phase-1 recompute (20 seeds): {k} HNNs ranked over {n} complete datasets",fontsize=10)
plt.tight_layout(); plt.savefig("cd_phase1.png",dpi=140,bbox_inches="tight"); plt.close()
print(f"CD: {k} HNNs x {n} datasets -> cd_phase1.png")

# reproducibility check: our mean vs paper Table1+A5
paper={**{m:{d:v for d,v in zip(PAPER_DS, T1[m]+A5[m]) if v is not None} for m in T1}}
# map paper name -> harness name
NAMEMAP={"AllSetTransformer":"AllSetformer","ED-HNN":"EDHNN","TF-HNN":"TFHNN",
         "T-HyperGNN":"TMPHN","UniGNN":"UniGCNII"}
DSMAP={"Cora":"cora","Pubmed":"pubmed","Cora-CA":"coauthor_cora","DBLP-CA":"coauthor_dblp",
       "Walmart":"walmart-trips-100","Trivago":"trivago","Actor":"actor","Gamers":"twitch",
       "Pokec":"pokec","Yelp":"yelp","NTU2012":"NTU2012","ModelNet40":"ModelNet40","Ratings":"amazon"}
rows=[]
for pm,pdct in paper.items():
    om=NAMEMAP.get(pm,pm)
    if om not in data and om not in M.index: continue
    for pd_ds,pv in pdct.items():
        ods=DSMAP.get(pd_ds); 
        if ods not in M.columns or pd.isna(M.loc[om,ods]): continue
        rows.append({"method":pm,"dataset":pd_ds,"paper":pv,"ours":M.loc[om,ods],"diff":M.loc[om,ods]-pv})
rep=pd.DataFrame(rows)
rep["abs_diff"]=rep["diff"].abs()
print(f"\nReproducibility: {len(rep)} cells compared")
print(f"  mean |Δ|={rep.abs_diff.mean():.2f}pp, median |Δ|={rep.abs_diff.median():.2f}pp, max |Δ|={rep.abs_diff.max():.2f}pp")
print(f"  within 1pp: {(rep.abs_diff<1).sum()}/{len(rep)} ({100*(rep.abs_diff<1).mean():.0f}%)")
print(f"  within 3pp: {(rep.abs_diff<3).sum()}/{len(rep)} ({100*(rep.abs_diff<3).mean():.0f}%)")
print(f"  worst 5:")
print(rep.nlargest(5,"abs_diff")[["method","dataset","paper","ours","diff"]].to_string(index=False))
rep.to_csv("repro_check.csv",index=False); print("-> repro_check.csv")
