#!/usr/bin/env python3
"""Merge DHG-Bench Table A1 structural features (SOURCED, cited) with recompute-derived
rho_d and per-dataset leading family. Compute descriptive Spearman associations (n=12).
Locked numbers (rho_d) are taken from the per-seed JSONs and must match the manuscript."""
import os, json, glob
import numpy as np, pandas as pd
from scipy import stats

ROOT = "repo/significance-hypergraph/results_phase1/node_cls"

# DHG-Bench Table A1 (arXiv 2508.12244 v2, Appendix A). repo_name -> features.
# (nodes, edges, feat, avg_e, Hedge, classes)
STRUCT = {
 "cora":            (2708,   1579,  1433, 3.03, 0.75, 7),
 "pubmed":          (19717,  7963,  500,  4.35, 0.78, 3),
 "coauthor_cora":   (2708,   1072,  1433, 4.28, 0.78, 7),   # Cora-CA
 "coauthor_dblp":   (41302,  22363, 1425, 4.45, 0.87, 6),   # DBLP-CA
 "NTU2012":         (2012,   2012,  100,  5.00, 0.79, 67),
 "ModelNet40":      (12311,  12311, 100,  5.00, 0.87, 40),
 "walmart-trips-100":(88860, 69906, 100,  6.59, 0.60, 11),  # Walmart
 "actor":           (16255,  10164, 50,   5.25, 0.46, 3),
 "amazon":          (22299,  2090,  111,  3.10, 0.37, 5),   # Amazon-ratings / Ratings
 "twitch":          (16812,  2627,  7,    6.23, 0.49, 2),   # Twitch-gamers / Gamers
 "pokec":           (14998,  2406,  65,   2.29, 0.45, 2),
 "yelp":            (50758,  679302,1862, 6.66, 0.29, 9),
 # trivago excluded from analysis
}

# DHG-Bench official taxonomy (paper sec 3.2 / A.2). repo names.
CAT = {  # spectral / spatial / tensor / baseline-expansion
 "HGNN":"spectral","HyperGCN":"spectral","HCHA":"spectral","LEGCN":"spectral",
 "HyperND":"spectral","PhenomNN":"spectral","SheafHyperGNN":"spectral","HJRL":"spectral",
 "DPHGNN":"spectral","TFHNN":"spectral",
 "HNHN":"spatial","UniGCNII":"spatial","AllSetformer":"spatial","EDHNN":"spatial","HyperGT":"spatial",
 "EHNN":"tensor","TMPHN":"tensor",
 "MLP":"baseline","CEGCN":"baseline","CEGAT":"baseline",
}
# finer mechanism reading (within DHG-Bench categories)
MECH = {
 "AllSetformer":"set/equivariant","EDHNN":"set/equivariant","EHNN":"set/equivariant",
 "HGNN":"spectral-conv","HCHA":"spectral-conv","HyperGCN":"spectral-conv",
 "TFHNN":"decoupled/training-free","SheafHyperGNN":"sheaf",
 "HNHN":"two-stage","UniGCNII":"two-stage",
 "LEGCN":"line/clique-expansion","CEGCN":"line/clique-expansion","CEGAT":"line/clique-expansion",
 "HyperND":"nonlinear-diffusion","DPHGNN":"nonlinear-diffusion","PhenomNN":"energy/diffusion",
 "HyperGT":"transformer","TMPHN":"tensor","HJRL":"cross-expansion(non-repro)","MLP":"feature-only",
}
HNNS = [m for m in CAT if CAT[m]!="baseline"]

# load per-seed test acc
data={}
for fp in glob.glob(os.path.join(ROOT,"*","*.json")):
    j=json.load(open(fp)); ds,m=j["dataset"],j["method"]; mk=next(iter(j["metrics"]))
    data.setdefault(ds,{})[m]=np.array([r[-1] for r in j["metrics"][mk]],float)

rows=[]
for ds,(nodes,edges,feat,avge,H,C) in STRUCT.items():
    vals=data[ds]
    seed_sd=np.mean([v.std(ddof=1) for v in vals.values() if len(v)>1])
    means={m:v.mean() for m,v in vals.items()}
    rho=np.std(list(means.values()),ddof=1)/seed_sd
    hnn_means={m:means[m] for m in means if m in HNNS}
    lead=max(hnn_means,key=hnn_means.get)
    # range of HNN means (accuracy spread) & overall ceiling
    rng=max(hnn_means.values())-min(hnn_means.values())
    rows.append(dict(dataset=ds,nodes=nodes,edges=edges,feat=feat,avg_e=avge,Hedge=H,classes=C,
                     density=round(edges/nodes,2), rho_d=round(rho,1),
                     lead=lead, lead_cat=CAT[lead], lead_mech=MECH[lead],
                     lead_acc=round(means[lead],1), hnn_spread=round(rng,1)))
df=pd.DataFrame(rows).sort_values("rho_d",ascending=False)
pd.set_option("display.width",200)
print("=== structural features (DHG-Bench Table A1) + recompute rho_d + leading HNN ===")
print(df[["dataset","nodes","feat","avg_e","Hedge","classes","density","rho_d","lead","lead_cat","lead_mech","lead_acc"]].to_string(index=False))

# Spearman associations (descriptive, n=12)
print("\n=== Spearman assoc of structural features with rho_d (n=12, DESCRIPTIVE) ===")
for col in ["Hedge","classes","feat","avg_e","density","nodes"]:
    x=df[col].values; r,p=stats.spearmanr(x,df["rho_d"].values)
    print(f"  rho_d vs {col:8s}: Spearman r={r:+.2f}  (p={p:.2f}, n=12)")
# Also: does Hedge predict whether HNNs spread (hnn_spread)?
r,p=stats.spearmanr(df["Hedge"],df["hnn_spread"]); print(f"  hnn_spread vs Hedge: r={r:+.2f} (p={p:.2f})")

# family mean rank per dataset under DHG-Bench categories + finer mechanism
def fam_table(famkey):
    acc={}
    for ds in STRUCT:
        means={m:data[ds][m].mean() for m in data[ds] if m in HNNS}
        s=pd.Series(means); ranks=s.rank(ascending=False)
        fr={}
        for m,rk in ranks.items(): fr.setdefault(famkey[m],[]).append(rk)
        for f,rs in fr.items(): acc.setdefault(f,[]).append(np.mean(rs))
    return {f:(round(np.mean(rs),2),len(rs)) for f,rs in acc.items()}
print("\n=== avg per-dataset mean-rank by DHG-Bench category (lower=better) ===")
for f,(mr,n) in sorted(fam_table(CAT).items(),key=lambda kv:kv[1][0]):
    print(f"  {f:9s}: {mr:5.2f}  (n={n} datasets)")
print("\n=== avg per-dataset mean-rank by finer mechanism (lower=better) ===")
for f,(mr,n) in sorted(fam_table(MECH).items(),key=lambda kv:kv[1][0]):
    print(f"  {f:26s}: {mr:5.2f}  (n={n})")

df.to_csv("analysis/structure_table.csv",index=False)
print("\nsaved analysis/structure_table.csv")
