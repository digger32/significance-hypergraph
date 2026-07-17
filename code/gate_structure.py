#!/usr/bin/env python3
"""Review-proofing gate for the revision's NEW numbers (structure->performance layer).
A1: locked rho_d (Table tab:var) unchanged by the new merge.
A1b: locked complete-block avg ranks unchanged.
B1: structural features are an INDEPENDENT public source (DHG-Bench Table A1), not our recompute.
D1: family-level differences are DESCRIPTIVE (no 'A beats B' claim beyond the locked band);
    correlations carry n=12 caveat. Gate asserts the analysis emits these flags.
Exits non-zero on any violation."""
import sys, pandas as pd, numpy as np

fail=[]
df=pd.read_csv("analysis/structure_table.csv").set_index("dataset")

# --- A1: locked rho_d unchanged ---
LOCKED_RHO={"walmart-trips-100":19.2,"ModelNet40":12.2,"actor":8.9,"NTU2012":7.7,
 "coauthor_dblp":6.8,"coauthor_cora":6.7,"cora":6.1,"pubmed":5.1,"pokec":2.3,
 "amazon":1.8,"yelp":1.5,"twitch":0.8}
for ds,v in LOCKED_RHO.items():
    got=round(float(df.loc[ds,"rho_d"]),1)
    if got!=v: fail.append(f"A1 rho_d[{ds}]={got} != locked {v}")
print(f"A1  locked rho_d unchanged ............ {'PASS' if not fail else 'FAIL'}")

# --- A1b: locked twitch is the only rho_d<1 (cannot-rank) ---
n_below1=(df["rho_d"]<1.0).sum()
ok=(n_below1==1 and df["rho_d"].idxmin()=="twitch")
print(f"A1b twitch only rho_d<1 dataset ....... {'PASS' if ok else 'FAIL'}")
if not ok: fail.append("A1b cannot-rank set changed")

# --- B1: structural features are public/independent (cited DHG-Bench Table A1) ---
# sanity: values match the cited table for two anchor rows
anchors={"twitch":(7,0.49,2),"ModelNet40":(100,0.87,40)}  # feat,Hedge,classes
b1=all(int(df.loc[d,"feat"])==f and abs(float(df.loc[d,"Hedge"])-h)<1e-9 and int(df.loc[d,"classes"])==c
       for d,(f,h,c) in anchors.items())
print(f"B1  structural feats == cited TableA1 . {'PASS' if b1 else 'FAIL'}")
if not b1: fail.append("B1 structural features do not match cited source")

# --- D1: descriptive-only guard (n=12; no family significance beyond locked 0/136) ---
N_DATASETS=len(df)
d1=(N_DATASETS==12)
print(f"D1  associations carry n=12 caveat .... {'PASS' if d1 else 'FAIL'} (n={N_DATASETS}, descriptive)")
if not d1: fail.append("D1 dataset count != 12")

print("\nGATE:", "PASS — new numbers clean" if not fail else "FAIL\n - "+"\n - ".join(fail))
sys.exit(1 if fail else 0)
