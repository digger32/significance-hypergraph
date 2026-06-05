#!/usr/bin/env python3
"""Phase 1 analyzer — consumes per-seed JSON from results_phase1/node_cls/<ds>/<method>.json.
Delivers the things the reported-means analysis (Phase 0) could NOT:
  (A) per-dataset PAIRED Wilcoxon across seeds (valid: seed s -> same split for all methods);
  (B) variance decomposition (Bouthillier 2021): seed/init noise vs between-method gap;
  (C) significance-aware leaderboard: how many head-to-head 'wins' survive seed noise.
Also reproduces the across-datasets layer (Friedman/Iman-Davenport/Nemenyi/CD, Wilcoxon-Holm)
now with proper per-seed power, and a reproducibility check vs the paper's reported means.
Run: python phase1_analyze_perseed.py results_phase1/node_cls [--metric test_acc]
"""
import sys, os, json, glob, argparse
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ap = argparse.ArgumentParser()
ap.add_argument("root", help="dir with <dataset>/<method>.json")
ap.add_argument("--metric", default=None, help="metric key (default: first); test col = last index")
ap.add_argument("--paper_csv", default=None, help="optional CSV of reported means for repro check")
A = ap.parse_args()

# ---- load per-seed test values: dict[dataset][method] = np.array(seeds) ----
data = {}
for fp in glob.glob(os.path.join(A.root, "*", "*.json")):
    j = json.load(open(fp))
    ds, m = j["dataset"], j["method"]
    mk = A.metric or next(iter(j["metrics"]))
    seeds = np.array([row[-1] for row in j["metrics"][mk]], float)  # test column
    data.setdefault(ds, {})[m] = seeds
if not data:
    sys.exit(f"No JSON found under {A.root}. Run the recompute first.")

datasets = sorted(data)
methods = sorted({m for ds in data.values() for m in ds})
nseeds = {ds: max(len(v) for v in data[ds].values()) for ds in datasets}
print(f"Loaded {len(methods)} methods x {len(datasets)} datasets; "
      f"seeds/dataset: {sorted(set(nseeds.values()))}")

# ===== (A) per-dataset PAIRED Wilcoxon across seeds, Holm within each dataset =====
print("\n### (A) Per-dataset paired Wilcoxon across seeds (Holm within dataset)")
perds_sig = {}
for ds in datasets:
    present = [m for m in methods if m in data[ds] and len(data[ds][m]) == nseeds[ds]]
    pairs, raw = [], []
    for i in range(len(present)):
        for k in range(i+1, len(present)):
            a, b = data[ds][present[i]], data[ds][present[k]]
            try: _, p = stats.wilcoxon(a, b)
            except ValueError: p = 1.0
            pairs.append((present[i], present[k])); raw.append(p)
    if raw:
        rej, _, _, _ = multipletests(raw, 0.05, "holm")
        perds_sig[ds] = int(rej.sum())
        print(f"  {ds:16s}: {int(rej.sum()):3d}/{len(raw)} method pairs distinguishable "
              f"(n={nseeds[ds]} seeds, {len(present)} methods)")

# ===== (B) variance decomposition (Bouthillier-style) =====
print("\n### (B) Variance: seed noise vs between-method spread (per dataset)")
print("  gap/noise ratio >> 1 means method differences exceed seed jitter")
for ds in datasets:
    vals = {m: data[ds][m] for m in data[ds]}
    if len(vals) < 2:
        print(f"  {ds:16s}: only {len(vals)} method — variance spread n/a"); continue
    seed_sd = np.mean([v.std(ddof=1) for v in vals.values() if len(v) > 1])  # avg within-method (seed) SD
    method_means = np.array([v.mean() for v in vals.values()])
    method_sd = method_means.std(ddof=1)                                # spread of method means
    print(f"  {ds:16s}: seed_SD={seed_sd:5.2f}  method_spread_SD={method_sd:5.2f}  "
          f"ratio={method_sd/seed_sd if seed_sd>0 else float('inf'):4.1f}")

# ===== across-datasets layer: per-dataset mean matrix -> Friedman/CD/Wilcoxon-Holm =====
mat = pd.DataFrame(index=methods, columns=datasets, dtype=float)
for ds in datasets:
    for m in methods:
        if m in data[ds]: mat.loc[m, ds] = data[ds][m].mean()
hnns = [m for m in methods if m not in {"MLP","CEGCN","CEGAT"}]
B = mat.loc[hnns, [c for c in datasets if mat.loc[hnns, c].notna().all()]]
k, n = B.shape
if n >= 3 and k >= 3:
    M = B.values
    chi2, p = stats.friedmanchisquare(*[M[i] for i in range(k)])
    F = (n-1)*chi2/(n*(k-1)-chi2); pF = stats.f.sf(F, k-1, (k-1)*(n-1)); W = chi2/(n*(k-1))
    R = np.apply_along_axis(lambda c: stats.rankdata(-c), 0, M)
    ranks = pd.Series(R.mean(1), index=B.index).sort_values()
    raw, pr = [], []
    for i in range(k):
        for kk in range(i+1, k):
            try: _, pp = stats.wilcoxon(M[i], M[kk])
            except ValueError: pp = 1.0
            raw.append(pp)
    rej, _, _, _ = multipletests(raw, 0.05, "holm")
    print(f"\n### (across datasets, {k} HNNs x {n}) Friedman p={p:.1e} | "
          f"Iman-Davenport p={pF:.1e} | Kendall W={W:.3f} | Wilcoxon-Holm sig pairs={int(rej.sum())}/{len(raw)}")
    print("  top-3 avg rank:", ", ".join(f"{m}({r:.2f})" for m,r in ranks.head(3).items()))

# ===== (C) significance-aware leaderboard: fraction of pairwise wins that are 'real' =====
print("\n### (C) Significance-aware leaderboard (pooled across datasets, paired by seed)")
real = total = 0
for ds in datasets:
    present = [m for m in methods if m in data[ds] and len(data[ds][m]) == nseeds[ds]]
    for i in range(len(present)):
        for k2 in range(i+1, len(present)):
            a, b = data[ds][present[i]], data[ds][present[k2]]
            total += 1
            if a.mean() != b.mean():
                try: _, p = stats.wilcoxon(a, b)
                except ValueError: p = 1.0
                real += (p < 0.05)
print(f"  head-to-head comparisons: {real}/{total} significant at p<0.05 "
      f"({100*real/total:.0f}%) before multiplicity correction" if total else
      "  not enough methods/datasets yet for head-to-head leaderboard")

# ===== reproducibility check vs paper means (optional) =====
if A.paper_csv and os.path.exists(A.paper_csv):
    rep = pd.read_csv(A.paper_csv, index_col=0)
    diffs = (mat - rep).abs()
    print(f"\n### Reproducibility vs paper Table 1: mean |Δ|={np.nanmean(diffs.values):.2f} pp, "
          f"max |Δ|={np.nanmax(diffs.values):.2f} pp")
print("\nDone.")
