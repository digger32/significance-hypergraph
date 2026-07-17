#!/usr/bin/env python3
"""Review-proofing gate -- reviewer round (truncated-cells disclosure).

Recomputes, from the released per-seed JSON, every number that the reviewer-round
revision adds or changes, and asserts (1) the new numbers, (2) that the locked
Layer-3 numbers are untouched. Non-zero exit on any mismatch.

Run: python gate_reviewer_round.py <results_phase1/node_cls>
"""
import sys, json, glob, os
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results_phase1/node_cls"
fails = []

def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, want {want!r}")
    if not ok:
        fails.append(name)

# ---- load ----
data, nseed = {}, {}
for fp in glob.glob(os.path.join(ROOT, "*", "*.json")):
    j = json.load(open(fp)); mk = next(iter(j["metrics"]))
    v = np.array([r[-1] for r in j["metrics"][mk]], float)
    data.setdefault(j["dataset"], {})[j["method"]] = v
methods = sorted({m for d in data.values() for m in d})
hnns = [m for m in methods if m not in {"MLP", "CEGCN", "CEGAT"}]

# ---- A. truncated-cell inventory (must be exactly these 7) ----
print("## A. truncated cells")
trunc = sorted((ds, m, len(v)) for ds, d in data.items() for m, v in d.items() if len(v) != 20)
check("truncated set", trunc, sorted([
    ("ModelNet40", "TMPHN", 2), ("actor", "TMPHN", 2), ("amazon", "TMPHN", 2),
    ("pokec", "TMPHN", 2), ("pubmed", "TMPHN", 2), ("twitch", "TMPHN", 2),
    ("yelp", "HyperGCN", 2)]))
check("total released cells", sum(len(d) for d in data.values()), 225)

# ---- B. Layer 2 on complete (20-seed) cells only -> Table 3 ----
print("## B. per-dataset Wilcoxon+Holm, complete cells only (Table 3)")
WANT_T3 = {  # dataset: (methods, sig, pairs, share%)
    "walmart-trips-100": (16, 118, 120, 98), "coauthor_dblp": (17, 130, 136, 96),
    "ModelNet40": (19, 161, 171, 94), "actor": (19, 154, 171, 90),
    "pubmed": (19, 150, 171, 88), "NTU2012": (20, 165, 190, 87),
    "coauthor_cora": (20, 160, 190, 84), "cora": (20, 140, 190, 74),
    "pokec": (19, 119, 171, 70), "yelp": (9, 24, 36, 67),
    "amazon": (19, 109, 171, 64), "twitch": (19, 36, 171, 21),
}
pooled_sig = pooled_tot = 0
for ds in sorted(data):
    present = [m for m in methods if m in data[ds] and len(data[ds][m]) == 20]
    raw = []
    for i in range(len(present)):
        for k in range(i + 1, len(present)):
            a, b = data[ds][present[i]], data[ds][present[k]]
            try: _, p = stats.wilcoxon(a, b)
            except ValueError: p = 1.0
            raw.append(p)
    if not raw: continue
    pooled_tot += len(raw)
    pooled_sig += int(sum(p < 0.05 for p in raw))
    rej, _, _, _ = multipletests(raw, 0.05, "holm")
    if ds in WANT_T3:
        share = round(100 * int(rej.sum()) / len(raw))
        check(f"T3 {ds}", (len(present), int(rej.sum()), len(raw), share), WANT_T3[ds])

print("## C. pooled leaderboard count (Section 4.5)")
check("pooled", (pooled_sig, pooled_tot, round(100 * pooled_sig / pooled_tot)), (1678, 1889, 89))

# ---- D. Layer 3 locked + sensitivity ----
print("## D. across-datasets layer: locked + TMPHN sensitivity")
def layer3(hnn_list):
    mat = pd.DataFrame(index=hnn_list, columns=sorted(data), dtype=float)
    for ds in sorted(data):
        for m in hnn_list:
            if m in data[ds]: mat.loc[m, ds] = data[ds][m].mean()
    B = mat[[c for c in mat.columns if mat[c].notna().all()]]
    k, n = B.shape; M = B.values
    chi2, p = stats.friedmanchisquare(*[M[i] for i in range(k)])
    W = chi2 / (n * (k - 1))
    R = np.apply_along_axis(lambda c: stats.rankdata(-c), 0, M)
    ranks = pd.Series(R.mean(1), index=B.index).sort_values()
    raw = []
    for i in range(k):
        for kk in range(i + 1, k):
            try: _, pp = stats.wilcoxon(M[i], M[kk])
            except ValueError: pp = 1.0
            raw.append(pp)
    rej, _, _, _ = multipletests(raw, 0.05, "holm")
    return k, n, round(W, 3), int(rej.sum()), len(raw), ranks

k, n, W, sig, tot, ranks = layer3(hnns)
check("locked block", (k, n), (17, 9))
check("locked Kendall W", W, 0.447)
check("locked Holm pairs", (sig, tot), (0, 136))
check("locked top-3", [f"{m}:{r:.2f}" for m, r in ranks.head(3).items()],
      ["EDHNN:4.56", "AllSetformer:4.67", "EHNN:5.00"])
check("TMPHN cells in block with 2 seeds",
      sum(1 for ds in ["ModelNet40","NTU2012","actor","amazon","coauthor_cora",
                       "cora","pokec","pubmed","twitch"] if len(data[ds]["TMPHN"]) == 2), 6)

k2, n2, W2, sig2, tot2, ranks2 = layer3([m for m in hnns if m != "TMPHN"])
check("sensitivity block", (k2, n2), (16, 9))
check("sensitivity Kendall W", W2, 0.445)
check("sensitivity Holm pairs", (sig2, tot2), (0, 120))
check("sensitivity top band unchanged",
      list(ranks2.head(5).index),
      ["AllSetformer", "EDHNN", "EHNN", "TFHNN", "SheafHyperGNN"])

# ---- E. numbers reused in the new S5.7 paragraph ----
print("## E. variance-compression paragraph inputs")
def rho(ds):
    vals = data[ds]
    seed_sd = np.mean([v.std(ddof=1) for v in vals.values() if len(v) > 1])
    msd = np.array([v.mean() for v in vals.values()]).std(ddof=1)
    return round(msd / seed_sd, 1)
check("rho walmart/MN40/NTU", (rho("walmart-trips-100"), rho("ModelNet40"), rho("NTU2012")),
      (19.2, 12.2, 7.7))
check("rho twitch/yelp/amazon/pokec", (rho("twitch"), rho("yelp"), rho("amazon"), rho("pokec")),
      (0.8, 1.5, 1.8, 2.3))

# ---- F. trivago appendix numbers (Appendix B) ----
print("## F. trivago appendix inputs")
for m, want in (("HNHN", (10.28, 0.09, 10.29, 0.13)), ("LEGCN", (28.42, 0.68, 28.38, 0.71))):
    a = np.array([r for r in json.load(open(os.path.join(ROOT, "trivago", m + ".json")))["metrics"]["acc"]])
    got = (round(a[:,0].mean(),2), round(a[:,0].std(ddof=1),2), round(a[:,2].mean(),2), round(a[:,2].std(ddof=1),2))
    check(f"trivago {m} train/test mean+-sd", got, want)
LOGDIR = os.environ.get("TRIVAGO_LOGS", "")
if LOGDIR and os.path.isdir(LOGDIR):
    ASSERT14 = ["MLP","HGNN","HyperGCN","HCHA","HyperND","PhenomNN","SheafHyperGNN",
                "TFHNN","UniGCNII","AllSetformer","EDHNN","HyperGT","CEGCN","CEGAT"]
    def has(m, s): return s in open(os.path.join(LOGDIR, f"trivago__{m}.log"), errors="replace").read()
    check("14 label-range assertions", all(has(m, "t >= 0 && t < n_classes") for m in ASSERT14), True)
    check("EHNN host-mem 522 GiB", has("EHNN", "Unable to allocate 522. GiB"), True)
    check("HJRL/TMPHN host-mem 261 GiB",
          has("HJRL", "Unable to allocate 261. GiB") and has("TMPHN", "Unable to allocate 261. GiB"), True)
    check("DPHGNN CUDA OOM 111.16 GiB", has("DPHGNN", "Tried to allocate 111.16 GiB"), True)
else:
    print("  [SKIP] trivago log checks (set TRIVAGO_LOGS to the logs_phase1 directory)")

# ---- G. upstream-stack control (Appendix C / Table A2) ----
print("## G. upstream-stack control")
T22 = os.environ.get("TORCH22_LOGS", "")
if T22 and os.path.isdir(T22):
    import re as _re
    WANT = {("cora","HJRL"): (34.8966, 3.19), ("coauthor_cora","HJRL"): (34.4756, 4.25),
            ("NTU2012","HJRL"): (4.3141, 0.75), ("ModelNet40","HJRL"): (7.5463, 0.97),
            ("cora","HGNN"): (79.0842, 1.32), ("coauthor_cora","HGNN"): (82.3855, 1.44),
            ("NTU2012","HGNN"): (66.2127, 3.14), ("ModelNet40","HGNN"): (88.7999, 0.79)}
    PAT = r"test_acc: ([\d.]+)\+-([\d.]+)"
    SEED = r"train_acc: [\d.]+, valid_acc: [\d.]+, test_acc: ([\d.]+) \n"
    for (ds, m), (mu, sd) in sorted(WANT.items()):
        fp = os.path.join(T22, f"logs_torch22_{ds}__{m}.log")
        if not os.path.exists(fp):
            check(f"log {ds}/{m}", "MISSING", "present"); continue
        log = open(fp, errors="replace").read()
        fin = _re.findall(PAT, log)[-1]
        check(f"t2.2 {ds}/{m} mean+-sd", (round(float(fin[0]), 4), round(float(fin[1]), 2)), (mu, sd))
        # bit-identity against the released torch 2.4 per-seed accuracies
        s22 = [float(x) for x in _re.findall(SEED, log)]
        s24 = [round(r[-1], 2) for r in json.load(
            open(os.path.join(ROOT, ds, m + ".json")))["metrics"]["acc"]]
        check(f"t2.2 == t2.4 per-seed {ds}/{m}", s22 == s24, True)
    v2 = os.path.join(T22, "logs_torch22_cora__HJRL_v2.log")
    if os.path.exists(v2):
        head = open(v2, errors="replace").read()[:200]
        check("env banner in v2 log", "py 3.9.21 torch 2.2.2+cu121" in head, True)
        check("v2 rerun reproduces", "test_acc: 34.8966+-3.19" in open(v2, errors="replace").read(), True)
else:
    print("  [SKIP] upstream-stack checks (set TORCH22_LOGS to the logs_torch22 directory)")

print()
# ---- H. trivago reported range quoted in S3.2 (DHG-Bench Table 1) ----
print("## H. trivago reported range (S3.2)")
TRIVAGO_REPORTED = [36.76, 47.24, 57.67, 42.39, 52.50, 33.45, 87.19, 90.79, 53.75, 36.15, 59.92, 75.99]
check("trivago reported min/max/count",
      (min(TRIVAGO_REPORTED), max(TRIVAGO_REPORTED), len(TRIVAGO_REPORTED)),
      (33.45, 90.79, 12))

print()
if fails:
    print(f"GATE FAIL ({len(fails)}): {fails}"); sys.exit(1)
print("GATE PASS -- all reviewer-round numbers verified against released per-seed data.")
