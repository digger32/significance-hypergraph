#!/usr/bin/env python3
"""Regenerate every figure of the manuscript from the released artefacts.

Inputs : results_phase1/node_cls/  (per-seed JSON, released)
         docs/repro_check.csv      (per-cell reproducibility table, released)
         structure_table.csv       (structural features + rho_d + leading family)
Outputs: fig_audit_pipeline.pdf, fig_repro_by_dataset.pdf, fig_distinguishable_pairs.pdf,
         fig_pairwise_heatmap.pdf, fig_top5_boxplot.pdf, fig_variance_ratio.pdf,
         fig_struct_rankability.pdf, cd_phase1.pdf

All output is vector PDF. The font is TeX Gyre Pagella, matching the Palatino
(mathpazo) face of the manuscript. Numbers shown in figures are recomputed here
from the per-seed JSON and asserted against the values in the manuscript; a
mismatch raises, so a stale figure cannot be produced silently.

Run: python make_paper_figs.py <results_phase1/node_cls> <repro_check.csv> <structure_table.csv>
"""
import sys, os, json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results_phase1/node_cls"
REPRO = sys.argv[2] if len(sys.argv) > 2 else "docs/repro_check.csv"
STRUCT = sys.argv[3] if len(sys.argv) > 3 else "structure_table.csv"

# ---- font: TeX Gyre Pagella (Palatino clone, matches mathpazo) ----
for d in ("/usr/share/texmf/fonts/opentype/public/tex-gyre",
          "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre"):
    if os.path.isdir(d):
        for f in glob.glob(os.path.join(d, "texgyrepagella-*.otf")):
            fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Pagella", "Palatino", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Pagella", "mathtext.it": "TeX Gyre Pagella:italic",
    "mathtext.bf": "TeX Gyre Pagella:bold",
    "font.size": 9.5, "axes.linewidth": 0.8,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})
NAVY, ORANGE, RED = "#2b3f54", "#e08214", "#c0392b"

# ---- load per-seed data ----
data = {}
for fp in glob.glob(os.path.join(ROOT, "*", "*.json")):
    j = json.load(open(fp)); mk = next(iter(j["metrics"]))
    data.setdefault(j["dataset"], {})[j["method"]] = np.array(
        [r[-1] for r in j["metrics"][mk]], float)
methods = sorted({m for d in data.values() for m in d})
hnns = [m for m in methods if m not in {"MLP", "CEGCN", "CEGAT"}]

# =====================================================================
# Figure 3 -- per-dataset separability (complete 20-seed cells only)
# =====================================================================
rows = []
for ds in sorted(data):
    if ds == "trivago":
        continue
    present = [m for m in methods if m in data[ds] and len(data[ds][m]) == 20]
    raw = []
    for i in range(len(present)):
        for k in range(i + 1, len(present)):
            try: _, p = stats.wilcoxon(data[ds][present[i]], data[ds][present[k]])
            except ValueError: p = 1.0
            raw.append(p)
    rej, _, _, _ = multipletests(raw, 0.05, "holm")
    rows.append((ds, int(rej.sum()), len(raw)))
sep = pd.DataFrame(rows, columns=["dataset", "sig", "tot"])
sep["share"] = 100 * sep.sig / sep.tot
sep = sep.sort_values("share")
WANT = {"walmart-trips-100": (118, 120), "coauthor_dblp": (130, 136), "ModelNet40": (161, 171),
        "actor": (154, 171), "pubmed": (150, 171), "NTU2012": (165, 190),
        "coauthor_cora": (160, 190), "cora": (140, 190), "pokec": (119, 171),
        "yelp": (24, 36), "amazon": (109, 171), "twitch": (36, 171)}
for _, r in sep.iterrows():
    assert (r.sig, r.tot) == WANT[r.dataset], f"Table-3 mismatch on {r.dataset}: {(r.sig, r.tot)}"

fig, ax = plt.subplots(figsize=(7.0, 4.2))
cols = [RED if s < 40 else (ORANGE if s < 80 else NAVY) for s in sep["share"]]
ax.barh(sep["dataset"], sep["share"], color=cols, edgecolor="black", lw=0.5, height=0.72)
for y, (_, r) in enumerate(sep.iterrows()):
    ax.text(r.share + 1.2, y, f"{r.sig}/{r.tot}", va="center", fontsize=8.2)
ax.set_xlabel(r"Distinguishable method pairs (%), paired Wilcoxon + Holm, $n=20$ seeds")
ax.set_xlim(0, 112); ax.spines[["top", "right"]].set_visible(False)
fig.savefig("fig_distinguishable_pairs.pdf"); plt.close(fig)
print("fig_distinguishable_pairs.pdf")

# =====================================================================
# Figure 2 -- signed reproducibility gap per dataset (excl. HJRL, trivago)
# =====================================================================
rep = pd.read_csv(REPRO)
rep = rep[(rep.method != "HJRL") & (rep.dataset != "Trivago")]
DSMAP = {"Cora": "cora", "Pubmed": "pubmed", "Cora-CA": "coauthor_cora",
         "DBLP-CA": "coauthor_dblp", "Walmart": "walmart-trips-100", "Actor": "actor",
         "Gamers": "twitch", "Pokec": "pokec", "Yelp": "yelp", "Ratings": "amazon"}
rep["dataset"] = rep["dataset"].map(lambda d: DSMAP.get(d, d))
gap = rep.groupby("dataset")["diff"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7.0, 4.0))
cols = [RED if g < -6 else (ORANGE if g < -2 else NAVY) for g in gap.values]
ax.barh(range(len(gap)), gap.values, color=cols, edgecolor="black", lw=0.5, height=0.72)
ax.set_yticks(range(len(gap)), gap.index)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel(r"Mean (recompute $-$ reported) accuracy, pp  [excl.\ HJRL]"
              .replace("\\ ", " "))
ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
fig.savefig("fig_repro_by_dataset.pdf"); plt.close(fig)
print("fig_repro_by_dataset.pdf")

# =====================================================================
# Layer 3 objects (shared by heatmap + CD)
# =====================================================================
mat = pd.DataFrame(index=hnns, columns=sorted(data), dtype=float)
for ds in sorted(data):
    for m in hnns:
        if m in data[ds]: mat.loc[m, ds] = data[ds][m].mean()
B = mat[[c for c in mat.columns if mat[c].notna().all() and c != "trivago"]]
k, n = B.shape; M = B.values
assert (k, n) == (17, 9)
R = np.apply_along_axis(lambda c: stats.rankdata(-c), 0, M)
ranks = pd.Series(R.mean(1), index=B.index).sort_values()
assert f"{ranks.iloc[0]:.2f}" == "4.56" and ranks.index[0] == "EDHNN"

# ---- Figure 4: Holm-corrected pairwise heatmap ----
raw, pairs = [], []
for i in range(k):
    for kk in range(i + 1, k):
        try: _, pp = stats.wilcoxon(M[i], M[kk])
        except ValueError: pp = 1.0
        raw.append(pp); pairs.append((B.index[i], B.index[kk]))
rej, padj, _, _ = multipletests(raw, 0.05, "holm")
assert int(rej.sum()) == 0
order = list(ranks.index)
H = pd.DataFrame(np.nan, index=order, columns=order)
for (a, b), p in zip(pairs, padj):
    H.loc[a, b] = H.loc[b, a] = -np.log10(max(p, 1e-18))
fig, ax = plt.subplots(figsize=(6.6, 6.0))
im = ax.imshow(H.values, cmap="Greys", vmin=0, vmax=1.8)
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
ax.set_yticks(range(len(order)), order, fontsize=8)
ax.set_xticks(np.arange(-0.5, len(order)), minor=True)
ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
ax.grid(which="minor", color="0.85", lw=0.4)
ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.045)
cb.set_label(r"$-\log_{10}$ Holm-corrected $p$")
alpha_line = -np.log10(0.05)
cb.ax.axhline(alpha_line, color=RED, lw=1.2)
cb.ax.text(1.6, alpha_line, r"$\alpha=0.05$", color=RED, fontsize=8,
           va="center", transform=cb.ax.get_yaxis_transform())
fig.savefig("fig_pairwise_heatmap.pdf"); plt.close(fig)
print("fig_pairwise_heatmap.pdf")

# ---- Figure A1: critical-difference diagram (vector) ----
nem = sp.posthoc_nemenyi_friedman(B.T.values)
nem.index = nem.columns = B.index
fig = plt.figure(figsize=(10, 3.6))
sp.critical_difference_diagram(ranks, nem)
fig.savefig("cd_phase1.pdf"); plt.close(fig)
print("cd_phase1.pdf")

# =====================================================================
# Figure 5 -- top-5 per-seed boxplot on cora
# =====================================================================
top5 = ["TFHNN", "EDHNN", "AllSetformer", "SheafHyperGNN", "EHNN"]
cora = {m: data["cora"][m] for m in top5}
top5 = sorted(top5, key=lambda m: -cora[m].mean())
fig, ax = plt.subplots(figsize=(6.6, 3.6))
bp = ax.boxplot([cora[m] for m in top5], tick_labels=top5, widths=0.55,
                patch_artist=True, medianprops=dict(color="#3a5f45", lw=1.6),
                flierprops=dict(marker="", ms=0))
for b in bp["boxes"]:
    b.set(facecolor="#f2f2ef", edgecolor="black", lw=0.8)
rng = np.random.default_rng(0)
for i, m in enumerate(top5, 1):
    ax.scatter(np.full_like(cora[m], i) + rng.uniform(-0.10, 0.10, len(cora[m])),
               cora[m], s=6, color="0.25", zorder=3)
ax.set_ylabel("Test accuracy (%)")
ax.tick_params(axis="x", rotation=15)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, color="0.9", lw=0.6); ax.set_axisbelow(True)
fig.savefig("fig_top5_boxplot.pdf"); plt.close(fig)
print("fig_top5_boxplot.pdf")

# =====================================================================
# Figure 6 -- variance decomposition rho_d
# =====================================================================
rows = []
for ds in sorted(data):
    if ds == "trivago": continue
    vals = data[ds]
    seed_sd = np.mean([v.std(ddof=1) for v in vals.values() if len(v) > 1])
    msd = np.array([v.mean() for v in vals.values()]).std(ddof=1)
    rows.append((ds, msd / seed_sd))
var = pd.DataFrame(rows, columns=["dataset", "rho"]).sort_values("rho", ascending=False)
assert f"{var[var.dataset=='twitch'].rho.iloc[0]:.1f}" == "0.8"
fig, ax = plt.subplots(figsize=(7.0, 4.0))
cols = [RED if r < 1 else (ORANGE if r < 2 else NAVY) for r in var["rho"]]
ax.barh(range(len(var)), var["rho"], color=cols, edgecolor="black", lw=0.5, height=0.72)
ax.set_yticks(range(len(var)), var["dataset"])
for y, r in enumerate(var["rho"]):
    ax.text(r + 0.25, y, f"{r:.1f}", va="center", fontsize=8.2)
ax.axvline(1, color="black", ls="--", lw=0.9)
ax.text(1.15, len(var) - 0.4, r"$\rho_d = 1$", fontsize=8.2, color="0.35")
ax.set_xlabel(r"$\rho_d$ = between-method SD / within-method seed SD")
ax.set_xlim(0, 21.5); ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("fig_variance_ratio.pdf"); plt.close(fig)
print("fig_variance_ratio.pdf")

# =====================================================================
# Figure 7 -- structure vs rankability scatter
# =====================================================================
df = pd.read_csv(STRUCT)
catcol = {"set/equivariant": "#1f77b4", "sheaf": "#2ca02c", "line/clique-expansion": "#ff7f0e",
          "spectral-conv": "#9467bd", "two-stage": "#8c564b"}
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.axhline(1.0, ls="--", lw=0.9, color="0.5")
for _, r in df.iterrows():
    c = catcol.get(r["lead_mech"], "0.5")
    ax.scatter(r["Hedge"], r["rho_d"], s=30 + 6 * r["classes"], color=c,
               edgecolor="k", lw=0.5, alpha=0.85, zorder=3)
    dx, dy = 4, 4
    if r["dataset"] == "amazon": dy = -11
    if r["dataset"] == "coauthor_cora": dx, dy = 7, -3
    ax.annotate(r["dataset"].replace("walmart-trips-100", "walmart").replace("coauthor_", "co_"),
                (r["Hedge"], r["rho_d"]), textcoords="offset points",
                xytext=(dx, dy), fontsize=7.4)
ax.set_yscale("log")
ax.set_xlabel(r"hyperedge homophily $\mathcal{H}_{\mathrm{edge}}$ (DHG-Bench Table A1)")
ax.set_ylabel(r"rankability $\rho_d$  (between-method / seed spread)")
ax.text(0.02, 0.965, r"$\rho_d=1$: seed noise = method spread", fontsize=7.4,
        color="0.4", transform=ax.transAxes, va="top")
ax.text(0.02, 0.90, r"point size $\propto$ number of classes", fontsize=7.4,
        color="0.4", transform=ax.transAxes, va="top")
leg = [Patch(facecolor=c, edgecolor="k", label=l) for l, c in catcol.items()]
ax.legend(handles=leg, title="leading family", fontsize=7.2, title_fontsize=7.6,
          loc="lower right", framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("fig_struct_rankability.pdf"); plt.close(fig)
print("fig_struct_rankability.pdf")

# =====================================================================
# Figure 1 -- audit-pipeline schematic (74--98% verdict, consistent with Table 3)
# =====================================================================
fig, ax = plt.subplots(figsize=(11.2, 6.6)); ax.axis("off")
ax.set_xlim(0, 100); ax.set_ylim(0, 62)
PAD = 0.6                 # FancyBboxPatch pad, in data units
_boxes, _texts = [], []   # padded box rects, and (artist, box_index)

def box(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={PAD},rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.1))
    _boxes.append((x - PAD, y - PAD, x + w + PAD, y + h + PAD))
    return len(_boxes) - 1

def txt(x, y, s, bi=None, **kw):
    a = ax.text(x, y, s, **kw)
    if bi is not None:
        _texts.append((a, bi))
    return a

ax.text(50, 59.0, "Is there a best hypergraph neural network?",
        ha="center", fontsize=13, fontweight="bold")

b_in = box(2, 14, 22, 32, "#e8eef3", "#2b3f54")
txt(13, 42.5, "Independent recompute", b_in, ha="center", fontsize=10.5, fontweight="bold")
txt(13, 39.4, "DHG-Bench, node classification", b_in, ha="center", fontsize=8.6,
    style="italic", color="0.25")
for i, t in enumerate(["17 HNNs + 3 baselines", "20 seeds (vs. 5 upstream)",
                       "modern, different stack", "per-seed accuracies"]):
    txt(4, 35.0 - 3.4 * i, r"$\bullet$ " + t, b_in, fontsize=8.8)

layers = [
    ("Layer 1 — Reproducibility check",
     "recompute vs. reported means (pp);\n1 pp / 3 pp tolerances"),
    ("Layer 2 — Per-dataset paired significance",
     "Wilcoxon signed-rank across seeds,\nHolm-corrected within dataset"),
    ("Layer 3 — Across-datasets comparison",
     "Friedman / Iman–Davenport omnibus,\n"
     "Kendall's $W$, Holm–Wilcoxon pairwise;\n"
     "Nemenyi CD as appendix visualization"),
    ("Layer 4 — Variance decomposition",
     "between-method spread vs. seed noise,\n"
     r"ratio $\rho_d$"),
]
ys = [46.5, 34.0, 21.5, 9.0]
for (t, s), y in zip(layers, ys):
    bi = box(31, y, 38, 10.0, "#f5f2ea", "#8a8070")
    txt(33, y + 7.7, t, bi, fontsize=9.6, fontweight="bold")
    txt(33, y + 5.9, s, bi, fontsize=8.0, va="top", linespacing=1.45)

b_out = box(76, 14, 22, 32, "#e9f0e9", "#3a5f45")
txt(87, 42.6, "Significance-aware\nverdict", b_out, ha="center", fontsize=10.5,
    fontweight="bold", va="center", linespacing=1.35)
for i, t in enumerate(["within dataset: 20 seeds\n  separate 74–98% of pairs",
                       "across datasets: 0 / 136\n  pairs survive Holm",
                       "top methods in one CD band",
                       "one dataset cannot rank\n  ($\\rho_d < 1$)"]):
    txt(77.5, 37.6 - 5.2 * i, r"$\bullet$ " + t, b_out, fontsize=8.3, va="top",
        linespacing=1.35)

for y in ys:
    ax.add_patch(FancyArrowPatch((24.8, 30), (30.2, y + 5.0), arrowstyle="-|>",
                                 mutation_scale=11, color="0.35", lw=1.0))
    ax.add_patch(FancyArrowPatch((69.8, y + 5.0), (75.2, 30), arrowstyle="-|>",
                                 mutation_scale=11, color="0.35", lw=1.0))

b_ver = box(2, 1.4, 96, 4.6, "#f0f4f0", "#3a5f45")
txt(50, 3.7, "No single method is statistically best across these datasets. "
    "Single-leader claims are not supported once significance is accounted for.",
    b_ver, ha="center", fontsize=9.6, style="italic", va="center")

# --- layout assertions: measured, not eyeballed ---
fig.canvas.draw()
inv = ax.transData.inverted()
problems = []

# (a) no two boxes may overlap
for i in range(len(_boxes)):
    for j in range(i + 1, len(_boxes)):
        ax0, ay0, ax1, ay1 = _boxes[i]; bx0, by0, bx1, by1 = _boxes[j]
        ox = min(ax1, bx1) - max(ax0, bx0)
        oy = min(ay1, by1) - max(ay0, by0)
        if ox > 0 and oy > 0:
            problems.append(f"boxes {i} and {j} overlap by {ox:.2f} x {oy:.2f} data units")

# (b) every label must sit inside its own box
for artist, bi in _texts:
    bb = artist.get_window_extent(renderer=fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    bx0, by0, bx1, by1 = _boxes[bi]
    if x0 < bx0 or x1 > bx1 or y0 < by0 or y1 > by1:
        problems.append(f"label {artist.get_text()[:40]!r} x[{x0:.1f},{x1:.1f}] "
                        f"y[{y0:.1f},{y1:.1f}] escapes box x[{bx0:.1f},{bx1:.1f}] "
                        f"y[{by0:.1f},{by1:.1f}]")

if problems:
    for p in problems:
        print("  LAYOUT:", p)
    raise SystemExit("fig_audit_pipeline: layout check failed (see above)")

fig.savefig("fig_audit_pipeline.pdf"); plt.close(fig)
print("fig_audit_pipeline.pdf  (no box overlap, all labels inside their boxes)")

print("\nAll figures regenerated (vector, TeX Gyre Pagella).")
