#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({"font.size":9,"font.family":"serif","axes.linewidth":0.8,
                     "savefig.bbox":"tight","pdf.fonttype":42})
df=pd.read_csv("analysis/structure_table.csv")

# ---- Fig 1: structure -> rankability (Hedge vs rho_d, size=classes, colour=lead family) ----
catcol={"set/equivariant":"#1f77b4","sheaf":"#2ca02c","line/clique-expansion":"#ff7f0e",
        "spectral-conv":"#9467bd","two-stage":"#8c564b"}
fig,ax=plt.subplots(figsize=(6.4,4.2))
ax.axhline(1.0,ls="--",lw=0.9,color="0.5")
ax.text(0.295,1.06,r"$\rho_d=1$ (seed noise = method spread)",fontsize=7.5,color="0.4")
for _,r in df.iterrows():
    c=catcol.get(r["lead_mech"],"0.5")
    ax.scatter(r["Hedge"],r["rho_d"],s=30+6*r["classes"],color=c,edgecolor="k",lw=0.5,alpha=0.85,zorder=3)
    dy = 0.6 if r["dataset"] not in ("amazon","yelp") else (-1.1 if r["dataset"]=="amazon" else 0.6)
    ax.annotate(r["dataset"].replace("walmart-trips-100","walmart").replace("coauthor_","co_"),
                (r["Hedge"],r["rho_d"]),textcoords="offset points",xytext=(4,4),fontsize=6.8)
ax.set_yscale("log"); ax.set_xlabel(r"hyperedge homophily $\mathcal{H}_{\mathrm{edge}}$ (DHG-Bench Table A1)")
ax.set_ylabel(r"rankability $\rho_d$  (between-method / seed spread)")
ax.set_title(r"Rankability rises with homophily and class count (Spearman $r\!=\!+0.58$/$+0.57$, $n\!=\!12$)",fontsize=8.5)
leg1=[Patch(facecolor=c,edgecolor="k",label=k) for k,c in catcol.items()]
ax.legend(handles=leg1,title="leading family",fontsize=6.8,title_fontsize=7,loc="lower right",framealpha=0.9)
ax.text(0.27,0.55,"point-size $\\propto$ #classes",fontsize=6.8,color="0.4")
fig.savefig("analysis/fig_struct_rankability.pdf"); plt.close(fig)

# NOTE: the model-selection decision table is now Table 6 in the manuscript (a real LaTeX
# table, not a figure), so fig_model_selection.pdf is no longer generated here.

print("wrote fig_struct_rankability.pdf (Figure 8)")
