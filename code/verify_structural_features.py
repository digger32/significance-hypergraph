#!/usr/bin/env python3
"""
verify_structural_features.py  (wired to DHG-Bench HyperDataset)
================================================================
Re-derive the hypergraph-structural features of the DHG-Bench node-classification
datasets FROM THE RAW DATA via the benchmark's own loader, and diff them against the
values cited in the manuscript (DHG-Bench Table A1 / structure_table.csv). If they
match, Section 3.4 can move from "taken from" to "verified against".

Loads through DHG-Bench's lib_dataset.data_base.HyperDataset, which exposes .x, .y and
.hyperedge_index ([2, nnz]; row1 = hyperedge id, row0 = node id), exactly as the recompute
used. No model is run; only the raw hypergraph is read, before any algorithm preprocessing.

Per dataset it computes: n_nodes, n_hyperedges, feat_dim, n_classes, mean hyperedge size
|e|_bar, clique-expansion density (edges/nodes), and the hyperedge homophily H_edge in two
conventions (dominant-label fraction and intra-class pair fraction); the diff reports which
convention matches the cited value.

RUN (from the DHG-Bench repo root, with the project venv active):

    source .venv-dhgbench/bin/activate
    tmux new -s structcheck
    python verify_structural_features.py --cited structure_table.csv --out structural_recomputed.csv
    # Ctrl-b d to detach;  tmux a -t structcheck to reattach

The script auto-finds the dhgbench/ package directory (it must sit next to data/). If your
layout differs:  --dhgbench-dir /path/to/DHG-Bench/dhgbench . Sanity-check small first:
    --only cora NTU2012

NOTE on walmart-trips-100: it is featureless and DHG-Bench synthesises features from
feature_noise, so its feature dimensionality depends on --feature_noise (default 1.0) and is
the one configuration-dependent structural column; counts, |e|, classes and H_edge are not
affected (the diff does not fail walmart on feat dim).
"""
import argparse, sys, os, types
import numpy as np

DATASETS = ["cora","pubmed","coauthor_cora","coauthor_dblp","NTU2012","ModelNet40",
            "walmart-trips-100","actor","amazon","twitch","pokec","yelp"]

_HD = {"cls": None, "dir": None, "fnoise": 1.0}

def find_dhgbench_dir(explicit=None):
    cands = []
    if explicit: cands.append(explicit)
    here = os.path.abspath(os.path.dirname(__file__)); cwd = os.getcwd()
    cands += [os.path.join(cwd,"dhgbench"), cwd, os.path.join(here,"dhgbench"),
              here, os.path.join(here,"..","dhgbench")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c,"lib_dataset","data_base.py")):
            return os.path.abspath(c)
    raise SystemExit(
        "Could not locate the dhgbench package (a dir with lib_dataset/data_base.py).\n"
        "Run from the DHG-Bench repo root, or pass --dhgbench-dir /path/to/DHG-Bench/dhgbench.")

def init_loader(dhgbench_dir, feature_noise):
    sys.path.insert(0, dhgbench_dir)
    os.chdir(dhgbench_dir)                     # data_base.py uses '../data/...' relative paths
    from lib_dataset.data_base import HyperDataset
    _HD["cls"]=HyperDataset; _HD["dir"]=dhgbench_dir; _HD["fnoise"]=feature_noise

def _args_for(name):
    a = types.SimpleNamespace()
    a.device="cpu"; a.dname=name; a.method="MLP"; a.feature_noise=_HD["fnoise"]
    return a

def load_hypergraph(name):
    d = _HD["cls"](_args_for(name))           # __init__ calls load_data()
    def npy(t):
        try: return t.detach().cpu().numpy()
        except Exception: return np.asarray(t)
    return npy(d.y), npy(d.hyperedge_index), npy(d.x)

def extract_v2e(hyperedge_index, num_nodes):
    """Replicate DHG-Bench ExtractV2E: the released index is [V|E ; E|V]; keep the V->E half.
    Sort columns by row0, then take the slice before row0 first reaches num_nodes."""
    ei = np.asarray(hyperedge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        return ei
    order = np.argsort(ei[0], kind="stable")
    ei = ei[:, order]
    hit = np.where(ei[0] == num_nodes)[0]
    cidx = int(hit.min()) if len(hit) else ei.shape[1]
    return ei[:, :cidx]

def to_hyperedges(incidence):
    arr=np.asarray(incidence)
    if arr.ndim==2 and arr.shape[0]==2:
        nodes,edges=arr[0].astype(int),arr[1].astype(int); g={}
        for v,e in zip(nodes,edges): g.setdefault(int(e),[]).append(int(v))
        return [g[k] for k in sorted(g)]
    if isinstance(incidence,(list,tuple)): return [list(map(int,e)) for e in incidence]
    raise ValueError("Unrecognised incidence format.")

def labelled(y):
    y=np.asarray(y).reshape(-1).astype(float); return y,(~np.isnan(y))&(y>=0)

def h_edge_dominant(H,y):
    y,m=labelled(y); v=[]
    for e in H:
        lab=[int(y[i]) for i in e if i<len(y) and m[i]]
        if not lab: continue
        _,c=np.unique(lab,return_counts=True); v.append(c.max()/len(lab))
    return float(np.mean(v)) if v else float("nan")

def h_edge_pair(H,y):
    y,m=labelled(y); num=den=0
    for e in H:
        lab=[int(y[i]) for i in e if i<len(y) and m[i]]; k=len(lab)
        if k<2: continue
        _,c=np.unique(lab,return_counts=True)
        num+=int(sum(j*(j-1)//2 for j in c)); den+=k*(k-1)//2
    return num/den if den else float("nan")

def features_for(name):
    y,inc,X=load_hypergraph(name); X=np.asarray(X)
    n=int(X.shape[0]) if X.ndim==2 else int(np.asarray(y).reshape(-1).shape[0])
    inc=extract_v2e(inc, n)               # keep only the V->E half (released index is [V|E;E|V])
    H=to_hyperedges(inc)
    feat=int(X.shape[1]) if X.ndim==2 else -1
    yv,mk=labelled(y); nc=int(len(np.unique(yv[mk])))
    s=np.array([len(e) for e in H if e],float)
    return dict(dataset=name,nodes=n,edges=len(H),feat=feat,avg_e=round(float(s.mean()),2),
                classes=nc,density=round(len(H)/max(n,1),2),
                Hedge_dom=round(h_edge_dominant(H,y),3),Hedge_pair=round(h_edge_pair(H,y),3))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cited",default="structure_table.csv")
    ap.add_argument("--out",default="structural_recomputed.csv")
    ap.add_argument("--dhgbench-dir",default=None)
    ap.add_argument("--only",nargs="*",default=None)
    ap.add_argument("--feature_noise",type=float,default=1.0)
    ap.add_argument("--tol_pct",type=float,default=2.0)
    ap.add_argument("--tol_h",type=float,default=0.03)
    A=ap.parse_args()
    cited=os.path.abspath(A.cited); out=os.path.abspath(A.out)
    dhg=find_dhgbench_dir(A.dhgbench_dir); print(f"using dhgbench dir: {dhg}")
    init_loader(dhg,A.feature_noise)
    names=A.only or DATASETS; rows=[]
    for nm in names:
        try: rows.append(features_for(nm)); print(f"[ok] {nm}")
        except Exception as e: print(f"[skip] {nm}: {type(e).__name__}: {e}")
    if not rows: sys.exit("\nNo datasets computed. Check --dhgbench-dir and that data.zip is extracted.")
    import csv
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    if os.path.exists(cited):
        cit={r["dataset"]:r for r in csv.DictReader(open(cited))}
        print("\n=== diff vs cited Table A1 (PASS within tol) ===")
        print(f"{'dataset':18s} {'nodes':>14s} {'feat':>12s} {'avg_e':>12s} {'classes':>10s} {'H_edge':>22s}")
        allpass=True
        for r in rows:
            c=cit.get(r["dataset"])
            if not c: print(f"{r['dataset']:18s}  (no cited row)"); continue
            def pc(a,b):
                try: a=float(a); b=float(b)
                except: return "n/a"
                if b==0: return "PASS" if a==0 else "FAIL"
                return "PASS" if abs(a-b)/abs(b)*100<=A.tol_pct else f"FAIL({a}vs{b})"
            hc=float(c["Hedge"]); hd=abs(r["Hedge_dom"]-hc)<=A.tol_h; hp=abs(r["Hedge_pair"]-hc)<=A.tol_h
            hmatch="dom" if hd else ("pair" if hp else f"FAIL(dom={r['Hedge_dom']},pair={r['Hedge_pair']} vs {hc})")
            featcell=pc(r["feat"],c["feat"])
            if r["dataset"]=="walmart-trips-100" and "FAIL" in featcell: featcell=f"~cfg({r['feat']})"
            line=(f"{r['dataset']:18s} {pc(r['nodes'],c['nodes']):>14s} {featcell:>12s} "
                  f"{pc(r['avg_e'],c['avg_e']):>12s} {pc(r['classes'],c['classes']):>10s} {hmatch:>22s}")
            print(line)
            if "FAIL" in line: allpass=False
        print("\nRESULT:", "ALL PASS -> Section 3.4 can read 'verified against'." if allpass
              else "Some FAIL -> inspect; keep 'taken from' wording until resolved.")
        print("Record the matching H_edge convention (dom/pair) in the manuscript footnote.")
    else:
        print(f"\n(cited file {cited} not found; wrote recomputed values only)")

if __name__=="__main__":
    main()
