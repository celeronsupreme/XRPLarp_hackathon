import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def cluster_stats(df, lab):
    df = df.reset_index(drop=True)
    ok = lab >= 0
    g = pd.DataFrame({"c": lab[ok], "t": df.ts.values[ok], "s": df.sender.values[ok], "r": df.receiver.values[ok], "v": df.d_usd.values[ok]})
    size = g.groupby("c").size()
    ev = pd.concat([g[["c","t","r","v"]].rename(columns={"r":"a"}).assign(sv=lambda x: x.v),
                    g[["c","t","s","v"]].rename(columns={"s":"a"}).assign(sv=lambda x: -x.v)])
    ev = ev.sort_values(["c","a","t","sv"], ascending=[True,True,True,False])
    ev["bal"] = ev.groupby(["c","a"]).sv.cumsum()
    ev["inflow"] = ev.sv.clip(lower=0).groupby([ev.c, ev.a]).cumsum()
    acc = ev.groupby(["c","a"]).agg(has_in=("sv", lambda x: (x>0).any()), has_out=("sv", lambda x: (x<0).any()),
                                    minbal=("bal","min"), tin=("inflow","max"), tout=("sv", lambda x: -x[x<0].sum()))
    mid = acc[acc.has_in & acc.has_out].copy()
    mid["funded"] = mid.minbal >= -0.02*mid.tin.clip(lower=1)
    mid["cons"] = np.minimum(mid.tin, mid.tout)/np.maximum(mid.tin, mid.tout).clip(lower=1e-9)
    cs = mid.groupby("c").agg(n_mid=("funded","size"), funded_all=("funded","all"), funded_frac=("funded","mean"), cons=("cons","mean"))
    cs["size"] = size
    cs["span_h"] = g.groupby("c").t.agg(lambda x: (x.max()-x.min())/3600)
    cs["n_acct"] = pd.concat([g[["c","s"]].rename(columns={"s":"a"}), g[["c","r"]].rename(columns={"r":"a"})]).groupby("c").a.nunique()
    return cs


def group_features(df, pairs_p, thresholds=(0.7, 0.8, 0.9)):
    """pairs_p: DataFrame with i, j, p (link probability). Returns per-payment features of its rebuilt group."""
    df = df.reset_index(drop=True); n = len(df)
    f = pd.DataFrame(index=df.index)
    for th in thresholds:
        sel = pairs_p[pairs_p.p > th]
        G = sp.coo_matrix((np.ones(len(sel)), (sel.i.values, sel.j.values)), shape=(n, n))
        _, lab = connected_components(G, directed=False)
        sizes = np.bincount(lab); lab = np.where(sizes[lab] > 1, lab, -1)
        cs = cluster_stats(df, lab)
        tag = f"g{int(th*100)}"
        f[f"{tag}_size"] = np.where(lab >= 0, sizes[np.maximum(lab, 0)], 1)
        for col in ["n_mid", "funded_all", "funded_frac", "cons", "span_h", "n_acct"]:
            v = pd.Series(lab).map(cs[col]).values.astype(float)
            f[f"{tag}_{col}"] = np.where(lab >= 0, v, np.nan)
        f[f"{tag}_link_deg"] = np.bincount(np.concatenate([sel.i.values, sel.j.values]), minlength=n)
        ok = lab >= 0
        pair = pd.Series(df.sender.values + ">" + df.receiver.values); pn = pair.map(pair.value_counts()).values
        g = pd.DataFrame({"c": lab[ok], "t": df.ts.values[ok], "v": df.d_usd.values[ok], "first": (pn[ok] == 1).astype(float),
                          "cross": df.cross.values[ok], "da": df.d_asset.values[ok], "pair": pair.values[ok]})
        gs = g.sort_values("t")
        agg = g.groupby("c").agg(usd_med=("v", "median"), usd_cv=("v", lambda x: x.std() / x.mean() if len(x) > 1 else 0),
                                 first_share=("first", "mean"), cross_share=("cross", "mean"), n_assets=("da", "nunique"),
                                 n_pairs=("pair", "nunique"))
        agg["gap_min"] = gs.groupby("c").t.agg(lambda x: np.median(np.diff(x)) / 60 if len(x) > 1 else np.nan)
        agg["usd_med"] = np.log1p(agg["usd_med"])
        for col in agg.columns:
            v = np.full(n, np.nan); v[ok] = agg[col].reindex(lab[ok]).values
            f[f"{tag}_{col}"] = v
    return f.astype("float32")
