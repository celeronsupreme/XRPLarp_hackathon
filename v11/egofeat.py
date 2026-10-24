"""v5: per-account rebuilt groups ("ego groups").

Relay schemes route money through one or two middle accounts over several rounds days apart.
For every account A we cluster only A's own payments, joining two of them when the link model
says they belong together (p > th) and the link runs through A. Clusters therefore cannot chain
across accounts into giant groups. For each cluster we replay A's running balance and record
whether A ever pays out more than it has received so far (funded), how much passes through, etc.
Each payment gets these features from its sender's view and from its receiver's view."""
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def _ego_labels(df, P, th):
    s, r = df.sender.values, df.receiver.values
    sel = P[P.p > th]
    i, j, a = sel.i.values, sel.j.values, sel.acct.values
    ei = [np.c_[i, j, a]]
    sp_ = sel.samepair.values == 1           # same sender->receiver pair: link through both accounts
    other = np.where(a[sp_] == s[i[sp_]], r[i[sp_]], s[i[sp_]])
    ei.append(np.c_[i[sp_], j[sp_], other])
    E = np.vstack(ei)
    # nodes: (row, role) with role 0 = sender view, 1 = receiver view
    n = len(df)
    def node(rows, acct):
        return np.where(s[rows.astype(int)] == acct, rows.astype(int), rows.astype(int) + n)
    u = node(E[:, 0], E[:, 2]); v = node(E[:, 1], E[:, 2])
    G = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(2 * n, 2 * n))
    _, lab = connected_components(G, directed=False)
    return lab


def ego_features(df, P, thresholds=(0.8, 0.9)):
    df = df.reset_index(drop=True); n = len(df)
    s, r, t, v = df.sender.values, df.receiver.values, df.ts.values.astype(float), df.d_usd.values
    f = pd.DataFrame(index=df.index)
    for th in thresholds:
        lab = _ego_labels(df, P, th)
        # events: sender view = outflow of s, receiver view = inflow of r
        ev = pd.DataFrame({"c": lab, "t": np.r_[t, t], "sv": np.r_[-v, v], "row": np.r_[np.arange(n), np.arange(n)],
                           "view": np.r_[np.zeros(n), np.ones(n)]})
        ev = ev.sort_values(["c", "t", "sv"], ascending=[True, True, False])
        ev["bal"] = ev.groupby("c").sv.cumsum()
        ev["isin"] = ev.sv > 0
        g = ev.groupby("c")
        cs = pd.DataFrame({"size": g.size(), "n_in": g.isin.sum(), "minbal": g.bal.min(), "endbal": g.bal.last(),
                           "tin": g.sv.agg(lambda x: x[x > 0].sum()), "tout": g.sv.agg(lambda x: -x[x < 0].sum()),
                           "first_in": g.isin.first(), "span_h": g.t.agg(lambda x: (x.max() - x.min()) / 3600),
                           "rounds": g.t.agg(lambda x: 1 + int((np.diff(x) > 12 * 3600).sum()))})
        cs["n_out"] = cs["size"] - cs["n_in"]
        cs["both"] = ((cs.n_in > 0) & (cs.n_out > 0)).astype(float)
        cs["minbal_rel"] = cs.minbal / cs.tin.clip(lower=1)
        cs["funded"] = np.where(cs.both == 1, (cs.minbal_rel >= -0.02).astype(float), np.nan)
        cs["cons"] = np.where(cs.both == 1, np.minimum(cs.tin, cs.tout) / np.maximum(cs.tin, cs.tout).clip(lower=1e-9), np.nan)
        cs["endbal_rel"] = np.where(cs.both == 1, cs.endbal / cs.tin.clip(lower=1), np.nan)
        tag = f"e{int(th * 100)}"
        for view, vn in ((0, "s"), (1, "r")):
            L = lab[np.arange(n) + view * n]
            for col in ["size", "n_in", "n_out", "funded", "cons", "endbal_rel", "minbal_rel", "first_in", "span_h", "rounds"]:
                f[f"{tag}{vn}_{col}"] = cs[col].reindex(L).to_numpy().astype(float)
        # best of the two views: a payment is in a funded multi-round relay if either end is its middle account
        f[f"{tag}_funded_any"] = np.fmax(f[f"{tag}s_funded"], f[f"{tag}r_funded"])
        f[f"{tag}_maxsize"] = np.fmax(f[f"{tag}s_size"], f[f"{tag}r_size"])
    return f.astype("float32")
