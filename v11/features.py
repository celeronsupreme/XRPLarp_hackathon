"""Features for Decode & Detect. All computed within one split (wallets don't overlap between splits)."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

MIN = 60.0


def _count_window(keys, times, qkeys, qtimes, lo, hi):
    """For each query: number of events with the same key in [qtime+lo, qtime+hi]."""
    cats = pd.Index(np.unique(np.concatenate([keys, qkeys])))
    k = cats.get_indexer(keys).astype(float)
    qk = cats.get_indexer(qkeys).astype(float)
    big = (max(times.max(), qtimes.max()) - min(times.min(), qtimes.min())) + abs(lo) + abs(hi) + 10
    ev = np.sort(k * big + times)
    q = qk * big + qtimes
    return np.searchsorted(ev, q + hi, "right") - np.searchsorted(ev, q + lo, "left")


def _components(df, mask, gap_s):
    """Link payments that share an account and happen within gap_s of each other (consecutive
    per account). Returns component id per row (-1 where mask is False)."""
    idx = np.flatnonzero(mask)
    sub = df.iloc[idx]
    ev = pd.DataFrame({"row": np.concatenate([idx, idx]),
                       "acct": np.concatenate([sub["sender"].to_numpy(), sub["receiver"].to_numpy()]),
                       "t": np.concatenate([sub["ts"].to_numpy(), sub["ts"].to_numpy()])})
    ev = ev.sort_values(["acct", "t"], kind="mergesort")
    same = ev["acct"].to_numpy()[1:] == ev["acct"].to_numpy()[:-1]
    close = np.diff(ev["t"].to_numpy()) <= gap_s
    a = ev["row"].to_numpy()[:-1][same & close]
    b = ev["row"].to_numpy()[1:][same & close]
    n = len(df)
    G = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    lab = lab.astype(float)
    lab[~mask] = -1
    return lab


def _component_stats(df, lab, prefix):
    f = pd.DataFrame(index=df.index)
    ok = lab >= 0
    g = pd.DataFrame({"c": lab[ok], "ts": df["ts"].to_numpy()[ok], "s": df["sender"].to_numpy()[ok],
                      "r": df["receiver"].to_numpy()[ok], "usd": df["d_usd"].to_numpy()[ok],
                      "novel": df["pair_first"].to_numpy()[ok], "cross": df["cross"].to_numpy()[ok],
                      "da": df["d_asset"].to_numpy()[ok], "sa": df["src_asset"].to_numpy()[ok]})
    agg = g.groupby("c").agg(size=("ts", "size"), span=("ts", lambda t: t.max() - t.min()),
                             ns=("s", "nunique"), nr=("r", "nunique"), novel=("novel", "mean"),
                             usd_cv=("usd", lambda v: v.std() / v.mean() if len(v) > 1 else 0))
    agg["nacct"] = pd.concat([g[["c", "s"]].rename(columns={"s": "a"}), g[["c", "r"]].rename(columns={"r": "a"})]
                             ).groupby("c")["a"].nunique()
    agg["max_out"] = g.groupby(["c", "s"]).size().groupby("c").max()
    agg["max_in"] = g.groupby(["c", "r"]).size().groupby("c").max()
    agg["cyclic"] = (agg["size"] >= agg["nacct"]).astype(int)   # edges >= nodes -> contains a loop
    agg["gap_mean_min"] = agg["span"] / (agg["size"] - 1).clip(lower=1) / MIN
    agg["cross_share"] = g.groupby("c")["cross"].mean()
    agg["n_assets"] = pd.concat([g[["c", "da"]].rename(columns={"da": "a"}), g[["c", "sa"]].rename(columns={"sa": "a"})]
                                ).groupby("c")["a"].nunique()
    gs = g.sort_values("ts")
    agg["first_last"] = np.log((gs.groupby("c")["usd"].last() + 1) / (gs.groupby("c")["usd"].first() + 1))
    agg["usd_med"] = np.log1p(g.groupby("c")["usd"].median())
    for col in ["size", "span", "ns", "nr", "nacct", "novel", "usd_cv", "max_out", "max_in", "cyclic", "gap_mean_min",
                "cross_share", "n_assets", "first_last", "usd_med"]:
        v = np.full(len(df), np.nan)
        v[ok] = agg[col].reindex(lab[ok]).to_numpy()
        f[f"{prefix}_{col}"] = v
    return f


def _flow_features(df, nov):
    """'Money moved on' links between first-time-pair payments: payment i (u->v) is linked to a later
    payment j (v->w) by the receiver v within W hours. Amount-aware variants require j to carry a
    similar amount (0.5-1.05 x i) for chains, or pieces that add up for splits / merges."""
    f = pd.DataFrame(index=df.index)
    n = len(df)
    rows = np.flatnonzero(nov)
    usd = df["d_usd"].to_numpy()
    t = df["ts"].to_numpy().astype(float)
    inc = pd.DataFrame({"i": rows, "acct": df["receiver"].to_numpy()[rows], "ti": t[rows], "vi": usd[rows]})
    out = pd.DataFrame({"j": rows, "acct": df["sender"].to_numpy()[rows], "tj": t[rows], "vj": usd[rows]})
    L = inc.merge(out, on="acct")
    L = L[(L.tj > L.ti) & (L.tj <= L.ti + 4 * 3600) & (L.i != L.j)].copy()
    L["ratio"] = L.vj / L.vi.clip(lower=1e-9)
    L["dh"] = (L.tj - L.ti) / 3600
    for hrs in (1, 4):
        Lh = L[L.dh <= hrs]
        f[f"fl_fwd_{hrs}h"] = np.bincount(Lh.i, minlength=n)
        f[f"fl_back_{hrs}h"] = np.bincount(Lh.j, minlength=n)
        sim = Lh[(Lh.ratio >= 0.5) & (Lh.ratio <= 1.05)]
        f[f"fl_fwd_sim_{hrs}h"] = np.bincount(sim.i, minlength=n)
        f[f"fl_back_sim_{hrs}h"] = np.bincount(sim.j, minlength=n)
        pieces = Lh[Lh.ratio <= 1.05]
        f[f"fl_split_sum_{hrs}h"] = np.bincount(pieces.i, weights=pieces.ratio, minlength=n)
        merges = Lh[Lh.ratio >= 0.95]
        f[f"fl_merge_sum_{hrs}h"] = np.bincount(merges.j, weights=1 / merges.ratio, minlength=n)
    # chains of similar-amount hops (longest path forward / backward)
    S = L[(L.ratio >= 0.5) & (L.ratio <= 1.05)]
    i, j = S.i.to_numpy(), S.j.to_numpy()
    fwd, back = np.zeros(n), np.zeros(n)
    for _ in range(15):
        nf = fwd.copy(); b = pd.Series(fwd[j] + 1).groupby(i).max()
        nf[b.index.to_numpy()] = np.maximum(nf[b.index.to_numpy()], b.to_numpy())
        nb = back.copy(); b = pd.Series(back[i] + 1).groupby(j).max()
        nb[b.index.to_numpy()] = np.maximum(nb[b.index.to_numpy()], b.to_numpy())
        if (nf == fwd).all() and (nb == back).all():
            break
        fwd, back = nf, nb
    f["fl_chain_after"], f["fl_chain_before"] = fwd, back
    f["fl_chain_len"] = fwd + back
    # components over all flow links (4h), with shape stats
    G = sp.coo_matrix((np.ones(len(L)), (L.i.to_numpy(), L.j.to_numpy())), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    sizes = np.bincount(lab)
    linked = (np.bincount(L.i, minlength=n) + np.bincount(L.j, minlength=n)) > 0
    f["fl_comp_size"] = np.where(linked, sizes[lab], 0)
    comp = pd.DataFrame({"c": lab, "t": t, "usd": usd, "s": df["sender"].to_numpy(), "r": df["receiver"].to_numpy()})[linked]
    agg = comp.groupby("c").agg(span=("t", lambda x: (x.max() - x.min()) / 3600), usd_med=("usd", "median"),
                                ns=("s", "nunique"), nr=("r", "nunique"))
    for col in ["span", "usd_med", "ns", "nr"]:
        v = np.full(n, np.nan)
        v[linked] = agg[col].reindex(lab[linked]).to_numpy()
        f[f"fl_comp_{col}"] = v
    return f


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    t = df["ts"].to_numpy().astype(float)
    s, r = df["sender"].to_numpy(), df["receiver"].to_numpy()
    f = pd.DataFrame(index=df.index)

    # --- amount & payment shape
    f["log_usd"] = np.log1p(df["d_usd"])
    f["log_src_usd"] = np.log1p(df["src_usd"])
    f["conv_cost"] = df["conv_cost"].clip(-1, 1)
    f["cross"] = df["cross"]
    f["n_legs"] = df["n_legs"]
    f["round_usd"] = (np.abs(df["d_usd"] - df["d_usd"].round(-1)) < 0.5).astype(int)
    f["round_units"] = (np.abs(df["d_units"] - df["d_units"].round(0)) < 1e-9).astype(int)
    # v6: planted motifs are spread evenly through the day, background is daytime-heavy
    f["night"] = ((t % 86400) < 6 * 3600).astype(int)

    # --- relationship novelty
    pair = pd.Series(s + ">" + r)
    f["pair_n"] = pair.map(pair.value_counts()).to_numpy()
    f["pair_prior"] = pair.groupby(pair).cumcount().to_numpy()
    df["pair_first"] = (f["pair_n"] == 1).astype(int).to_numpy()
    f["pair_first"] = df["pair_first"]
    rev = pd.Series(r + ">" + s)
    f["rev_n"] = rev.map(pair.value_counts()).fillna(0).to_numpy()
    snd_novel = pd.Series(df["pair_first"].to_numpy()).groupby(s).mean()
    rcv_novel = pd.Series(df["pair_first"].to_numpy()).groupby(r).mean()
    f["s_novel_rate"] = pd.Series(s).map(snd_novel).to_numpy()
    f["r_novel_rate"] = pd.Series(r).map(rcv_novel).to_numpy()

    # --- account activity level
    deg_out = pd.Series(s).value_counts()
    deg_in = pd.Series(r).value_counts()
    f["s_out_n"] = pd.Series(s).map(deg_out).to_numpy()
    f["s_in_n"] = pd.Series(s).map(deg_in).fillna(0).to_numpy()
    f["r_in_n"] = pd.Series(r).map(deg_in).to_numpy()
    f["r_out_n"] = pd.Series(r).map(deg_out).fillna(0).to_numpy()
    f["s_cpty"] = pd.Series(s).map(pd.Series(r).groupby(s).nunique()).to_numpy()
    f["r_cpty"] = pd.Series(r).map(pd.Series(s).groupby(r).nunique()).to_numpy()

    # --- bursts around this payment (all payments, and novel-pair payments only)
    nov = df["pair_first"].to_numpy() == 1
    for w in (2, 5, 15, 60):
        W = w * MIN
        f[f"s_out_{w}m"] = _count_window(s, t, s, t, -W, W)
        f[f"r_in_{w}m"] = _count_window(r, t, r, t, -W, W)
        f[f"s_in_before_{w}m"] = _count_window(r, t, s, t, -W, 0)
        f[f"r_out_after_{w}m"] = _count_window(s, t, r, t, 0, W)
        f[f"s_out_nov_{w}m"] = _count_window(s[nov], t[nov], s, t, -W, W)
        f[f"r_in_nov_{w}m"] = _count_window(r[nov], t[nov], r, t, -W, W)
        f[f"s_in_nov_before_{w}m"] = _count_window(r[nov], t[nov], s, t, -W, 0)
        f[f"r_out_nov_after_{w}m"] = _count_window(s[nov], t[nov], r, t, 0, W)

    # --- longer windows for slower ("medium") schemes: novel-pair activity only
    for w in (120, 240, 480):
        W = w * MIN
        f[f"s_out_nov_{w}m"] = _count_window(s[nov], t[nov], s, t, -W, W)
        f[f"r_in_nov_{w}m"] = _count_window(r[nov], t[nov], r, t, -W, W)
        f[f"s_in_nov_before_{w}m"] = _count_window(r[nov], t[nov], s, t, -W, 0)
        f[f"r_out_nov_after_{w}m"] = _count_window(s[nov], t[nov], r, t, 0, W)

    # --- time to neighbouring activity (minutes, capped)
    order = np.argsort(t, kind="mergesort")
    out_t = pd.DataFrame({"acct": s, "t": t}).sort_values("t")
    q = pd.DataFrame({"acct": r, "t": t, "row": np.arange(len(df))}).sort_values("t")
    m = pd.merge_asof(q, out_t.rename(columns={"t": "t_next"}).assign(t=lambda x: x["t_next"]),
                      on="t", by="acct", direction="forward", allow_exact_matches=False).sort_values("row")
    f["r_min_to_next_send"] = ((m["t_next"] - m["t"]) / MIN).clip(upper=24 * 60).fillna(24 * 60).to_numpy()
    in_t = pd.DataFrame({"acct": r, "t": t}).sort_values("t")
    q2 = pd.DataFrame({"acct": s, "t": t, "row": np.arange(len(df))}).sort_values("t")
    m2 = pd.merge_asof(q2, in_t.rename(columns={"t": "t_prev"}).assign(t=lambda x: x["t_prev"]),
                       on="t", by="acct", direction="backward", allow_exact_matches=False).sort_values("row")
    f["s_min_since_recv"] = ((m2["t"] - m2["t_prev"]) / MIN).clip(upper=24 * 60).fillna(24 * 60).to_numpy()

    # --- temporal motif components (linked payments sharing accounts within a gap)
    for gap in (3, 10, 30, 90):
        lab_all = _components(df, np.ones(len(df), bool), gap * MIN)
        f = pd.concat([f, _component_stats(df, lab_all, f"cA{gap}")], axis=1)
    for gap in (3, 10, 20, 45, 90, 180):
        lab_nov = _components(df, nov, gap * MIN)
        f = pd.concat([f, _component_stats(df, lab_nov, f"cN{gap}")], axis=1)
    f = pd.concat([f, _flow_features(df, nov)], axis=1)
    return f.astype("float32")
