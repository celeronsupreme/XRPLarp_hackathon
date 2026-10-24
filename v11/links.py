"""Learned 'same scenario?' links between payments that share an account within WINDOW hours."""
import numpy as np, pandas as pd

WINDOW_H = 8

def candidate_pairs(df):
    df = df.reset_index(drop=True)
    t = df.ts.values.astype(float); s = df.sender.values; r = df.receiver.values
    pair = pd.Series(s + ">" + r); pn = pair.map(pair.value_counts()).values
    ev = pd.concat([pd.DataFrame({"row": np.arange(len(df)), "acct": s, "role": 0, "t": t}),
                    pd.DataFrame({"row": np.arange(len(df)), "acct": r, "role": 1, "t": t})])
    ev = ev.sort_values(["acct", "t"]).reset_index(drop=True)
    # account activity (events in +/-4h) for context
    out = []
    acct = ev.acct.values; tt = ev.t.values; role = ev.role.values; row = ev.row.values
    n = len(ev)
    for k in range(1, 12):  # pair each event with its next k-th event of the same account
        a = np.arange(n - k); b = a + k
        ok = (acct[a] == acct[b]) & ((tt[b] - tt[a]) <= WINDOW_H * 3600) & (row[a] != row[b])
        out.append(pd.DataFrame({"i": row[a][ok], "j": row[b][ok], "ri": role[a][ok], "rj": role[b][ok],
                                 "dt": (tt[b][ok] - tt[a][ok]) / 3600, "acct": acct[a][ok], "k": k}))
    # v5: cross-round links. Relay schemes repeat the same sender->receiver payment in rounds days
    # apart, beyond the 8h window; link each payment to the next 1-2 occurrences of the same pair.
    pr = pd.DataFrame({"row": np.arange(len(df)), "pair": pair.values, "t": t}).sort_values(["pair", "t"])
    prow, ppair, pt = pr.row.values, pr.pair.values, pr.t.values
    for k in (1, 2):
        a = np.arange(len(pr) - k); b = a + k
        ok = (ppair[a] == ppair[b]) & ((pt[b] - pt[a]) > WINDOW_H * 3600)
        out.append(pd.DataFrame({"i": prow[a][ok], "j": prow[b][ok], "ri": 0, "rj": 0,
                                 "dt": (pt[b][ok] - pt[a][ok]) / 3600, "acct": s[prow[a][ok]], "k": 100 + k}))
    P = pd.concat(out, ignore_index=True).drop_duplicates(["i", "j"])
    P["samepair"] = (pair.values[P.i] == pair.values[P.j]).astype(int)
    P["xr"] = (P.k > 100).astype(int)       # cross-round link (beyond the 8h window)
    v = df.d_usd.values
    P["pattern"] = P.ri * 2 + P.rj          # 0 send-send, 1 send-then-recv, 2 recv-then-send (flow), 3 recv-recv
    P["pn_i"], P["pn_j"] = pn[P.i], pn[P.j]
    P["lratio"] = np.log((v[P.j] + 1) / (v[P.i] + 1))
    P["lv_i"], P["lv_j"] = np.log1p(v[P.i]), np.log1p(v[P.j])
    P["same_dasset"] = (df.d_asset.values[P.i] == df.d_asset.values[P.j]).astype(int)
    P["same_sasset"] = (df.src_asset.values[P.i] == df.src_asset.values[P.j]).astype(int)
    P["legs_i"], P["legs_j"] = df.n_legs.values[P.i], df.n_legs.values[P.j]
    P["cross_i"], P["cross_j"] = df.cross.values[P.i], df.cross.values[P.j]
    acnt = ev.groupby("acct").size()
    P["acct_events"] = P.acct.map(acnt).values
    return P

FEATS = ["pattern", "dt", "k", "pn_i", "pn_j", "lratio", "lv_i", "lv_j", "same_dasset", "same_sasset",
         "legs_i", "legs_j", "cross_i", "cross_j", "acct_events", "samepair"]


LINK_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0)


def same_scenario(df, P):
    sc = df["scenario_id"].values
    return np.asarray((sc[P.i] == sc[P.j]) & pd.notna(sc[P.i])).astype(int)


def train_link_probs(df_train, df_test):
    """Link probabilities for train (out-of-fold by scenario halves) and test (model on all train pairs)."""
    import lightgbm as lgb
    Ptr = candidate_pairs(df_train)
    ytr = same_scenario(df_train.reset_index(drop=True), Ptr)
    scen = pd.Series(df_train["scenario_id"].fillna(df_train["hash"])).values
    u = np.unique(scen)
    half = set(np.random.default_rng(0).choice(u, len(u) // 2, replace=False))
    inA = np.array([x in half for x in scen])
    Ptr["p"] = np.nan
    for side in (True, False):
        trn = (inA[Ptr.i] == side) & (inA[Ptr.j] == side)
        m = lgb.LGBMClassifier(**LINK_PARAMS).fit(Ptr.loc[trn, FEATS], ytr[trn])
        tgt = (inA[Ptr.i] != side) | (inA[Ptr.j] != side)
        Ptr.loc[tgt, "p"] = m.predict_proba(Ptr.loc[tgt, FEATS])[:, 1]
    Pte = candidate_pairs(df_test)
    m = lgb.LGBMClassifier(**LINK_PARAMS).fit(Ptr[FEATS], ytr)
    Pte["p"] = m.predict_proba(Pte[FEATS])[:, 1]
    cols = ["i", "j", "p", "acct", "samepair", "xr"]
    return Ptr[cols], Pte[cols]
