"""Version 8: link-probability-weighted neighbour features.

Every earlier gain came from replacing a hard group with a probability-weighted sum (soft funding, p-weighted
smoothing). This file applies the same idea to features: each payment receives the p-weighted mean of a few
key features of the payments it is linked to, computed separately over

- within-8-hour links (pattern-aware: neighbours that are inflows to the shared account vs outflows), and
- cross-round same-pair links (the same sender->receiver payment in earlier or later rounds, days apart).

The cross-round half carries soft-funding evidence across relay rounds: an outflow in round 3 inherits what
the same-pair outflows in rounds 1 and 2 saw about money arriving at the account before them, which no
within-8-hour feature can express. (Weber et al. 2019 one-hop feature aggregation; Savage et al. 2016.)
"""
import numpy as np
import pandas as pd

KEY = ["sf_fund_p", "sf_prepaid_p", "sf_out_net", "sf_fwd_p", "sf_late_p", "sf_in_net", "sf_fund_ratio", "sf_fwd_ratio",
       "pair_first", "s_min_since_recv", "r_min_to_next_send", "s_out_nov_15m", "r_in_nov_15m", "log_usd"]


def _wmean(n, i, j, p, F, cols, tag):
    """For each payment, p-weighted mean over its neighbours (both directions of every pair) of F[cols]."""
    out = pd.DataFrame(index=range(n))
    den = np.bincount(i, weights=p, minlength=n) + np.bincount(j, weights=p, minlength=n)
    out[f"{tag}_wdeg"] = den
    for c in cols:
        v = np.nan_to_num(F[c].to_numpy(dtype=float))
        num = np.bincount(i, weights=p * v[j], minlength=n) + np.bincount(j, weights=p * v[i], minlength=n)
        out[f"{tag}_{c}"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return out


def neighbour_features(df, F, P, pmin=0.3):
    """df: tidy payments; F: feature table so far (must contain KEY columns); P: all candidate pairs with p and xr."""
    df = df.reset_index(drop=True)
    n = len(df)
    cols = [c for c in KEY if c in F.columns]
    sel = P[P.p > pmin]
    parts = []
    # within-8h links, split by whether the neighbour is on the "money came in" side or the "money went out" side
    w = sel[sel.xr == 0]
    s, r = df.sender.to_numpy(), df.receiver.to_numpy()
    i, j, p = w.i.to_numpy(), w.j.to_numpy(), w.p.to_numpy()
    parts.append(_wmean(n, i, j, p, F, cols, "nb8"))
    flow = r[i] == s[j]            # i into a then j out of a
    parts.append(_wmean(n, i[flow], j[flow], p[flow], F, cols, "nb8flow"))
    # cross-round same-pair links
    x = sel[sel.xr == 1]
    parts.append(_wmean(n, x.i.to_numpy(), x.j.to_numpy(), x.p.to_numpy(), F, cols, "nbxr"))
    out = pd.concat(parts, axis=1)
    # contrast: own soft funding vs what same-pair earlier/later rounds saw
    if "sf_fund_p" in F.columns:
        out["nbxr_fund_gap"] = np.nan_to_num(F["sf_fund_p"].to_numpy(dtype=float)) - out["nbxr_sf_fund_p"].to_numpy()
    out.index = df.index
    return out.astype("float32")


def spread_features(df, lab_sets):
    """Gap-regularity features per group labelling. lab_sets: dict tag -> label array (-1 = no group).
    AMLSim-style schedules plant payments at fixed intervals; innocent traffic is irregular."""
    df = df.reset_index(drop=True)
    n = len(df)
    t = df.ts.to_numpy(dtype=float)
    out = pd.DataFrame(index=df.index)
    for tag, lab in lab_sets.items():
        ok = lab >= 0
        g = pd.DataFrame({"c": lab[ok], "t": t[ok]}).sort_values(["c", "t"])
        d = g.groupby("c").t.diff().dropna() / 60.0
        agg = d.groupby(g.loc[d.index, "c"]).agg(gap_std="std", gap_min="min", gap_max="max", gap_mean="mean")
        agg["gap_cv"] = agg.gap_std / agg.gap_mean.clip(lower=1e-6)
        for c in ["gap_std", "gap_min", "gap_max", "gap_cv"]:
            v = np.full(n, np.nan); v[ok] = agg[c].reindex(lab[ok]).to_numpy()
            out[f"{tag}_{c}"] = v
    return out.astype("float32")


# ---------------------------------------------------------------------------------------------
# Version 9: typed (role- and direction-aware) channels, worst-round evidence, two-hop cross-round
# ---------------------------------------------------------------------------------------------
TYPED_KEY = ["sf_fund_p", "sf_prepaid_p", "sf_out_net", "sf_fwd_p", "sf_in_net", "pair_first", "log_usd",
             "sbs_cover", "sbs_req_fund_rel", "sbs_deficit_rel", "sbr_fwd_cover", "sbr_minbal_rel", "ab_s_cover", "ab_r_bal_after_rel"]
WORST_KEY = ["sf_fund_p", "sf_prepaid_p", "sf_out_net", "sbs_cover", "sbs_req_fund_rel", "nb8flow_sf_fund_p", "nb8_sf_out_net"]
TWOHOP_KEY = ["nb8flow_sf_fund_p", "nb8flow_sf_prepaid_p", "nb8_sf_out_net", "nb8flow_sf_fwd_p", "nbxr_sf_fund_p"]


def _wdir(n, src, dst, p, F, cols, tag):
    """dst receives the p-weighted mean of F[cols] over its src neighbours (one direction only)."""
    out = pd.DataFrame(index=range(n))
    den = np.bincount(dst, weights=p, minlength=n)
    out[f"{tag}_wdeg"] = den
    for c in cols:
        v = np.nan_to_num(F[c].to_numpy(dtype=float))
        num = np.bincount(dst, weights=p * v[src], minlength=n)
        out[f"{tag}_{c}"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return out


def _wminmax(n, i, j, p, F, cols, tag):
    """Weighted min / max over neighbours (both directions): worst-round and best-round evidence."""
    out = pd.DataFrame(index=range(n))
    a = np.concatenate([i, j]); b = np.concatenate([j, i])
    for c in cols:
        if c not in F.columns:
            continue
        v = F[c].to_numpy(dtype=float)[b]
        ok = ~np.isnan(v)
        g = pd.Series(v[ok]).groupby(a[ok])
        out[f"{tag}_min_{c}"] = g.min().reindex(range(n)).to_numpy()
        out[f"{tag}_max_{c}"] = g.max().reindex(range(n)).to_numpy()
    return out


def typed_neighbour_features(df, F, P, pmin=0.3):
    """Direction- and role-aware one-hop channels over within-8h links, min/max over cross-round links,
    and a two-hop summary (what the same pair's other rounds saw in THEIR within-8h neighbourhoods)."""
    df = df.reset_index(drop=True)
    n = len(df)
    s, r = df.sender.to_numpy(), df.receiver.to_numpy()
    cols = [c for c in TYPED_KEY if c in F.columns]
    sel = P[P.p > pmin]
    w = sel[sel.xr == 0]
    i, j, p, a = w.i.to_numpy(), w.j.to_numpy(), w.p.to_numpy(), w.acct.to_numpy()
    i_out, j_out = s[i] == a, s[j] == a
    parts = []
    flow = ~i_out & j_out                      # i into a, then j out of a
    parts.append(_wdir(n, i[flow], j[flow], p[flow], F, cols, "nbIN2OUT"))   # outflow hears its funders
    parts.append(_wdir(n, j[flow], i[flow], p[flow], F, cols, "nbOUT2IN"))   # inflow hears its forwarders
    coout = i_out & j_out
    parts.append(_wmean(n, i[coout], j[coout], p[coout], F, cols, "nbCOOUT"))
    coin = ~i_out & ~j_out
    parts.append(_wmean(n, i[coin], j[coin], p[coin], F, cols, "nbCOIN"))
    pre = i_out & ~j_out                       # i out of a, then j into a (money arrived after it left)
    parts.append(_wmean(n, i[pre], j[pre], p[pre], F, cols, "nbPRE"))
    x = sel[sel.xr == 1]
    xi, xj, xp = x.i.to_numpy(), x.j.to_numpy(), x.p.to_numpy()
    parts.append(_wminmax(n, xi, xj, xp, F, WORST_KEY, "nbxr"))
    parts.append(_wmean(n, xi, xj, xp, F, [c for c in TWOHOP_KEY if c in F.columns], "nbxr2"))
    out = pd.concat(parts, axis=1)
    out.index = df.index
    return out.astype("float32")
