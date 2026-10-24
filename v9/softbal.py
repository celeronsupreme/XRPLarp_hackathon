"""Version 9: soft cumulative balance histories and label-free account balance trajectories.

softflow.py asks "how much probable same-scenario money arrived before this outflow?" but values the
outgoing side with the delivered amount and lets every outflow claim the same inflow. This file adds

1. soft_balance_features: for each payment, replay the shared account's events that the link model
   connects to it (within 8 h), weighting each event by its link probability, valuing inflows at the
   delivered USD and outflows at the ACTUAL source debit (first executed leg). From the sender's view of
   an outflow: weighted funds received before it minus weighted funds already spent before it, the
   prefix minimum of that trajectory (how much outside funding the history would require), the
   amount-weighted funding delay and the mass of competing earlier outflows. From the receiver's view
   of an inflow: the mirror (how much of it is passed on, after how long, how much other money pools
   with it, whether the account's balance after it ever dips below what this inflow brought).

2. account_balance_features: no links at all. Each account's running USD position over the whole
   observed week (inflows at delivered value, outflows at source debit), evaluated at each payment:
   balance before an outflow relative to its debit, balance after an inflow relative to its value,
   the account's week-long minimum balance and pass-through ratio, and amount sums in +/-24 h windows.

All of this is computed within one split and uses no labels.
"""
import numpy as np
import pandas as pd


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _expand_samepair(P, df):
    """Same-pair links share both accounts; keep one row per (link, shared account)."""
    s, r = df.sender.values, df.receiver.values
    sp_ = P.samepair.values == 1
    if not sp_.any():
        return P
    Q = P[sp_].copy()
    a = Q.acct.values
    Q["acct"] = np.where(a == s[Q.i.values], r[Q.i.values], s[Q.i.values])
    return pd.concat([P, Q], ignore_index=True)


def soft_balance_features(df, pairs_p):
    df = df.reset_index(drop=True)
    n = len(df)
    s, r = df.sender.values, df.receiver.values
    t = df.ts.values.astype(float)
    vin = df.d_usd.values                       # value an inflow brings
    vout = np.nan_to_num(df.src_usd.values, nan=0.0)   # value an outflow actually costs
    vout = np.where(vout > 0, vout, vin)
    P = _expand_samepair(pairs_p[pairs_p.p > 0], df)
    i, j, a, p = P.i.values, P.j.values, P.acct.values, P.p.values
    i_out, j_out = s[i] == a, s[j] == a            # role at the shared account
    dt = (t[j] - t[i]) / 3600.0
    f = pd.DataFrame(index=df.index)

    # ---------- sender view of j (j is an outflow of a; i precedes it at a) ----------
    m = j_out
    jj, ii, pp, aa = j[m], i[m], p[m], a[m]
    io = i_out[m]
    sv = np.where(io, -pp * vout[ii], pp * vin[ii])            # signed weighted value of the predecessor
    fund = np.bincount(jj, weights=np.where(io, 0, pp * vin[ii]), minlength=n)
    spent = np.bincount(jj, weights=np.where(io, pp * vout[ii], 0), minlength=n)
    n_out_p = np.bincount(jj, weights=np.where(io, pp, 0), minlength=n)
    bal = fund - spent
    f["sbs_fund_debit_ratio"] = np.log1p(fund) - np.log1p(vout)
    f["sbs_spent_vs_fund"] = np.log1p(spent) - np.log1p(fund)
    f["sbs_bal_slog"] = _slog(bal)
    f["sbs_cover"] = np.clip(np.maximum(bal, 0) / np.maximum(vout, 1e-6), 0, 1)
    f["sbs_deficit_rel"] = np.clip((vout - bal) / np.maximum(vout, 1e-6), -5, 5)
    f["sbs_prior_out_p"] = n_out_p
    # prefix trajectory: sort predecessors by time, cumulative signed value, min including j's own debit
    o = np.lexsort((t[ii], jj))
    jj_o, sv_o = jj[o], sv[o]
    cum = pd.Series(sv_o).groupby(jj_o).cumsum().to_numpy()
    pre_min = pd.Series(cum).groupby(jj_o).min().reindex(range(n)).to_numpy()
    pre_min = np.fmin(np.nan_to_num(pre_min, nan=0.0), 0.0)       # empty history -> 0
    end_min = np.minimum(pre_min, bal - vout)
    req = np.maximum(0, -end_min)
    f["sbs_req_fund_rel"] = req / np.maximum(fund + vout, 1e-6)
    f["sbs_req_fund_slog"] = _slog(req)
    # amount-weighted funding delay (hours) and best single funding explanation
    wv = np.where(io, 0, pp * vin[ii])
    num = np.bincount(jj, weights=wv * dt[m], minlength=n)
    f["sbs_fund_delay_h"] = np.where(fund > 0, num / np.maximum(fund, 1e-9), np.nan)
    best = pp * np.minimum(vin[ii], vout[jj]) / np.maximum(vout[jj], 1e-6)
    best = np.where(io, 0, best)
    f["sbs_best_single"] = pd.Series(best).groupby(jj).max().reindex(range(n)).fillna(0).to_numpy()

    # money arriving at a only AFTER j left (i = j's role reversed: here j is the first endpoint)
    m2 = i_out & ~j_out                                        # i outflow of a, then j inflow to a
    late = np.bincount(i[m2], weights=p[m2] * vin[j[m2]], minlength=n)
    f["sbs_late_fund_ratio"] = np.log1p(late) - np.log1p(vout)
    f["sbs_late_minus_fund"] = np.log1p(late) - np.log1p(fund)

    # ---------- receiver view of i (i is an inflow to a; j follows it at a) ----------
    m = ~i_out
    ii, jj, pp = i[m], j[m], p[m]
    jo = j_out[m]
    fwd = np.bincount(ii, weights=np.where(jo, pp * vout[jj], 0), minlength=n)
    coin = np.bincount(ii, weights=np.where(jo, 0, pp * vin[jj]), minlength=n)
    n_coin_p = np.bincount(ii, weights=np.where(jo, 0, pp), minlength=n)
    f["sbr_fwd_debit_ratio"] = np.log1p(fwd) - np.log1p(vin)
    f["sbr_fwd_cover"] = np.clip(fwd / np.maximum(vin, 1e-6), 0, 1)
    f["sbr_coin_ratio"] = np.log1p(coin) - np.log1p(vin)
    f["sbr_fwd_share"] = fwd / np.maximum(vin + coin, 1e-6)
    f["sbr_coin_p"] = n_coin_p
    sv = np.where(jo, -pp * vout[jj], pp * vin[jj])
    o = np.lexsort((t[jj], ii))
    ii_o, sv_o = ii[o], sv[o]
    cum = pd.Series(sv_o).groupby(ii_o).cumsum().to_numpy()
    post_min = pd.Series(cum).groupby(ii_o).min().reindex(range(n)).to_numpy()
    post_min = np.fmin(np.nan_to_num(post_min, nan=0.0), 0.0)
    f["sbr_minbal_rel"] = np.clip((vin + post_min) / np.maximum(vin, 1e-6), -5, 1)
    wv = np.where(jo, pp * vout[jj], 0)
    num = np.bincount(ii, weights=wv * dt[m], minlength=n)
    f["sbr_fwd_delay_h"] = np.where(fwd > 0, num / np.maximum(fwd, 1e-9), np.nan)
    # outflows of a before i arrived (a was already paying out)
    m3 = i_out & ~j_out
    prior = np.bincount(j[m3], weights=p[m3] * vout[i[m3]], minlength=n)
    f["sbr_prior_out_ratio"] = np.log1p(prior) - np.log1p(vin)
    return f.astype("float32")


def account_balance_features(df):
    df = df.reset_index(drop=True)
    n = len(df)
    s, r = df.sender.values, df.receiver.values
    t = df.ts.values.astype(float)
    vin = df.d_usd.values
    vout = np.nan_to_num(df.src_usd.values, nan=0.0)
    vout = np.where(vout > 0, vout, vin)
    ev = pd.DataFrame({"row": np.r_[np.arange(n), np.arange(n)], "acct": np.r_[s, r],
                       "t": np.r_[t, t], "ord": np.r_[np.arange(n), np.arange(n)],
                       "sv": np.r_[-vout, vin], "view": np.r_[np.zeros(n, int), np.ones(n, int)]})
    # ledger order: df is sorted by (ts, tx_index); an inflow and outflow of the same account in the same
    # transaction cannot happen, so ordering by row index is ledger order
    ev = ev.sort_values(["acct", "ord"], kind="mergesort").reset_index(drop=True)
    g = ev.groupby("acct", sort=False)
    ev["bal_after"] = g.sv.cumsum()
    ev["bal_before"] = ev.bal_after - ev.sv
    ev["in_cum"] = g.sv.transform(lambda x: x.clip(lower=0).cumsum())
    tot_in = g.sv.transform(lambda x: x.clip(lower=0).sum())
    tot_out = g.sv.transform(lambda x: (-x.clip(upper=0)).sum())
    minbal = g.bal_after.transform("min").clip(upper=0)
    ev["minbal_rel"] = minbal / tot_in.clip(lower=1)
    ev["pass_through"] = np.minimum(tot_in, tot_out) / np.maximum(tot_in, tot_out).clip(lower=1e-9)
    ev["in_share_before"] = ev.in_cum / tot_in.clip(lower=1e-9)
    ev["week_net_slog"] = _slog(tot_in - tot_out)
    # +/- 24h amount windows per account
    acct_codes = pd.factorize(ev.acct)[0]
    big = (t.max() - t.min()) + 3 * 86400
    key = acct_codes * big + ev.t.values
    order = np.argsort(key, kind="mergesort")
    ks = key[order]
    cin = np.concatenate([[0], np.cumsum(ev.sv.values[order].clip(min=0))])
    cout = np.concatenate([[0], np.cumsum((-ev.sv.values[order]).clip(min=0))])
    def win(lo, hi, c):
        a = np.searchsorted(ks, key + lo, "left"); b = np.searchsorted(ks, key + hi, "right")
        return c[b] - c[a]
    W = 24 * 3600
    ev["in_24h_before"] = win(-W, 0, cin)
    ev["out_24h_after"] = win(0, W, cout)
    ev["in_24h_after"] = win(0, W, cin)
    ev["out_24h_before"] = win(-W, 0, cout)
    f = pd.DataFrame(index=df.index)
    S = ev[ev.view == 0].set_index("row").reindex(range(n))
    R = ev[ev.view == 1].set_index("row").reindex(range(n))
    # sender view (payment is an outflow of s)
    f["ab_s_bal_before_slog"] = _slog(S.bal_before.values)
    f["ab_s_cover"] = np.clip(np.maximum(S.bal_before.values, 0) / np.maximum(vout, 1e-6), 0, 1)
    f["ab_s_minbal_rel"] = S.minbal_rel.values
    f["ab_s_pass_through"] = S.pass_through.values
    f["ab_s_in_share_before"] = S.in_share_before.values
    f["ab_s_week_net_slog"] = S.week_net_slog.values
    f["ab_s_in24h_ratio"] = np.log1p(S.in_24h_before.values) - np.log1p(vout)
    f["ab_s_in24h_after_ratio"] = np.log1p(S.in_24h_after.values - vin * 0) - np.log1p(vout)
    # receiver view (payment is an inflow to r)
    f["ab_r_bal_after_rel"] = np.clip(R.bal_after.values / np.maximum(vin, 1e-6), -5, 5)
    f["ab_r_minbal_rel"] = R.minbal_rel.values
    f["ab_r_pass_through"] = R.pass_through.values
    f["ab_r_in_share_before"] = R.in_share_before.values
    f["ab_r_week_net_slog"] = R.week_net_slog.values
    f["ab_r_out24h_ratio"] = np.log1p(R.out_24h_after.values) - np.log1p(vin)
    f["ab_r_out24h_before_ratio"] = np.log1p(R.out_24h_before.values) - np.log1p(vin)
    return f.astype("float32")
