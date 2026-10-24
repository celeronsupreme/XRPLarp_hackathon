"""Soft funding features from link probabilities (v5 option --soft-flow).

Hard groups break the relay 'funded' test: one wrong member makes a funded group look unfunded. Instead,
use each link probability as a soft 'same scenario' weight between two payments at one account:

- for a payment OUT of account a: probable same-scenario money INTO a just before it (funded) and just
  after it (paid out before being funded: the innocent look-alike's tell);
- for a payment INTO account a: probable same-scenario money OUT of a after it (passed on) and before it.

Candidate pairs (links.py) are ordered in time: i before j at a shared account. pattern 2 = i into a,
then j out of a; pattern 1 = i out of a, then j into a.
"""
import numpy as np
import pandas as pd


def _pattern(P, df):
    """Recompute the pattern for each candidate pair from the payments themselves."""
    s, r = df.sender.values, df.receiver.values
    i, j = P.i.values, P.j.values
    flow = (r[i] == s[j])            # i into a, then j out of a   (a = r[i] = s[j])
    pre = (s[i] == r[j])             # i out of a, then j into a
    return flow, pre


def soft_flow_features(df, pairs_p):
    df = df.reset_index(drop=True)
    n = len(df)
    v = df.d_usd.values
    P = pairs_p
    flow, pre = _pattern(P, df)
    p = P.p.values
    i, j = P.i.values, P.j.values
    f = pd.DataFrame(index=df.index)
    # payment OUT of a (j in a flow pair): funding received before it
    f["sf_fund_p"] = np.bincount(j[flow], weights=p[flow], minlength=n)
    f["sf_fund_pmax"] = pd.Series(p[flow]).groupby(j[flow]).max().reindex(range(n)).fillna(0).values
    fund_usd = np.bincount(j[flow], weights=p[flow] * v[i[flow]], minlength=n)
    f["sf_fund_ratio"] = np.log1p(fund_usd) - np.log1p(v)
    # payment OUT of a (i in a pre pair): money arriving at a only AFTER it left
    f["sf_prepaid_p"] = np.bincount(i[pre], weights=p[pre], minlength=n)
    f["sf_prepaid_pmax"] = pd.Series(p[pre]).groupby(i[pre]).max().reindex(range(n)).fillna(0).values
    # payment INTO a (i in a flow pair): passed on afterwards
    f["sf_fwd_p"] = np.bincount(i[flow], weights=p[flow], minlength=n)
    fwd_usd = np.bincount(i[flow], weights=p[flow] * v[j[flow]], minlength=n)
    f["sf_fwd_ratio"] = np.log1p(fwd_usd) - np.log1p(v)
    # payment INTO a (j in a pre pair): a had already paid out before this arrived
    f["sf_late_p"] = np.bincount(j[pre], weights=p[pre], minlength=n)
    # contrast: funded minus prepaid, for outflows; forwarded minus late, for inflows
    f["sf_out_net"] = f.sf_fund_p - f.sf_prepaid_p
    f["sf_in_net"] = f.sf_fwd_p - f.sf_late_p
    return f.astype("float32")
