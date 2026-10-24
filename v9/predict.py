"""
Train on the labelled training split and score the test split.

    python predict.py --train train_pkg/train --test test_pkg --out predictions.csv

Pipeline: tidy payments (load.py) -> features per split (features.py) -> two models, each an
average of LightGBM and XGBoost over 5 seeds: (a) suspicious vs not, (b) "part of any scheme-shaped group" (suspicious OR
innocent look-alike), which teaches the shape of slow schemes from twice the examples
-> smooth each model's score with the mean score of its linked burst (novel-pair payments
sharing an account within 10 / 20 minutes) -> final = (a) + 0.25 x (b), rescaled to 0-1.

Version 4 adds rebuilt groups: a link model (links.py), trained on which training payments belong
to the same planted scenario, predicts links between test payments that share an account within
8 hours; confident links (p > 0.7 / 0.8 / 0.9) are clustered into groups, and each payment gets its
group's properties (groupfeat.py): size, span, spacing, amounts, currencies, middle accounts, whether
they ever pay out more than they have received, and how much passes through. Scores are also
smoothed over the 0.9-confidence groups.

Version 5 adds cross-round links and per-account groups: relay schemes repeat the same sender->receiver
payment in rounds 1-6 days apart, so the link model also scores each payment against the next 1-2
payments of the same pair. For every account, its own payments are clustered on confident links that
run through it (egofeat.py), which rebuilds multi-round relay groups without chaining across accounts;
each payment gets its sender's and receiver's group: size, rounds, whether the account ever pays out
more than it has received so far (funded), and pass-through.

Version 8 adds link-probability-weighted neighbour features (neighfeat.py), including across relay rounds,
and gap-regularity features per group.

Version 7 = version 6_1 (soft funding features, softflow.py) + the version 6 changes below.

Version 6 keeps the models and adds: unique senders/receivers per motif component, a night-time flag,
and link-probability-weighted smoothing (SMOOTH_MODE "pweighted", SMOOTH_WEIGHT 0.85): each model's score
is blended with the mean of its burst neighbours and the p-weighted mean of its linked payments.
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb
import pandas as pd

import features
from egofeat import ego_features
from groupfeat import group_features
from softflow import soft_flow_features
from neighfeat import neighbour_features, spread_features, typed_neighbour_features
from softbal import soft_balance_features, account_balance_features
from egofeat import _ego_labels
from links import train_link_probs
from load import load_split

PARAMS = dict(n_estimators=300, learning_rate=0.06, num_leaves=7, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", n_jobs=-1, verbosity=0)
SEEDS = (0, 1, 2, 3, 4)
SMOOTH_GAPS_MIN = (10, 20)
SMOOTH_WEIGHT = 0.85     # v6: 0.7 in v4/v5
SHAPE_WEIGHT = 0.25
BLEND = "linear"          # or "rankaware"
SMOOTH_MODE = "pweighted"  # v6 default; "flat" reproduces v4/v5 smoothing
PWEIGHT_MIN = 0.5        # minimum link probability used by pweighted smoothing
XR_SMOOTH = 0.0          # v9 option: weight of cross-round links in pweighted smoothing (0 = v8 behaviour)
FEATS = {"v8", "softbal", "acctbal", "typed"}  # feature families: v8 | softbal | acctbal | typed
RELAY = ("aggregate_relay", "distributed_relay", "ambiguous")


def shape_target(df: pd.DataFrame) -> np.ndarray:
    """1 = payment belongs to a planted scheme-shaped scenario, suspicious or innocent look-alike."""
    return (df["scenario_id"].notna() & ~df["family"].isin(RELAY)).astype(int).to_numpy()


def group_labels(n: int, pairs: pd.DataFrame, th: float) -> np.ndarray:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    sel = pairs[pairs.p > th]
    G = sp.coo_matrix((np.ones(len(sel)), (sel.i.values, sel.j.values)), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    sizes = np.bincount(lab)
    return np.where(sizes[lab] > 1, lab, -1)


def smooth(df: pd.DataFrame, X: pd.DataFrame, raw: np.ndarray, pairs: pd.DataFrame = None) -> np.ndarray:
    """Blend each payment's score with the mean score of its linked payments.
    SMOOTH_MODE 'flat': mean over first-time-pair bursts (10/20 min) and 0.9-confidence rebuilt groups (v4/v5).
    SMOOTH_MODE 'pweighted': the burst means plus a link-probability-weighted mean over the payment's links."""
    dd = df.reset_index(drop=True).copy()
    dd["pair_first"] = X["pair_first"].to_numpy().astype(int)
    nov = dd["pair_first"].to_numpy() == 1
    labs = [features._components(dd, nov, gap * 60) for gap in SMOOTH_GAPS_MIN]
    means = [np.where(lab < 0, raw, pd.Series(raw).groupby(lab).transform("mean").to_numpy()) for lab in labs]
    if pairs is not None:
        if SMOOTH_MODE == "pweighted":
            n = len(dd)
            sel = pairs[pairs.p > PWEIGHT_MIN]
            i, j, p = sel.i.to_numpy(), sel.j.to_numpy(), sel.p.to_numpy()
            if "xr" in sel.columns:
                p = np.where(sel.xr.to_numpy() == 1, p * XR_SMOOTH, p)
            num = np.bincount(i, weights=p * raw[j], minlength=n) + np.bincount(j, weights=p * raw[i], minlength=n)
            den = np.bincount(i, weights=p, minlength=n) + np.bincount(j, weights=p, minlength=n)
            # self counts with weight 1 so unlinked payments keep their own score
            means.append((num + raw) / (den + 1))
        else:
            lab = group_labels(len(dd), pairs, 0.9)
            means.append(np.where(lab < 0, raw, pd.Series(raw).groupby(lab).transform("mean").to_numpy()))
    return (1 - SMOOTH_WEIGHT) * raw + SMOOTH_WEIGHT * np.mean(means, axis=0)


def combine(df, X, raw, raw_shape, pairs):
    """Final score from the two models' raw scores. BLEND 'linear': a + w*b (v4/v5).
    BLEND 'rankaware': a + w*b*(1-a), so the shape model lifts uncertain payments but not confident ones."""
    sa, sb = smooth(df, X, raw, pairs), smooth(df, X, raw_shape, pairs)
    if BLEND == "rankaware":
        return (sa + SHAPE_WEIGHT * sb * (1 - sa)) / (1 + SHAPE_WEIGHT)
    return (sa + SHAPE_WEIGHT * sb) / (1 + SHAPE_WEIGHT)


def build_all(d, X, P, S):
    """Full feature table: v4 per-payment, v4 groups, v5 ego groups, v6.1 soft funding, v8 neighbour and spread."""
    F = pd.concat([X, group_features(d, S), ego_features(d, P), soft_flow_features(d, S)], axis=1)
    if "softbal" in FEATS:
        F = pd.concat([F, soft_balance_features(d, S)], axis=1)
    if "acctbal" in FEATS:
        F = pd.concat([F, account_balance_features(d)], axis=1)
    N = neighbour_features(d, F, P)
    if "typed" in FEATS:
        N = pd.concat([N, typed_neighbour_features(d, pd.concat([F, N], axis=1), P)], axis=1)
    dd = d.reset_index(drop=True).copy()
    dd["pair_first"] = X["pair_first"].to_numpy().astype(int)
    nov = dd["pair_first"].to_numpy() == 1
    n = len(dd)
    ego = _ego_labels(dd, P, 0.8)
    labs = {"cN20": features._components(dd, nov, 20 * 60), "cA90": features._components(dd, np.ones(n, bool), 90 * 60),
            "e80s": ego[:n].astype(float), "e80r": ego[n:].astype(float)}
    Sp = spread_features(d, labs)
    return pd.concat([F, N, Sp], axis=1)


def prepare(train_dir, test_dir):
    tr = load_split(train_dir)
    te = load_split(test_dir)
    X_tr, X_te = features.build(tr), features.build(te)
    P_tr, P_te = train_link_probs(tr, te)
    # v4 group features and smoothing use the within-8h links only, exactly as before;
    # v5 per-account groups (egofeat.py) also use the cross-round links
    S_tr, S_te = P_tr[P_tr.xr == 0], P_te[P_te.xr == 0]
    F_tr = build_all(tr, X_tr, P_tr, S_tr)
    F_te = build_all(te, X_te, P_te, S_te)[F_tr.columns]
    return tr, te, X_te, S_te, F_tr, F_te


def fit_predict(F_tr, y, shape, F_te, seeds):
    """Sum of predictions over the given seeds (LightGBM + XGBoost, both targets); divide by 2*len(SEEDS) later."""
    raw, raw_shape = np.zeros(len(F_te)), np.zeros(len(F_te))
    for seed in seeds:
        for target, acc in ((y, raw), (shape, raw_shape)):
            acc += lgb.LGBMClassifier(**PARAMS, random_state=seed).fit(F_tr, target).predict_proba(F_te)[:, 1]
            acc += xgb.XGBClassifier(**XGB_PARAMS, random_state=seed).fit(F_tr, target).predict_proba(F_te)[:, 1]
    return raw, raw_shape


def finish(te, X_te, P_te, raw_sum, raw_shape_sum):
    n = 2 * len(SEEDS)
    raw, raw_shape = raw_sum / n, raw_shape_sum / n
    score = combine(te, X_te, raw, raw_shape, P_te)
    return pd.DataFrame({"TRANSACTION_HASH": te["hash"].to_numpy(), "score": np.clip(score, 0, 1)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_pkg/train")
    ap.add_argument("--test", default="test_pkg")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    t0 = time.time()
    tr, te, X_te, P_te, F_tr, F_te = prepare(a.train, a.test)
    y, shape = tr["is_suspicious"].to_numpy(), shape_target(tr)
    print(f"train {F_tr.shape}, test {F_te.shape}, positives {y.sum()}  ({time.time() - t0:.0f}s)", flush=True)
    raw, raw_shape = fit_predict(F_tr, y, shape, F_te, SEEDS)
    out = finish(te, X_te, P_te, raw, raw_shape)
    assert out["TRANSACTION_HASH"].is_unique and len(out) == len(te)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out):,} rows -> {a.out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
