"""Post-only experiment: cross-fitted meta-blend over OOF scores (teammate's item 9 / untried item 1).

    python exp_blend.py --train ../data/train --cache ../cache --name v8 [--feats-pkl F_v8.pkl]

For each repeat, recover the outer folds (same StratifiedGroupKFold seeds as validate.py); for each outer fold
fit a small logistic model on the OTHER folds' OOF rows (inputs: raw a, raw b, smoothed a, smoothed b, and a
few hard-evidence features) and score the held-out fold. Compared against the hand-set blend on the same rows.
"""
import argparse, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import predict
from load import load_split
from validate import report

GATE = ["nbxr_wdeg", "nb8_wdeg", "sbs_cover", "sbs_req_fund_rel", "e80_funded_any", "e80_maxsize", "sf_fund_p", "sf_prepaid_p",
        "nbxr_sf_fund_p", "nbxr_fund_gap", "ab_s_cover", "ab_s_minbal_rel", "pair_first"]


def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/train"); ap.add_argument("--cache", default="../cache")
    ap.add_argument("--name", default="v8"); ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--xr-smooth", type=float, default=0.0)
    a = ap.parse_args()
    predict.XR_SMOOTH = a.xr_smooth
    d = load_split(a.train)
    X = pd.read_pickle(os.path.join(a.cache, "X.pkl")); P = pd.read_pickle(os.path.join(a.cache, "P.pkl"))
    S = P[P.xr == 0]
    F = pd.read_pickle(os.path.join(a.cache, f"F_{a.name}.pkl"))
    z = np.load(os.path.join(a.cache, f"oof_{a.name}.npz"))
    oofs, oof_shapes = z["oof"], z["oof_shape"]
    y = d["is_suspicious"].to_numpy(); groups = d["scenario_id"].fillna(d["hash"]).to_numpy()
    gate = [c for c in GATE if c in F.columns]
    print("gate features:", gate)
    for rep in range(len(oofs)):
        ra, rb = oofs[rep], oof_shapes[rep]
        pairs = P if predict.XR_SMOOTH > 0 else S
        sa, sb = predict.smooth(d, X, ra, pairs), predict.smooth(d, X, rb, pairs)
        base = (sa + predict.SHAPE_WEIGHT * sb) / (1 + predict.SHAPE_WEIGHT)
        report(d, base, f"rep{rep} hand blend")
        M_small = np.c_[logit(ra), logit(rb), logit(sa), logit(sb)]
        G = np.nan_to_num(F[gate].to_numpy(dtype=float))
        M_big = np.c_[M_small, G, M_small[:, 2:3] * G]       # scores, gates, and gate x smoothed-a interactions
        for tag, M in (("stack4", M_small), ("stack+gate", M_big)):
            out = np.zeros(len(d))
            for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(F, y, groups):
                sc = StandardScaler().fit(M[tr])
                m = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(M[tr]), y[tr])
                out[va] = m.decision_function(sc.transform(M[va]))
            report(d, out, f"rep{rep} {tag} C={a.C}")
        # rank-average of the hand blend and the stacker: a hedge
        from scipy.stats import rankdata
        report(d, rankdata(base) + rankdata(out), f"rep{rep} hand+stack rankavg")


if __name__ == "__main__":
    main()
