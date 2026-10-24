"""
Local validation: repeated grouped cross-validation on train/, same metric as the site (100 x AP),
with a breakdown by tier and family, top-1% precision/recall, and the median rank of hard relay payments.

    python validate.py --train train_pkg/train [--repeats 2] [--cache cache_dir]

With --cache, features and out-of-fold raw model scores are saved, so post-processing experiments
(blend / smoothing) can be re-evaluated without retraining:

    python validate.py --train train_pkg/train --cache cache --post-only --blend rankaware --smooth-weight 0.5
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

import features
from egofeat import ego_features
from groupfeat import group_features
from softflow import soft_flow_features
from links import train_link_probs
from load import load_split
from predict import shape_target, smooth, combine, build_all, fit_predict


def report(d, final, label=""):
    y = d["is_suspicious"].to_numpy()
    n = len(d)
    order = np.argsort(-final, kind="mergesort")
    rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
    cum_pos = np.cumsum(y[order])
    prec_at = cum_pos / np.arange(1, n + 1)          # precision at each rank
    pos_prec = np.zeros(n); pos_prec[order] = prec_at
    total = y.sum()
    ap = 100 * average_precision_score(y, final)
    out = {"AP": ap}
    for tier in ("easy", "medium", "hard"):
        m = (d["difficulty"] == tier).to_numpy()
        out[f"pts_{tier}"] = 100 * pos_prec[m].sum() / total
    out["pts_ambig"] = 100 * pos_prec[(d["family"] == "ambiguous").to_numpy()].sum() / total
    k = int(round(0.01 * n))
    top = rank <= k
    out["top1%_prec"] = y[top].mean(); out["top1%_rec"] = y[top].sum() / total
    rel = d["family"].isin(["aggregate_relay", "distributed_relay"]).to_numpy() & (y == 1)
    out["relay_median_rank"] = np.median(rank[rel])
    out["fp_top800"] = int((~y.astype(bool))[rank <= 800].sum())
    print(f"{label:28s} AP {ap:6.2f} | easy {out['pts_easy']:5.2f} med {out['pts_medium']:5.2f} hard {out['pts_hard']:5.2f} "
          f"| top1% prec {out['top1%_prec']:.3f} rec {out['top1%_rec']:.3f} | FP@800 {out['fp_top800']:3d} | relay med rank {out['relay_median_rank']:.0f}")
    fam = pd.DataFrame({"fam": d["family"], "y": y, "pp": pos_prec, "rank": rank})
    f = fam[fam.y == 1].groupby("fam").agg(n=("y", "size"), pts=("pp", lambda v: 100 * v.sum() / total), med_rank=("rank", "median"))
    out["by_family"] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_pkg/train")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--post-only", action="store_true", help="reuse cached OOF scores; only redo blend/smoothing")
    ap.add_argument("--blend", default=None, choices=["linear", "rankaware"])
    ap.add_argument("--shape-weight", type=float, default=None)
    ap.add_argument("--smooth-weight", type=float, default=None)
    ap.add_argument("--smooth-mode", default=None, choices=["flat", "pweighted"])
    ap.add_argument("--family-table", action="store_true")
    ap.add_argument("--libs", default="lgb,xgb")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    a.libs = tuple(a.libs.split(",")); a.seeds = tuple(range(a.seeds))
    t0 = time.time()
    import predict
    if a.blend: predict.BLEND = a.blend
    if a.shape_weight is not None: predict.SHAPE_WEIGHT = a.shape_weight
    if a.smooth_weight is not None: predict.SMOOTH_WEIGHT = a.smooth_weight
    if a.smooth_mode: predict.SMOOTH_MODE = a.smooth_mode

    d = load_split(a.train)
    if a.cache and a.post_only and os.path.exists(os.path.join(a.cache, "oof.npz")):
        z = np.load(os.path.join(a.cache, "oof.npz"), allow_pickle=True)
        X = pd.read_pickle(os.path.join(a.cache, "X.pkl")); P_tr = pd.read_pickle(os.path.join(a.cache, "P.pkl"))
        oofs, oof_shapes = z["oof"], z["oof_shape"]
    else:
        if a.cache and os.path.exists(os.path.join(a.cache, "F.pkl")):
            X = pd.read_pickle(os.path.join(a.cache, "X.pkl")); P_tr = pd.read_pickle(os.path.join(a.cache, "P.pkl")); F = pd.read_pickle(os.path.join(a.cache, "F.pkl"))
        else:
            X = features.build(d)
            P_tr, _ = train_link_probs(d, d.head(2000))
            F = build_all(d, X, P_tr, P_tr[P_tr.xr == 0])
        y = d["is_suspicious"].to_numpy()
        groups = d["scenario_id"].fillna(d["hash"]).to_numpy()
        shape = shape_target(d)
        print(f"features {F.shape} ({time.time() - t0:.0f}s)", flush=True)
        oofs, oof_shapes = [], []
        for rep in range(a.repeats):
            oof, oof_shape = np.zeros(len(d)), np.zeros(len(d))
            for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(F, y, groups):
                oof[va], oof_shape[va] = fit_predict(F.iloc[tr], y[tr], shape[tr], F.iloc[va], seeds=a.seeds, libs=a.libs)
            oofs.append(oof); oof_shapes.append(oof_shape)
            print(f"repeat {rep} trained ({time.time() - t0:.0f}s)", flush=True)
        oofs, oof_shapes = np.array(oofs), np.array(oof_shapes)
        if a.cache:
            os.makedirs(a.cache, exist_ok=True)
            np.savez(os.path.join(a.cache, f"oof_{'-'.join(a.libs)}_s{len(a.seeds)}.npz"), oof=oofs, oof_shape=oof_shapes)
            np.savez(os.path.join(a.cache, "oof.npz"), oof=oofs, oof_shape=oof_shapes)
            X.to_pickle(os.path.join(a.cache, "X.pkl")); P_tr.to_pickle(os.path.join(a.cache, "P.pkl")); F.to_pickle(os.path.join(a.cache, "F.pkl"))
    S_tr = P_tr[P_tr.xr == 0]
    results = []
    for rep in range(len(oofs)):
        final = combine(d, X, oofs[rep], oof_shapes[rep], S_tr)
        results.append(report(d, final, f"repeat {rep}"))
    keys = ["AP", "pts_easy", "pts_medium", "pts_hard", "top1%_prec", "fp_top800", "relay_median_rank"]
    m = {k: np.mean([r[k] for r in results]) for k in keys}; s = {k: np.std([r[k] for r in results]) for k in keys}
    print(f"MEAN   AP {m['AP']:6.2f} +/- {s['AP']:.2f} | easy {m['pts_easy']:5.2f} med {m['pts_medium']:5.2f} hard {m['pts_hard']:5.2f} "
          f"| top1% prec {m['top1%_prec']:.3f} | FP@800 {m['fp_top800']:.0f} | relay med rank {m['relay_median_rank']:.0f}  ({time.time() - t0:.0f}s)")
    if a.family_table:
        fam = pd.concat([r["by_family"] for r in results]).groupby(level=0).mean().round(2)
        print(fam.to_string())


if __name__ == "__main__":
    main()
