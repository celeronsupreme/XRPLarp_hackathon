"""
Version 12: blend the version 9 and version 11 submissions into one score.

    python blend.py --a predictions_v9.csv --b predictions_v11.csv --out predictions.csv

Method (default "rank"): each version's scores are turned into normalised ranks (average rank for
ties, divided by the number of rows, so 0 < r <= 1) and combined as a weighted mean,
    score = W_V9 * rank_v9 + (1 - W_V9) * rank_v11,   W_V9 = 0.5.
The metric (average precision) depends only on the ordering, and the two pipelines' scores sit on
different scales in the low tail (v11's smallest scores are roughly 10x smaller than v9's), so ranks
combine the two orderings without either version's scale dominating. Equal weights because the two
versions score the same on the leaderboard (73.4 and 73.3).

Alternatives (--method):
    "mean"  weighted mean of the scores. Because smoothing and the (a + 0.25 b) blend in predict.py are
            linear, this equals averaging the two pipelines' model scores before smoothing.
    "logit" weighted mean of log-odds, mapped back to 0-1 with a sigmoid.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata

METHOD = "rank"
W_V9 = 0.5
METHODS = ("rank", "mean", "logit")


def _logit(x):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


def blend_scores(sa, sb, method=METHOD, w=W_V9):
    """Combine two score vectors (same row order). w is the weight of the first one."""
    sa, sb = np.asarray(sa, dtype=float), np.asarray(sb, dtype=float)
    if method == "rank":
        n = len(sa)
        return w * rankdata(sa) / n + (1 - w) * rankdata(sb) / n
    if method == "mean":
        return w * sa + (1 - w) * sb
    if method == "logit":
        return 1.0 / (1.0 + np.exp(-(w * _logit(sa) + (1 - w) * _logit(sb))))
    raise ValueError(f"unknown blend method {method!r}; choose from {METHODS}")


def _read(path):
    df = pd.read_csv(path)
    missing = {"TRANSACTION_HASH", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df = df[["TRANSACTION_HASH", "score"]]
    if not df["TRANSACTION_HASH"].is_unique:
        raise ValueError(f"{path}: duplicate TRANSACTION_HASH values")
    if not np.isfinite(df["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: non-finite scores")
    return df


def blend_files(path_a, path_b, out, method=METHOD, w=W_V9):
    """Blend two prediction CSVs on TRANSACTION_HASH and write the result (row order of path_a)."""
    a, b = _read(path_a), _read(path_b)
    m = a.merge(b, on="TRANSACTION_HASH", how="inner", suffixes=("_a", "_b"), validate="one_to_one")
    if not (len(m) == len(a) == len(b)):
        raise ValueError(f"hash sets differ: {len(a)} rows in {path_a}, {len(b)} in {path_b}, {len(m)} in common")
    score = blend_scores(m["score_a"].to_numpy(), m["score_b"].to_numpy(), method, w)
    res = pd.DataFrame({"TRANSACTION_HASH": m["TRANSACTION_HASH"].to_numpy(), "score": np.clip(score, 0, 1)})
    res.to_csv(out, index=False)
    print(f"blended {len(res):,} rows ({method}, w_v9={w}) -> {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="predictions_v9.csv", help="version 9 predictions")
    ap.add_argument("--b", default="predictions_v11.csv", help="version 11 predictions")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--method", default=METHOD, choices=METHODS)
    ap.add_argument("--w", type=float, default=W_V9, help="weight of --a (version 9)")
    a = ap.parse_args()
    blend_files(a.a, a.b, a.out, a.method, a.w)


if __name__ == "__main__":
    main()
