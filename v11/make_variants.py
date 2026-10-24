"""Train once on train/, score test/, cache raw scores, and write one CSV per post-processing variant."""
import os, sys, time
import numpy as np, pandas as pd
import predict
from predict import prepare, fit_predict, SEEDS, shape_target, combine
train_dir, test_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True); t0 = time.time()
tr, te, X_te, S_te, F_tr, F_te = prepare(train_dir, test_dir)
y, shape = tr["is_suspicious"].to_numpy(), shape_target(tr)
print(f"features {F_tr.shape} ({time.time()-t0:.0f}s)", flush=True)
raw, raw_shape = fit_predict(F_tr, y, shape, F_te, SEEDS)
n = 2 * len(SEEDS); raw, raw_shape = raw / n, raw_shape / n
np.savez(os.path.join(out_dir, "raw_test.npz"), raw=raw, raw_shape=raw_shape, hash=te["hash"].to_numpy())
X_te.to_pickle(os.path.join(out_dir, "X_te.pkl")); S_te.to_pickle(os.path.join(out_dir, "S_te.pkl")); te.to_pickle(os.path.join(out_dir, "te.pkl"))
print(f"models trained ({time.time()-t0:.0f}s)", flush=True)
variants = {"v11_alt_flat": ("linear", "flat", 0.7), "v11": ("linear", "pweighted", 0.85), "v11_alt_pweighted_sw070": ("linear", "pweighted", 0.7)}
for name, (blend, mode, sw) in variants.items():
    predict.BLEND, predict.SMOOTH_MODE, predict.SMOOTH_WEIGHT = blend, mode, sw
    score = combine(te, X_te, raw, raw_shape, S_te)
    out = pd.DataFrame({"TRANSACTION_HASH": te["hash"].to_numpy(), "score": np.clip(score, 0, 1)})
    assert out.TRANSACTION_HASH.is_unique and len(out) == len(te)
    out.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"), index=False); print("wrote", name, flush=True)
print(f"done ({time.time()-t0:.0f}s)")
