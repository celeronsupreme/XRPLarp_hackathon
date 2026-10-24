"""
Version 12: blend of version 9 and version 11. Train on the labelled training split and score the test split.

    python predict.py --train train_pkg/train --test test_pkg --out predictions.csv

Runs v9/predict.py and v11/predict.py exactly as uploaded (each in its own Python process, so their
same-named modules such as neighfeat.py do not clash), writes their CSVs into --workdir, then blends
them with blend.py (default: equal-weight mean of normalised ranks; see blend.py).

    --parallel   run the two pipelines at the same time (needs about 8 GB RAM instead of 4)
    --reuse      skip a pipeline whose CSV already exists in --workdir
"""
import argparse
import os
import subprocess
import sys
import time

from blend import METHOD, METHODS, W_V9, blend_files

HERE = os.path.dirname(os.path.abspath(__file__))
VERSIONS = ("v9", "v11")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train_pkg/train")
    ap.add_argument("--test", default="test_pkg")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--workdir", default="blend_work")
    ap.add_argument("--method", default=METHOD, choices=METHODS)
    ap.add_argument("--w", type=float, default=W_V9, help="weight of version 9")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.workdir, exist_ok=True)
    outs = {v: os.path.join(a.workdir, f"predictions_{v}.csv") for v in VERSIONS}
    procs = []
    for v in VERSIONS:
        if a.reuse and os.path.exists(outs[v]):
            print(f"{v}: reusing {outs[v]}", flush=True)
            continue
        cmd = [sys.executable, os.path.join(HERE, v, "predict.py"), "--train", a.train, "--test", a.test, "--out", outs[v]]
        print(f"{v}: {' '.join(cmd)}", flush=True)
        if a.parallel:
            procs.append((v, subprocess.Popen(cmd)))
        else:
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise SystemExit(f"{v}/predict.py failed (exit code {r.returncode})")
            print(f"{v}: done ({time.time() - t0:.0f}s)", flush=True)
    for v, p in procs:
        if p.wait() != 0:
            raise SystemExit(f"{v}/predict.py failed (exit code {p.returncode})")
        print(f"{v}: done ({time.time() - t0:.0f}s)", flush=True)
    blend_files(outs["v9"], outs["v11"], a.out, a.method, a.w)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
