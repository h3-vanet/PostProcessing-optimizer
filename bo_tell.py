#!/usr/bin/env python3
"""
bo_tell.py — runs INSIDE the container.

Runs post-processing over the freshly produced logs, computes the weighted OBJ_C
objective, feeds it back to the persistent skopt Optimizer (closing the iteration
opened by bo_ask.py), and appends the result to the BO trace CSV.

Usage (host calls this via apptainer exec, after the array job finished):
    python3 bo_tell.py --state DIR --log-base LOGS --post-out POSTDIR \
        --results bo_results.csv --iter N --job-id JID
"""
from __future__ import annotations
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Objective weights (OBJ_C); must match the design discussion.
W_CLEAN_SUCCESS = 0.40
W_UNFAIR        = 0.35
W_LATENCY       = 0.25
LAT_NORM        = 2000.0
HIGH_OCC_THRESHOLD = 85
HIGH_OCC_WEIGHT    = 2.0

POST_PROCESS = Path(__file__).resolve().parent / "post_process.py"


def run_post(log_base: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(POST_PROCESS),
         "--log-base", str(log_base), "--out", str(out_dir)],
        check=True,
    )
    summ = out_dir / "summary_all_experiments.csv"
    if not summ.exists():
        summ = log_base.parent / "summary_all_experiments.csv"
    return pd.read_csv(summ)


def objective(df: pd.DataFrame) -> float:
    d = df.copy()
    # A bad candidate config can make some scenarios produce no assignments,
    # leaving metric columns missing or NaN. Those must score as *bad* (high
    # objective), not crash the loop. Fill missing columns with worst-case
    # values and NaNs per-row likewise.
    defaults = {
        "A5b_double_rate_pct": 100.0,  # worst: every slot double-assigned
        "A2_success_rate_pct": 0.0,    # worst: nothing assigned
        "A5_jain":             0.0,    # worst: maximally unfair
        "A3_p50_ms":           LAT_NORM,  # worst: saturate latency term
        "occupancy":           50.0,   # neutral weight if unknown
    }
    for col, fill in defaults.items():
        if col not in d.columns:
            d[col] = fill
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(fill)

    double_n = (d["A5b_double_rate_pct"] / 100.0).clip(0, 1)
    net_ok   = ((d["A2_success_rate_pct"] / 100.0) * (1.0 - double_n)).clip(0, 1)
    unfair_n = (1.0 - d["A5_jain"]).clip(0, 1)
    lat_n    = (d["A3_p50_ms"] / LAT_NORM).clip(0, 1)
    obj = (W_CLEAN_SUCCESS * (1.0 - net_ok)
           + W_UNFAIR * unfair_n
           + W_LATENCY * lat_n)
    w = d["occupancy"].apply(
        lambda o: HIGH_OCC_WEIGHT if o >= HIGH_OCC_THRESHOLD else 1.0)
    score = float((obj * w).sum() / w.sum())
    # Final guard: never hand NaN/inf back to skopt.
    if not (score == score) or score in (float("inf"), float("-inf")):
        return 1.0
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--log-base", required=True, type=Path,
                    help="parent dir; each batch point's logs are in "
                         "<log-base>/b{i}/")
    ap.add_argument("--post-out", required=True, type=Path,
                    help="parent dir for post-processing; <post-out>/b{i}/")
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--iter", required=True, type=int)
    ap.add_argument("--job-id", default="")
    args = ap.parse_args()

    pending = json.loads((args.state / "pending.json").read_text())
    batch = pending["batch"]

    with open(args.state / "optimizer.pkl", "rb") as f:
        opt = pickle.load(f)

    xs, scores, rows = [], [], []
    for i, point in enumerate(batch):
        # each batch point post-processed from its own log dir (its offset made
        # the scenarios write to a separate place; the host arranges b{i}/)
        lb = args.log_base / f"b{i}"
        po = args.post_out / f"b{i}"
        try:
            df = run_post(lb, po)
            score = objective(df)
        except Exception as e:
            print(f"[warn] b{i} post-process failed ({e}); score=1.0 (worst)")
            score = 1.0
        xs.append(point["x"])
        scores.append(score)
        r = dict(point["params"])
        r.update({"iter": args.iter, "batch_index": i,
                  "job_id": args.job_id, "objective": round(score, 6),
                  "beta": round(1 - point["params"]["alfa"], 4)})
        rows.append(r)
        print(f"[tell] b{i} OBJECTIVE={score:.6f}")

    # tell ALL points of the batch together, then persist once
    opt.tell(xs, scores)
    with open(args.state / "optimizer.pkl", "wb") as f:
        pickle.dump(opt, f)

    header = not args.results.exists()
    pd.DataFrame(rows).to_csv(args.results, mode="a", header=header, index=False)

    best_idx = int(min(range(len(opt.yi)), key=lambda i: opt.yi[i]))
    best = {"objective": float(opt.yi[best_idx]),
            "params": dict(zip([d.name for d in opt.space.dimensions],
                               opt.Xi[best_idx]))}
    (args.state / "best_params.json").write_text(json.dumps(best, indent=2))

    plots_dir = args.results.parent / "plots"
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "bo_plots.py"),
             "--results", str(args.results), "--out-dir", str(plots_dir)],
            check=True,
        )
    except Exception as e:
        print(f"[warn] bo_plots failed (non-fatal): {e}")

    # best score of this batch, for the host log
    print(f"OBJECTIVE={min(scores):.6f}")


if __name__ == "__main__":
    main()
