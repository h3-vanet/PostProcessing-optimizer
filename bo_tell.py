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
    ap.add_argument("--log-base", required=True, type=Path)
    ap.add_argument("--post-out", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--iter", required=True, type=int)
    ap.add_argument("--job-id", default="")
    args = ap.parse_args()

    df = run_post(args.log_base, args.post_out)
    score = objective(df)

    pending = json.loads((args.state / "pending.json").read_text())
    with open(args.state / "optimizer.pkl", "rb") as f:
        opt = pickle.load(f)
    opt.tell(pending["x"], score)
    with open(args.state / "optimizer.pkl", "wb") as f:
        pickle.dump(opt, f)

    row = dict(pending["params"])
    row.update({"iter": args.iter, "job_id": args.job_id,
                "objective": round(score, 6),
                "beta": round(1 - pending["params"]["alfa"], 4)})
    header = not args.results.exists()
    pd.DataFrame([row]).to_csv(args.results, mode="a", header=header, index=False)

    # update best
    best_idx = int(min(range(len(opt.yi)), key=lambda i: opt.yi[i]))
    best = {"objective": float(opt.yi[best_idx]),
            "params": dict(zip([d.name for d in opt.space.dimensions],
                               opt.Xi[best_idx]))}
    (args.state / "best_params.json").write_text(json.dumps(best, indent=2))

    # refresh BO-specific plots (convergence + parameter importance)
    plots_dir = args.results.parent / "plots"
    try:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "bo_plots.py"),
             "--results", str(args.results), "--out-dir", str(plots_dir)],
            check=True,
        )
    except Exception as e:
        print(f"[warn] bo_plots failed (non-fatal): {e}")

    print(f"OBJECTIVE={score:.6f}")  # host parses this line


if __name__ == "__main__":
    main()
