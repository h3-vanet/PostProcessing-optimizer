#!/usr/bin/env python3
"""
recover_observations.py

Recovers ALL Bayesian-optimization observations from disk, including configs
from rounds that bo_tell.py discarded because their batch was incomplete.

Background: bo_tell.py only records a round in bo_results.csv when ALL configs
of the batch produced a summary. If even one config is missing, the whole round
is dropped — including the good configs. But every summary_all_experiments.csv
is still on disk. This script walks them all, recomputes the objective with the
EXACT same formula as bo_tell.py, pairs it with the parameters from each
config_bX.toml, and writes a complete bo_results_RECOVERED.csv.

Objective formula (mirrors bo_tell.py lines 24-78):
    net_ok   = (success_rate/100 * (1 - double_rate/100)).clip(0,1)
    unfair_n = (1 - jain).clip(0,1)
    lat_n    = (p50_ms / 2000).clip(0,1)
    obj_row  = 0.40*(1-net_ok) + 0.35*unfair_n + 0.25*lat_n
    rows with occupancy >= 85 are weighted x2 in the mean
    empty df -> score 1.0

Usage:
    python3 recover_observations.py \
        --bo-dir ~/vanet-parking/bo_runs_parallel \
        --out ~/vanet-parking/bo_results_RECOVERED.csv
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

# ── Objective weights (must match bo_tell.py exactly) ────────────────────────
W_CLEAN_SUCCESS = 0.40
W_UNFAIR = 0.35
W_LATENCY = 0.25
LAT_NORM = 2000.0
HIGH_OCC_THRESHOLD = 85
HIGH_OCC_WEIGHT = 2.0

# Worst-case fills for missing columns (mirrors bo_tell.py lines 56-58)
FILL = {
    "A2_success_rate_pct": 0.0,
    "A5b_double_rate_pct": 0.0,
    "A5_jain": 0.0,
    "A3_p50_ms": LAT_NORM,
    "occupancy": 50.0,
}


def objective(df: pd.DataFrame) -> float:
    """Recompute the weighted objective for one config's summary dataframe.
    Mirrors bo_tell.py:objective()."""
    if df is None or len(df) == 0:
        return 1.0

    d = df.copy()
    for col, fillval in FILL.items():
        if col not in d.columns:
            d[col] = fillval
        else:
            d[col] = d[col].fillna(fillval)

    double_n = (d["A5b_double_rate_pct"] / 100.0).clip(0, 1)
    net_ok = ((d["A2_success_rate_pct"] / 100.0) * (1.0 - double_n)).clip(0, 1)
    unfair_n = (1.0 - d["A5_jain"]).clip(0, 1)
    lat_n = (d["A3_p50_ms"] / LAT_NORM).clip(0, 1)

    obj = (W_CLEAN_SUCCESS * (1.0 - net_ok)
           + W_UNFAIR * unfair_n
           + W_LATENCY * lat_n)

    weights = d["occupancy"].apply(
        lambda o: HIGH_OCC_WEIGHT if o >= HIGH_OCC_THRESHOLD else 1.0)

    score = (obj * weights).sum() / weights.sum()
    if pd.isna(score):
        return 1.0
    return float(score)


def parse_config(toml_path: str) -> dict:
    """Extract the 6 optimized parameters from a config_bX.toml.
    Robust to section layout: matches keys anywhere in the file."""
    params = {
        "alfa": None, "gossip_interval_ms": None, "neighbor_k": None,
        "t_base_ms": None, "cluster_resolution": None, "spot_resolution": None,
    }
    if not os.path.isfile(toml_path):
        return params
    with open(toml_path) as f:
        text = f.read()
    for key in params:
        # match: key = value   (value may be int or float, ignore trailing comment)
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([0-9]+\.?[0-9]*)",
                      text, re.MULTILINE)
        if m:
            val = m.group(1)
            params[key] = float(val) if "." in val else int(val)
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bo-dir", required=True,
                    help="bo_runs_parallel directory")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    bo_dir = os.path.expanduser(args.bo_dir)
    out_path = os.path.expanduser(args.out)

    rows = []
    n_recovered = 0
    n_missing_config = 0
    n_empty = 0

    # Walk every round_*/post/b*/summary_all_experiments.csv
    pattern = os.path.join(bo_dir, "round_*", "post", "b*",
                           "summary_all_experiments.csv")
    summaries = sorted(glob.glob(pattern))

    if not summaries:
        print(f"No summaries found under {pattern}", file=sys.stderr)
        sys.exit(1)

    for summ_path in summaries:
        # Derive round number and batch index from the path
        m = re.search(r"round_(\d+)/post/b(\d+)/", summ_path)
        if not m:
            continue
        round_num = int(m.group(1))
        batch_idx = int(m.group(2))

        # The matching config is round_NNN/config_bX.toml
        round_dir = os.path.join(bo_dir, f"round_{round_num:03d}")
        config_path = os.path.join(round_dir, f"config_b{batch_idx}.toml")
        params = parse_config(config_path)
        if params["alfa"] is None:
            n_missing_config += 1
            print(f"  WARN: no config params for {config_path}", file=sys.stderr)

        # Read summary and compute objective
        try:
            df = pd.read_csv(summ_path)
        except Exception as e:
            print(f"  WARN: cannot read {summ_path}: {e}", file=sys.stderr)
            df = None
        if df is None or len(df) == 0:
            n_empty += 1
        score = objective(df)

        rows.append({
            "round": round_num,
            "batch_index": batch_idx,
            "alfa": params["alfa"],
            "gossip_interval_ms": params["gossip_interval_ms"],
            "neighbor_k": params["neighbor_k"],
            "t_base_ms": params["t_base_ms"],
            "cluster_resolution": params["cluster_resolution"],
            "spot_resolution": params["spot_resolution"],
            "objective": round(score, 6),
            "n_scenarios": (0 if df is None else len(df)),
            "summary_path": summ_path,
        })
        n_recovered += 1

    out = pd.DataFrame(rows).sort_values(["round", "batch_index"])
    out.to_csv(out_path, index=False)

    # Report
    print(f"Recovered {n_recovered} observations from disk -> {out_path}")
    print(f"  (config params missing: {n_missing_config}, empty summaries: {n_empty})")
    valid = out[out["objective"] < 1.0]
    print(f"  observations with real objective (<1.0): {len(valid)}")
    if len(valid):
        best = valid.loc[valid["objective"].idxmin()]
        print(f"  BEST objective: {best['objective']:.6f}  "
              f"(round {int(best['round'])}, b{int(best['batch_index'])}: "
              f"alfa={best['alfa']}, gossip={best['gossip_interval_ms']}, "
              f"k={best['neighbor_k']}, t_base={best['t_base_ms']}, "
              f"res={best['cluster_resolution']}/{best['spot_resolution']})")
        # distribution
        print(f"  objective range: {valid['objective'].min():.4f} "
              f"– {valid['objective'].max():.4f}, "
              f"mean {valid['objective'].mean():.4f}")


if __name__ == "__main__":
    main()
