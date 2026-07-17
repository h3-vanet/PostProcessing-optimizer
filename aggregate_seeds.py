#!/usr/bin/env python3
"""
aggregate_seeds.py — collapse per-seed rows of summary_all_experiments.csv
into one row per base scenario with mean / std / 95% CI for every numeric
metric.

Input rows are one per (scenario, seed), as produced by post_process.py with
the seed-aware _parse_experiment_name. Output: summary_seed_aggregated.csv
with columns <metric>_mean, <metric>_std, <metric>_ci95, plus n_seeds so that
partial seed coverage (SLURM failures) is explicit, never hidden.

CI uses the t-distribution (small n): ci95 = t(0.975, n-1) * std / sqrt(n).

Usage:
    python3 aggregate_seeds.py --summary summary_all_experiments.csv \
        [--out summary_seed_aggregated.csv] [--expected-seeds 5]
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

META_COLS = {"experiment", "scenario_base", "seed", "traffic", "occupancy"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--expected-seeds", type=int, default=5)
    args = ap.parse_args()
    out = args.out or args.summary.with_name("summary_seed_aggregated.csv")

    df = pd.read_csv(args.summary)

    # scenario_base/seed come from the patched post_process; if the CSV was
    # produced by an older version, derive them from the experiment name.
    if "scenario_base" not in df.columns or df["scenario_base"].isna().all():
        df["seed"] = df["experiment"].str.extract(r"_seed(\d+)$")[0].astype("Int64")
        df["scenario_base"] = df["experiment"].str.replace(
            r"_seed\d+$", "", regex=True)

    metric_cols = [c for c in df.columns
                   if c not in META_COLS
                   and pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        print("ERROR: no numeric metric columns found", file=sys.stderr)
        return 1

    rows = []
    for base, g in df.groupby("scenario_base", sort=True):
        n = len(g)
        row: dict = {
            "scenario_base": base,
            "traffic": g["traffic"].iloc[0] if "traffic" in g else "",
            "occupancy": g["occupancy"].iloc[0] if "occupancy" in g else np.nan,
            "n_seeds": n,
            "seeds": ",".join(str(s) for s in sorted(g["seed"].dropna().astype(int)))
                     if "seed" in g else "",
        }
        tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce").dropna()
            m = len(vals)
            mean = vals.mean() if m else np.nan
            std = vals.std(ddof=1) if m > 1 else np.nan
            ci = (stats.t.ppf(0.975, m - 1) * std / np.sqrt(m)) if m > 1 else np.nan
            row[f"{c}_mean"] = round(mean, 6) if m else np.nan
            row[f"{c}_std"] = round(std, 6) if m > 1 else np.nan
            row[f"{c}_ci95"] = round(ci, 6) if m > 1 else np.nan
        rows.append(row)
        flag = "" if n == args.expected_seeds else \
            f"   <-- PARTIAL ({n}/{args.expected_seeds} seeds)"
        print(f"  {base}: n={n}{flag}")

    agg = pd.DataFrame(rows)
    agg.to_csv(out, index=False)

    n_partial = int((agg["n_seeds"] != args.expected_seeds).sum())
    print(f"\nWrote {out}  ({len(agg)} scenarios, {len(metric_cols)} metrics)")
    if n_partial:
        print(f"WARNING: {n_partial} scenario(s) with partial seed coverage — "
              f"report n_seeds explicitly, do not average it away.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
