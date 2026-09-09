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

META_COLS = {"experiment", "scenario_base", "seed", "traffic", "occupancy", "arm",
             "skipped_experiments"}


def _traffic_occupancy_from_name(scenario_base: str) -> tuple[str, float]:
    """Best-effort traffic/occupancy parse for a scenario_base that has no
    surviving rows to read them from (every seed skipped/failed). Mirrors
    post_process.py's ``_parse_experiment_name`` (``parts[0]``/``parts[-1]``
    after stripping ``"combination_"``), duplicated here rather than
    imported so this script stays independently runnable, same as this
    file's existing scenario_base/seed fallback below.
    """
    core = scenario_base.replace("combination_", "")
    parts = core.split("_")
    traffic = parts[0] if parts else scenario_base
    try:
        occupancy = int(parts[-1])
    except (ValueError, IndexError):
        occupancy = float("nan")
    return traffic, occupancy


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
            "arm": g["arm"].iloc[0] if "arm" in g else "",
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

    # ── Fully-missing scenarios: every seed skipped/failed, so
    # scenario_base never appears in df at all — the groupby loop above has
    # no group to iterate for it, and would otherwise just omit it from the
    # output with no trace (a 16-row aggregate silently standing in for a
    # 20-scenario sweep looks the same as a complete one unless you already
    # know to count). Reads run_batch's persisted skipped_experiments.csv
    # (sibling of --summary) to recover these and emit an explicit
    # n_seeds=0 row instead. Absent (older summary, or a --summary produced
    # without post_process.py's skip-tracking) is not an error — this block
    # simply finds nothing to add.
    present_bases = set(df["scenario_base"].dropna())
    skipped_path = args.summary.parent / "skipped_experiments.csv"
    if skipped_path.is_file():
        skipped_df = pd.read_csv(skipped_path)
        skipped_df["scenario_base"] = skipped_df["experiment"].str.replace(
            r"_seed\d+$", "", regex=True)
        missing_bases = sorted(set(skipped_df["scenario_base"]) - present_bases)
        for base in missing_bases:
            traffic, occupancy = _traffic_occupancy_from_name(base)
            skipped_names = sorted(
                skipped_df.loc[skipped_df["scenario_base"] == base, "experiment"])
            row = {
                "scenario_base": base,
                "arm": df["arm"].dropna().iloc[0] if "arm" in df.columns
                       and not df["arm"].dropna().empty else "",
                "traffic": traffic,
                "occupancy": occupancy,
                "n_seeds": 0,
                "seeds": "",
                "skipped_experiments": ",".join(skipped_names),
            }
            for c in metric_cols:
                row[f"{c}_mean"] = np.nan
                row[f"{c}_std"] = np.nan
                row[f"{c}_ci95"] = np.nan
            rows.append(row)
            print(f"  {base}: n=0   <-- MISSING (all {len(skipped_names)} "
                  f"seed(s) skipped/failed)")

    agg = pd.DataFrame(rows)
    agg.to_csv(out, index=False)

    n_missing = int((agg["n_seeds"] == 0).sum())
    n_partial = int(((agg["n_seeds"] != args.expected_seeds) & (agg["n_seeds"] > 0)).sum())
    print(f"\nWrote {out}  ({len(agg)} scenarios, {len(metric_cols)} metrics)")
    if n_missing:
        print(f"WARNING: {n_missing} scenario(s) fully missing (every seed "
              f"skipped/failed) — see their skipped_experiments column; "
              f"do not read the row count alone as a complete sweep.")
    if n_partial:
        print(f"WARNING: {n_partial} scenario(s) with partial seed coverage — "
              f"report n_seeds explicitly, do not average it away.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
