#!/usr/bin/env python3
"""Assemble the sensitivity dataset from per-configuration post-processing.

It reads, for every configuration in ``design.json``, the **per-run**
``<post-base>/<log_subdir>/summary_all_experiments.csv`` produced by the
``post`` step in the ppbo container (post_process.py + extract_nr_metrics.py),
applies the paper's validity filter (``nr_sim_t_reached >= 179``), and writes:

* ``dataset_per_run.csv``  — tidy, one row per (config, occupancy, seed);
* ``dataset_aggregated.csv`` — one row per config with mean/std/n per response
  plus validity counts and failure rate (invalid runs are recorded, never
  silently dropped);
* ``manifest.json`` — config -> files used, invalid run reasons.

This module deliberately does **not** run any post-processing (that is the ppbo
container's responsibility) and does **not** consume ``summary_seed_aggregated
.csv``: collapsing the 5 seeds would destroy the occupancy x seed pairing that
the common-random-number design relies on. Every config is checked to carry the
full 5x5 = 25-cell grid before it is accepted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402

DEFAULT_RESPONSES = [
    "C1_park_rate",
    "A5_jain",
    "A5b_overlap_rate_pct_d010",
    "A1_p50_ms",
    "D1_cp_tx_bytes_per_vehicle_mean",
    "nr_plr_pct",
]
PARAM_COLS = ["alfa", "t_base_ms", "max_distance_m", "trajectory_window",
              "gossip_interval_ms"]


def _read_design(path: Path) -> dict:
    return json.loads(path.read_text())


def _assert_cell_coverage(df: pd.DataFrame, cid: str, density: str) -> None:
    """Fail loudly unless this config carries the full per-run cell grid.

    The common-random-number/paired design needs one row per (occupancy, seed);
    ``summary_seed_aggregated.csv`` has one row per occupancy (5 rows) and would
    otherwise be silently accepted, producing wrong numbers. This is the same
    coverage property the preflight checks on the scenarios, applied at the
    other end of the pipeline.
    """
    missing = {"occupancy", "seed"} - set(df.columns)
    if missing:
        raise ValueError(f"{cid}: summary lacks columns {sorted(missing)}")
    cells = {(int(o), int(s)) for o, s in zip(df["occupancy"], df["seed"])}
    expected = {(o, s) for o in spec.OCCUPANCIES for s in spec.SEEDS}
    if len(df) != spec.RUNS_PER_CONFIG or cells != expected:
        raise ValueError(
            f"{cid}: expected exactly {spec.RUNS_PER_CONFIG} per-run rows "
            f"covering the 5x5 occupancy x seed grid, got {len(df)} rows / "
            f"{len(cells)} unique cells. Refusing to continue — this usually "
            f"means the aggregated summary was passed by mistake.")
    if "traffic" in df.columns:
        densities = set(df["traffic"].astype(str))
        if densities != {density}:
            raise ValueError(
                f"{cid}: expected traffic={density!r} only, got {sorted(densities)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True, type=Path)
    ap.add_argument("--post-base", required=True, type=Path,
                    help="dir holding <log_subdir>/summary_all_experiments.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="dataset output dir (default: design dir)")
    ap.add_argument("--arm", default="leaderless",
                    choices=("leaderless", "centralized"),
                    help="label recorded in the manifest; the actual post-processing "
                         "runs in the ppbo 'post' step")
    ap.add_argument("--valid-threshold", type=float, default=179.0)
    ap.add_argument("--responses", default=",".join(DEFAULT_RESPONSES))
    args = ap.parse_args()

    design = _read_design(args.design)
    expected_density = design.get("density", spec.DENSITY)
    out = (args.out or args.design.parent).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    responses = [r for r in args.responses.split(",") if r]

    per_run_frames: list[pd.DataFrame] = []
    manifest: dict = {"design": str(args.design), "arm": args.arm,
                      "valid_threshold": args.valid_threshold,
                      "responses": responses, "configs": {}}

    for config in design["configs"]:
        cid = config["config_id"]
        summary = args.post_base.expanduser() / config["log_subdir"] / \
            "summary_all_experiments.csv"
        entry: dict = {"summary": str(summary), "is_baseline": config["is_baseline"]}
        if not summary.is_file():
            entry.update({"n_runs": 0, "n_valid": 0,
                          "missing": True})
            manifest["configs"][cid] = entry
            print(f"  [warn] {cid}: no summary at {summary}")
            continue

        df = pd.read_csv(summary)
        try:
            _assert_cell_coverage(df, cid, expected_density)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            manifest["configs"][cid] = {**entry, "error": str(exc)}
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            return 1
        df["config_id"] = cid
        df["is_baseline"] = config["is_baseline"]
        for k in PARAM_COLS:
            df[k] = config["params"][k]
        df["beta"] = config["beta"]

        if "nr_sim_t_reached" not in df.columns:
            print(f"  [warn] {cid}: no nr_sim_t_reached column; "
                  f"treating all rows as invalid")
            df["nr_sim_t_reached"] = np.nan
        df["valid"] = pd.to_numeric(
            df["nr_sim_t_reached"], errors="coerce") >= args.valid_threshold

        per_run_frames.append(df)
        n_valid = int(df["valid"].sum())
        invalid_cells = df.loc[~df["valid"], ["traffic", "occupancy", "seed"]] \
            if {"traffic", "occupancy", "seed"} <= set(df.columns) else None
        entry.update({
            "n_runs": int(len(df)),
            "n_valid": n_valid,
            "n_invalid": int(len(df) - n_valid),
            "failure_rate": round((len(df) - n_valid) / len(df), 4) if len(df) else None,
            "mean_nr_sim_t_reached": round(float(
                pd.to_numeric(df["nr_sim_t_reached"], errors="coerce").mean()), 3),
            "invalid_cells": (invalid_cells.astype(str).to_dict("records")
                              if invalid_cells is not None and len(invalid_cells)
                              else []),
        })
        manifest["configs"][cid] = entry
        print(f"  {cid}: {n_valid}/{len(df)} valid")

    if not per_run_frames:
        print("ERROR: no summaries found; nothing to assemble", file=sys.stderr)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return 1

    per_run = pd.concat(per_run_frames, ignore_index=True)
    per_run_path = out / "dataset_per_run.csv"
    per_run.to_csv(per_run_path, index=False)

    # ── Aggregate over valid runs, one row per config ──────────────────────
    agg_rows = []
    for cid, g in per_run.groupby("config_id", sort=True):
        valid = g[g["valid"]]
        row = {"config_id": cid, "is_baseline": bool(g["is_baseline"].iloc[0]),
               "n_runs": int(len(g)), "n_valid": int(len(valid)),
               "failure_rate": round((len(g) - len(valid)) / len(g), 4)}
        for k in PARAM_COLS:
            row[k] = float(g[k].iloc[0])
        row["beta"] = float(g["beta"].iloc[0])
        for r in responses:
            vals = pd.to_numeric(valid.get(r), errors="coerce").dropna() \
                if r in valid.columns else pd.Series(dtype=float)
            row[f"{r}_mean"] = round(float(vals.mean()), 6) if len(vals) else np.nan
            row[f"{r}_std"] = round(float(vals.std(ddof=1)), 6) if len(vals) > 1 else np.nan
            row[f"{r}_n"] = int(len(vals))
        # claims per winner = ratio of means (paper's definition), config-level.
        if {"A2_assigned_uniq", "A2_vehicles_with_slot"} <= set(valid.columns):
            cw = pd.to_numeric(valid["A2_assigned_uniq"], errors="coerce").mean()
            ww = pd.to_numeric(valid["A2_vehicles_with_slot"], errors="coerce").mean()
            row["claims_per_winner"] = round(float(cw) / float(ww), 4) if ww else np.nan
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows).sort_values("config_id")
    agg_path = out / "dataset_aggregated.csv"
    agg.to_csv(agg_path, index=False)

    manifest["outputs"] = {"per_run": str(per_run_path),
                           "aggregated": str(agg_path)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {per_run_path} ({len(per_run)} rows)")
    print(f"Wrote {agg_path} ({len(agg)} configs)")
    print(f"Wrote {out / 'manifest.json'}")
    n_bad = sum(1 for c in manifest["configs"].values() if c.get("n_valid", 0) < 25)
    if n_bad:
        print(f"WARNING: {n_bad} config(s) with <25 valid runs — see manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
