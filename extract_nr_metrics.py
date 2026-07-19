#!/usr/bin/env python3
"""
extract_nr_metrics.py — extract emergent NR-V2X metrics ([nr-plr] counters and
[nr-drop] events) from van3twin logs and merge them as columns into
summary_all_experiments.csv, so aggregate_seeds.py picks them up for free.

Log line formats handled (tolerant):
  [nr-plr] t=42.1005 tx=285379 rx_ok=2535179 rx_ko=16688869 [pscch_ok=.. pscch_ko=..]
  [nr-drop] ... total=N          (falls back to counting lines if no total=)

Per experiment, the LAST [nr-plr] line gives the run totals:
  nr_sim_t_reached, nr_tx, nr_rx_ok, nr_rx_ko,
  nr_plr_pct = 100*rx_ko/(rx_ok+rx_ko), nr_drops.
nr_sim_t_reached also doubles as a run-completeness indicator (vs simTime).

Usage:
  python3 extract_nr_metrics.py --log-base LOGDIR [--summary summary.csv]
     [--out nr_metrics.csv]
  --summary: merge columns into the given summary (matched on 'experiment');
             writes <summary> in place with a .bak backup.
"""
from __future__ import annotations
import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PLR_RE = re.compile(
    r"\[nr-plr\]\s+t=(?P<t>[\d.]+)\s+tx=(?P<tx>\d+)\s+"
    r"rx_ok=(?P<rx_ok>\d+)\s+rx_ko=(?P<rx_ko>\d+)")
DROP_TOTAL_RE = re.compile(r"\[nr-drop\].*?total[=:]\s*(\d+)")
DROP_LINE_RE = re.compile(r"\[nr-drop\]")


def scan_run(run_dir: Path) -> dict | None:
    """Parse the van3twin log(s) of one run directory."""
    logs = sorted((run_dir / "van3twin").glob("stdout.log")) or \
        sorted((run_dir / "van3twin").glob("van3twin.log*"))
    if not logs:
        return None
    last_plr, drops_total, drop_lines = None, None, 0
    for lf in logs:
        try:
            with open(lf, errors="replace") as fh:
                for line in fh:
                    m = PLR_RE.search(line)
                    if m:
                        last_plr = m
                        continue
                    if DROP_LINE_RE.search(line):
                        drop_lines += 1
                        mt = DROP_TOTAL_RE.search(line)
                        if mt:
                            drops_total = int(mt.group(1))
        except OSError:
            continue
    row: dict = {"nr_drops": drops_total if drops_total is not None else drop_lines}
    if last_plr:
        tx = int(last_plr["tx"])
        ok = int(last_plr["rx_ok"])
        ko = int(last_plr["rx_ko"])
        row.update(
            nr_sim_t_reached=float(last_plr["t"]),
            nr_tx=tx, nr_rx_ok=ok, nr_rx_ko=ko,
            nr_plr_pct=round(100.0 * ko / (ok + ko), 3) if (ok + ko) else np.nan,
            nr_avg_rx_per_tx=round(ok / tx, 3) if tx else np.nan,
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-base", required=True, type=Path,
                    help="dir containing <experiment>/<run_ts>/van3twin/ trees")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None,
                    help="summary_all_experiments.csv to merge columns into")
    args = ap.parse_args()
    out = args.out or args.log_base / "nr_metrics.csv"

    rows = []
    for exp_dir in sorted(p for p in args.log_base.iterdir() if p.is_dir()):
        run_dirs = sorted((p for p in exp_dir.iterdir() if p.is_dir()),
                          key=lambda p: p.name)
        if not run_dirs:
            continue
        row = scan_run(run_dirs[-1])  # most recent run of this experiment
        if row is None:
            print(f"  [warn] no van3twin log for {exp_dir.name}")
            continue
        row["experiment"] = exp_dir.name
        rows.append(row)

    if not rows:
        print("ERROR: no [nr-plr]/[nr-drop] data found", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows).set_index("experiment").sort_index()
    df.to_csv(out)
    print(f"Wrote {out} ({len(df)} experiments)")
    incomplete = df[df.get("nr_sim_t_reached", pd.Series(dtype=float)) < 290]
    if len(incomplete):
        print(f"NOTE: {len(incomplete)} run(s) with sim_t < 290s "
              f"(incomplete — report explicitly):")
        for e, t in incomplete["nr_sim_t_reached"].items():
            print(f"    {e}: t={t}")

    if args.summary:
        shutil.copy2(args.summary, args.summary.with_suffix(".csv.bak"))
        s = pd.read_csv(args.summary)
        merged = s.merge(df.reset_index(), on="experiment", how="left")
        merged.to_csv(args.summary, index=False)
        print(f"Merged {len(df.columns)} columns into {args.summary} "
              f"(backup: {args.summary.with_suffix('.csv.bak').name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
