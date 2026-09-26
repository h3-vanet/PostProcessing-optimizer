#!/usr/bin/env python3
"""Broker RTT 0 ms vs 50 ms comparison, from the two broker variants.

The unsuffixed broker series (`post_simtime_br`) is the RTT 0 ms sensitivity
run; the `_rtt50` series is the main configuration. Reads both, computes the
broker park rate per density, writes a small LaTeX table and macros.

Usage:
    python3 rtt_compare_macros.py --metrics DIR --out DIR \
        [--broker-suffix _rtt50] [--rtt0-suffix '']
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_tables import (
    density_display,
    DENSITY_LABELS,
    DENSITY_ORDER,
    VALID_NR_SIM_T_REACHED,
    NumberRegistry,
    render_latex,
)


def load_series(metrics_dir: Path, series: str) -> pd.DataFrame | None:
    path = metrics_dir / series / "summary_all_experiments.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["density"] = df["traffic"].map(DENSITY_LABELS)
    df["valid"] = df["nr_sim_t_reached"] >= VALID_NR_SIM_T_REACHED
    return df


def _mean(df: pd.DataFrame | None, density: str):
    if df is None:
        return None
    values = df[(df["valid"]) & (df["density"] == density)]["C1_park_rate"]
    return float(values.mean()) if len(values) else None


def compare(
    metrics_dir: Path,
    base: str = "post_simtime_br",
    suffix: str = "_rtt50",
    rtt0_suffix: str = "",
):
    rtt0 = load_series(metrics_dir, base + rtt0_suffix)
    rtt50 = load_series(metrics_dir, base + suffix)
    rows = []
    for density in DENSITY_ORDER:
        m0 = _mean(rtt0, density)
        m50 = _mean(rtt50, density)
        rows.append((density, m0, m50))
    return rows


def build(rows, numbers: NumberRegistry):
    diffs = []
    table_rows = []
    for density, m0, m50 in rows:
        if m0 is None or m50 is None:
            continue
        m0r, m50r = round(m0, 2), round(m50, 2)
        diff = round(m50r - m0r, 2)
        diffs.append(abs(diff))
        cap = density.capitalize()
        numbers.add(f"rttZero{cap}", f"{m0r:.2f}")
        numbers.add(f"rttFifty{cap}", f"{m50r:.2f}")
        numbers.add(f"rttDiff{cap}", f"{diff:+.2f}")
        table_rows.append([density_display(density), f"{m0r:.2f}", f"{m50r:.2f}", f"{diff:+.2f}"])
    if diffs:
        numbers.add("rttMaxAbsDiff", f"{max(diffs):.2f}")
    header = ["Density", "Broker RTT 0 (\\%)", "Broker RTT 50 (\\%)", "Diff (pts)"]
    caption = (
        "Broker park rate (\\%) with a 0\\,ms and a 50\\,ms modelled uplink "
        "round-trip time, by density. Diff = RTT 50 $-$ RTT 0, in percentage "
        "points."
    )
    return render_latex(header, table_rows, caption, "tab:rtt_compare", fontsize="\\footnotesize")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--broker-suffix", default="_rtt50")
    parser.add_argument(
        "--rtt0-suffix",
        default="",
        help="suffix for the RTT-0 broker base directory; '' is the JSON-era "
        "name, '_pc' is the postcard rerun",
    )
    args = parser.parse_args(argv)

    rows = compare(
        args.metrics, suffix=args.broker_suffix, rtt0_suffix=args.rtt0_suffix
    )
    numbers = NumberRegistry()
    latex = build(rows, numbers)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    numbers.write(args.out / "rtt_numbers.tex")
    (args.out / "tables" / "14_rtt_compare.tex").write_text(latex)
    for density, m0, m50 in rows:
        if m0 is not None and m50 is not None:
            print(f"{density}: RTT0={m0:.1f} RTT50={m50:.1f} diff={m50 - m0:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
