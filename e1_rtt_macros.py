#!/usr/bin/env python3
"""E1 (RSU coverage) at RTT 0 ms vs RTT 50 ms.

Reads the unsuffixed broker sensitivity series (RTT 0) and their `_rtt50`
counterparts (RTT 50), at nominal density for the RSU/MCS variants and per
density for the 4-RSU baseline, writes a small table and macros.

Usage:
    python3 e1_rtt_macros.py --metrics DIR --out DIR [--broker-suffix _rtt50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_tables import (
    DENSITY_LABELS,
    DENSITY_ORDER,
    VALID_NR_SIM_T_REACHED,
    NumberRegistry,
    render_latex,
)

# (macro key, unsuffixed series, display label, density or None=per-density)
CONFIGS = [
    ("Four", "post_simtime_br", "Broker, 4 RSUs, MCS 14", None),
    ("Nine", "rsu_simtime_9", "Broker, 9 RSUs, MCS 14", "nominal"),
    ("Sixteen", "rsu_simtime_16", "Broker, 16 RSUs, MCS 14", "nominal"),
    ("TwentyFive", "rsu_simtime_25", "Broker, 25 RSUs, MCS 14", "nominal"),
    ("McsFive", "mcs5_simtime_br", "Broker, 4 RSUs, MCS 5", "nominal"),
]


def _load(metrics_dir: Path, series: str) -> pd.DataFrame | None:
    path = metrics_dir / series / "summary_all_experiments.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["density"] = df["traffic"].map(DENSITY_LABELS)
    df["valid"] = df["nr_sim_t_reached"] >= VALID_NR_SIM_T_REACHED
    return df


def _e1(df: pd.DataFrame | None, density: str):
    if df is None or "E1_rsu_coverage_pct" not in df.columns:
        return None
    sub = df[(df["valid"]) & (df["density"] == density)]
    return float(sub["E1_rsu_coverage_pct"].mean()) if len(sub) else None


def build(metrics_dir: Path, suffix: str = "_rtt50"):
    rows = []
    for key, series, label, density in CONFIGS:
        rtt0 = _load(metrics_dir, series)
        rtt50 = _load(metrics_dir, series + suffix)
        if density is None:
            for d in DENSITY_ORDER:
                m0, m50 = _e1(rtt0, d), _e1(rtt50, d)
                if m0 is None or m50 is None:
                    continue
                rows.append((f"{label} ({d})", m0, m50,
                             f"coverageRttZero{d.capitalize()}", f"coverageRttFifty{d.capitalize()}"))
        else:
            m0, m50 = _e1(rtt0, density), _e1(rtt50, density)
            if m0 is None or m50 is None:
                continue
            rows.append((label, m0, m50, f"coverageRttZero{key}", f"coverageRttFifty{key}"))
    return rows


def to_latex(rows, numbers: NumberRegistry) -> str:
    table_rows = []
    for label, m0, m50, k0, k50 in rows:
        numbers.add(k0, f"{m0:.1f}")
        numbers.add(k50, f"{m50:.1f}")
        table_rows.append([label, f"{m0:.1f}", f"{m50:.1f}"])
    header = ["Configuration", "E1 RTT 0 (\\%)", "E1 RTT 50 (\\%)"]
    caption = (
        "Broker RSU coverage (E1) with a 0\\,ms and a 50\\,ms modelled uplink "
        "round-trip time, at the stated density. Coverage is lower at RTT "
        "50\\,ms while park rate is unaffected; the mechanism is not measured."
    )
    return render_latex(header, table_rows, caption, "tab:e1_rtt",
                        fontsize="\\footnotesize", tabcolsep="3pt")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--broker-suffix", default="_rtt50")
    args = parser.parse_args(argv)

    rows = build(args.metrics, args.broker_suffix)
    numbers = NumberRegistry()
    latex = to_latex(rows, numbers)
    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    (args.out / "tables" / "15_e1_rtt.tex").write_text(latex)
    numbers.write(args.out / "e1_numbers.tex")
    for label, m0, m50, _k0, _k50 in rows:
        print(f"{label}: E1 RTT0={m0:.1f} RTT50={m50:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
