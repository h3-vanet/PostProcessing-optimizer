#!/usr/bin/env python3
"""Vector figures kept by the paper, from the campaign summary CSVs.

Currently: park rate versus density, one line per arm (GeoGrid k=1, k=2,
broker), mean over valid runs with sample-SD error bars, written as a vector
PDF to `<out>/figures/park_rate_density.pdf`.

Usage:
    python3 paper_figures.py --metrics DIR [DIR ...] --out DIR \
        [--broker-suffix _rtt50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_tables import (
    DENSITY_LABELS,
    DENSITY_ORDER,
    VALID_NR_SIM_T_REACHED,
)

# Densities common to both arms (leaderless caos is not evaluable).
PLOT_DENSITIES = ["sparse", "nominal", "congested"]

SERIES = [
    ("GeoGrid k=1", "post_simtime_ll"),
    ("GeoGrid k=2", "post_simtime_ll_k2"),
    ("Broker", "post_simtime_br"),
]


def _load(metrics_dirs: list[Path], name: str, suffix: str) -> pd.DataFrame | None:
    dirname = name + suffix if name == "post_simtime_br" else name
    path = None
    for metrics_dir in metrics_dirs:
        candidate = metrics_dir / dirname / "summary_all_experiments.csv"
        if candidate.exists():
            path = candidate
            break
    if path is None:
        return None
    df = pd.read_csv(path)
    df["density"] = df["traffic"].map(DENSITY_LABELS)
    df["valid"] = df["nr_sim_t_reached"] >= VALID_NR_SIM_T_REACHED
    return df


def park_rate_by_density(metrics_dirs: list[Path] | Path, suffix: str = "_rtt50") -> dict:
    """{arm: {density: (mean, sd, n)}} for the densities with valid runs."""
    import statistics

    if isinstance(metrics_dirs, (str, Path)):
        metrics_dirs = [Path(metrics_dirs)]
    else:
        metrics_dirs = [Path(d) for d in metrics_dirs]
    out = {}
    for arm, series in SERIES:
        df = _load(metrics_dirs, series, suffix)
        if df is None:
            continue
        per = {}
        for density in DENSITY_ORDER:
            vals = df[(df["valid"]) & (df["density"] == density)]["C1_park_rate"].tolist()
            if not vals:
                continue
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            per[density] = (mean, sd, len(vals))
        out[arm] = per
    return out


def plot_park_rate(data: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {"GeoGrid k=1": "o", "GeoGrid k=2": "s", "Broker": "^"}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for arm, per in data.items():
        xs = [d for d in PLOT_DENSITIES if d in per]
        ys = [per[d][0] for d in xs]
        es = [per[d][1] for d in xs]
        ax.errorbar(
            range(len(xs)), ys, yerr=es, marker=markers.get(arm, "o"),
            capsize=2, lw=1, ms=3, label=arm,
        )
    ax.set_xticks(range(len(PLOT_DENSITIES)))
    ax.set_xticklabels(PLOT_DENSITIES, fontsize=6)
    ax.set_ylabel("Park rate (%)", fontsize=7)
    ax.legend(fontsize=5, frameon=False)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--broker-suffix", default="_rtt50")
    args = parser.parse_args(argv)

    data = park_rate_by_density(args.metrics, args.broker_suffix)
    plot_park_rate(data, args.out / "figures" / "park_rate_density.pdf")
    for arm, per in data.items():
        print(arm, {d: f"{v[0]:.1f}" for d, v in per.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
