#!/usr/bin/env python3
"""bo_sensitivity.py — sensitivity analysis for BO results.

Generates scatter + trend plots for each of the 6 parameters, coloured by
n_scenarios coverage.  Also prints per-level descriptive statistics.

Usage:
    python3 bo_sensitivity.py --results ../risultati_PostProcessing/bo_results_RECOVERED.csv \
                              --out-dir ../risultati_PostProcessing/bo_runs/sensitivity
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PARAMS = [
    ("alfa", "float", r"$\alpha$"),
    ("gossip_interval_ms", "int", "gossip interval (ms)"),
    ("neighbor_k", "int", r"neighbor $k$"),
    ("t_base_ms", "float", r"$t_{base}$ (ms)"),
    ("cluster_resolution", "int", "cluster resolution"),
    ("spot_resolution", "int", "spot resolution"),
]

CSS = {20: "#2166ac", 14: "#b2182b"}  # full / min coverage


def _coverage_label(n: int) -> str:
    if n == 20:
        return "full (20/20)"
    return f"partial ({n}/20)"


def _discrete_stats(df: pd.DataFrame, col: str, out_lines: list[str]) -> None:
    """Print per-level statistics for a discrete parameter."""
    grouped = df.groupby(col)["objective"]
    out_lines.append(f"\n## {col} (discrete)")
    for val, grp in grouped:
        n = len(grp)
        marker = " ⚠ <3 pts" if n < 3 else ""
        out_lines.append(
            f"  {col}={val}: n={n}, mean={grp.mean():.4f}, "
            f"min={grp.min():.4f}, max={grp.max():.4f}{marker}"
        )


def _continuous_binned(df: pd.DataFrame, col: str, out_lines: list[str]) -> None:
    """Bin a continuous parameter into 3 equal-frequency bins and print stats."""
    n_unique = df[col].nunique()
    n_bins = min(3, n_unique)
    try:
        _, bins = pd.qcut(df[col], q=n_bins, retbins=True, duplicates="drop")
        n_actual = len(bins) - 1
        labels = [f"bin{i+1}" for i in range(n_actual)]
        df["_bin"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    except ValueError:
        # fallback: cut into equal-width
        n_actual = min(3, n_unique)
        labels = [f"bin{i+1}" for i in range(n_actual)]
        df["_bin"] = pd.cut(df[col], bins=n_actual, labels=labels)
    grouped = df.groupby("_bin", observed=True)["objective"]
    out_lines.append(f"\n## {col} (continuous, {n_actual}-bin)")
    for label, grp in grouped:
        n = len(grp)
        lo = df.loc[grp.index, col].min()
        hi = df.loc[grp.index, col].max()
        marker = " ⚠ <3 pts" if n < 3 else ""
        out_lines.append(
            f"  {label} [{lo:.0f}–{hi:.0f}]: n={n}, "
            f"mean={grp.mean():.4f}, min={grp.min():.4f}, max={grp.max():.4f}{marker}"
        )
    df.drop(columns="_bin", inplace=True)


def _rolling_trend(x: np.ndarray, y: np.ndarray, window: int = 5
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Simple rolling-mean trend, handling NaNs."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    trend = np.full_like(ys, np.nan, dtype=float)
    half = window // 2
    for i in range(len(ys)):
        lo = max(0, i - half)
        hi = min(len(ys), i + half + 1)
        trend[i] = np.nanmean(ys[lo:hi])
    return xs, trend


def plot_sensitivity_all(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """Generate one scatter+trend plot per parameter, return stats lines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_lines: list[str] = []

    for col, ptype, label in PARAMS:
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Colour by coverage
        for n_val, grp in df.groupby("n_scenarios"):
            ax.scatter(grp[col], grp["objective"],
                       c=CSS.get(n_val, "#888888"),
                       label=_coverage_label(n_val), alpha=0.7, edgecolors="none",
                       s=40)

        # Rolling trend (over all points)
        xs, trend = _rolling_trend(df[col].values.astype(float),
                                   df["objective"].values.astype(float),
                                   window=7)
        ax.plot(xs, trend, "-", color="black", linewidth=1.5, alpha=0.6,
                label="trend (rolling mean, w=7)")

        ax.set_xlabel(label)
        ax.set_ylabel("Objective (lower is better)")
        ax.set_title(f"Objective vs {label}")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"BO_{col}.png", dpi=150)
        plt.close(fig)

        # Statistics
        if ptype == "int" and df[col].nunique() <= 6:
            _discrete_stats(df, col, stats_lines)
        else:
            _continuous_binned(df, col, stats_lines)

    return stats_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    print(f"Read {len(df)} observations from {args.results}")

    stats = plot_sensitivity_all(df, args.out_dir)
    print(f"\nSensitivity plots → {args.out_dir}")

    # Print stats to stdout
    print("\n===== PER-LEVEL STATISTICS =====")
    for line in stats:
        print(line)
    print("===== END STATISTICS =====\n")


if __name__ == "__main__":
    main()
