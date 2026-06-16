#!/usr/bin/env python3
"""
bo_plots.py — runs INSIDE the container.

Generates the Bayesian-optimization-specific figures that post_process.py does
not produce (it knows about single runs, not about the optimization trace):

  1. convergence  — objective vs iteration, plus running best-so-far
  2. param importance — correlation of each parameter with the objective
                        (a cheap, model-free sensitivity proxy)

Called by bo_tell.py after each iteration (convergence updates live), and can be
run standalone at the end.

    python3 bo_plots.py --results bo_results.csv --out-dir bo_runs/plots
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PARAMS = ["alfa", "gossip_interval_ms", "neighbor_k",
          "t_base_ms", "cluster_resolution", "spot_resolution"]


def plot_convergence(df: pd.DataFrame, out_dir: Path) -> None:
    df = df.sort_values("iter")
    best = df["objective"].cummin()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["iter"], df["objective"], "o-", alpha=0.5,
            label="objective per iteration")
    ax.plot(df["iter"], best, "-", linewidth=2.5, label="best so far")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective (lower is better)")
    ax.set_title("Bayesian optimization convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "BO_convergence.png", dpi=150)
    plt.close(fig)


def plot_importance(df: pd.DataFrame, out_dir: Path) -> None:
    # model-free proxy: |Spearman correlation| of each param with the objective
    present = [p for p in PARAMS if p in df.columns]
    if len(df) < 4 or not present:
        return
    corr = (df[present + ["objective"]]
            .corr(method="spearman")["objective"]
            .drop("objective").abs().sort_values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(corr.index, corr.values)
    ax.set_xlabel("|Spearman correlation| with objective")
    ax.set_title("Parameter influence on the objective (model-free proxy)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "BO_param_importance.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.results.exists():
        print("no results yet, skipping plots")
        return
    df = pd.read_csv(args.results)
    if df.empty:
        return
    plot_convergence(df, args.out_dir)
    plot_importance(df, args.out_dir)
    print(f"BO plots → {args.out_dir}")


if __name__ == "__main__":
    main()
