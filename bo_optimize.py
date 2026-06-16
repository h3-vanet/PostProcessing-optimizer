#!/usr/bin/env python3
"""
bo_optimize.py — Bayesian optimization of the vanet-parking coordination
parameters on the Athena cluster.

Per iteration:
  1. skopt proposes a parameter vector.
  2. A candidate Config.toml is generated (beta = 1 - alfa; spot > cluster).
  3. The 20-scenario array job is submitted with BO_CONFIG pointing at it.
  4. We poll `squeue` until the array finishes.
  5. post_process.py is run; summary_all_experiments.csv is read.
  6. The weighted OBJ_C objective is computed and returned to skopt.
  7. Logs + post-processing output are tarred for download; state is checkpointed.

Run it on the LOGIN NODE inside tmux:
    tmux new -s bo
    source venv/bin/activate        # env with scikit-optimize, pandas
    python3 bo_optimize.py --n-calls 60 --max-parallel 10
    # detach: Ctrl-b then d ;  reattach: tmux attach -t bo
"""

from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import tarfile
import time
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# ─────────────────────────────────────────────────────────────────────────────
# Paths (adjust BASE if your tree differs)
# ─────────────────────────────────────────────────────────────────────────────
HOME          = Path.home()
BASE          = HOME / "vanet-parking"
DATA_DIR      = BASE / "data"
LOG_BASE      = BASE / "logs"
TEMPLATE_CFG  = DATA_DIR / "Config.toml"          # base config (localhost zmq etc.)
SLURM_SCRIPT  = BASE / "run_all_scenarios.slurm"
POST_PROCESS  = Path(__file__).resolve().parent / "post_process.py"

BO_DIR        = BASE / "bo_runs"                   # all BO artifacts live here
BO_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT    = BO_DIR / "checkpoint.json"
RESULTS_CSV   = BO_DIR / "bo_results.csv"

N_SCENARIOS   = 20                                 # array 0-19

# ─────────────────────────────────────────────────────────────────────────────
# Search space (6 dims). beta = 1 - alfa is derived, not searched.
# Constraint spot_resolution > cluster_resolution is enforced at config gen.
# ─────────────────────────────────────────────────────────────────────────────
SPACE = [
    Real(0.3, 0.9,      name="alfa"),
    Integer(200, 1000,  name="gossip_interval_ms"),
    Integer(1, 4,       name="neighbor_k"),
    Real(100.0, 500.0,  name="t_base_ms"),
    Integer(8, 10,      name="cluster_resolution"),
    Integer(13, 15,     name="spot_resolution"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Objective weights (OBJ_C) and occupancy weighting
# ─────────────────────────────────────────────────────────────────────────────
W_CLEAN_SUCCESS = 0.40   # (1 - success*(1-double))
W_UNFAIR        = 0.35   # (1 - jain)
W_LATENCY       = 0.25   # latency_p50 / LAT_NORM
LAT_NORM        = 2000.0
HIGH_OCC_THRESHOLD = 85  # occupancy >= 85 weighted x2
HIGH_OCC_WEIGHT    = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Config generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_config(params: dict, dest: Path) -> None:
    """Write a Config.toml from the template, overriding the searched params.
    Enforces beta = 1 - alfa and spot_resolution > cluster_resolution."""
    alfa    = round(float(params["alfa"]), 4)
    beta    = round(1.0 - alfa, 4)
    cluster = int(params["cluster_resolution"])
    spot    = int(params["spot_resolution"])
    # Safety: spot must be finer (larger H3 res number) than cluster.
    if spot <= cluster:
        spot = cluster + 1

    text = TEMPLATE_CFG.read_text()

    def sub(pattern, replacement, s):
        new, n = re.subn(pattern, replacement, s, count=1)
        if n == 0:
            raise RuntimeError(f"pattern not found in Config.toml: {pattern}")
        return new

    text = sub(r"cluster_resolution\s*=\s*\d+",
               f"cluster_resolution   = {cluster}", text)
    text = sub(r"spot_resolution\s*=\s*\d+",
               f"spot_resolution      = {spot}", text)
    text = sub(r"gossip_interval_ms\s*=\s*\d+",
               f"gossip_interval_ms          = {int(params['gossip_interval_ms'])}", text)
    text = sub(r"neighbor_k\s*=\s*\d+",
               f"neighbor_k                  = {int(params['neighbor_k'])}", text)
    text = sub(r"alfa\s*=\s*[\d.]+",
               f"alfa                     = {alfa}", text)
    text = sub(r"beta\s*=\s*[\d.]+",
               f"beta                     = {beta}", text)
    text = sub(r"t_base_ms\s*=\s*[\d.]+",
               f"t_base_ms                = {float(params['t_base_ms']):.1f}", text)

    dest.write_text(text)


# ─────────────────────────────────────────────────────────────────────────────
# SLURM submission + polling
# ─────────────────────────────────────────────────────────────────────────────
def submit_array(config_path: Path, max_parallel: int) -> str:
    """sbatch the 20-scenario array with BO_CONFIG set; return the job id."""
    env = dict(os.environ, BO_CONFIG=str(config_path))
    out = subprocess.run(
        ["sbatch", f"--array=0-{N_SCENARIOS - 1}%{max_parallel}",
         str(SLURM_SCRIPT)],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(f"could not parse job id from: {out!r}")
    return m.group(1)


def wait_for_job(job_id: str, poll_s: int = 60, timeout_h: float = 24) -> None:
    """Poll squeue until no task of this array job remains."""
    deadline = time.time() + timeout_h * 3600
    user = os.environ.get("USER", "")
    while True:
        r = subprocess.run(["squeue", "-u", user, "-h", "-o", "%A %t"],
                            capture_output=True, text=True)
        # match both "12345" and array tasks "12345_3"
        active = [ln for ln in r.stdout.splitlines()
                  if ln.strip().startswith(job_id)]
        if not active:
            return
        if time.time() > deadline:
            raise TimeoutError(f"job {job_id} exceeded {timeout_h}h")
        running = sum(1 for ln in active if ln.split()[-1] == "R")
        print(f"    [{datetime.now():%H:%M:%S}] job {job_id}: "
              f"{len(active)} task(s) left ({running} running)", flush=True)
        time.sleep(poll_s)


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing + objective
# ─────────────────────────────────────────────────────────────────────────────
def run_post_process(out_dir: Path) -> pd.DataFrame:
    """Run post_process.py over LOG_BASE, writing into out_dir; return summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(POST_PROCESS),
         "--log-base", str(LOG_BASE), "--out", str(out_dir)],
        check=True,
    )
    summ = out_dir / "summary_all_experiments.csv"
    if not summ.exists():
        # fall back to default location used by the script
        summ = LOG_BASE.parent / "summary_all_experiments.csv"
    return pd.read_csv(summ)


def compute_objective(df: pd.DataFrame) -> float:
    """Weighted OBJ_C across the 20 scenarios (occupancy >=85 weighted x2)."""
    d = df.copy()
    double_n = d["A5b_double_rate_pct"] / 100.0
    net_ok   = (d["A2_success_rate_pct"] / 100.0) * (1.0 - double_n)
    unfair_n = 1.0 - d["A5_jain"]
    lat_n    = (d["A3_p50_ms"] / LAT_NORM).clip(0, 1)
    obj = (W_CLEAN_SUCCESS * (1.0 - net_ok)
           + W_UNFAIR * unfair_n
           + W_LATENCY * lat_n)
    w = d["occupancy"].apply(
        lambda o: HIGH_OCC_WEIGHT if o >= HIGH_OCC_THRESHOLD else 1.0)
    return float((obj * w).sum() / w.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Archiving + checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def archive_iteration(it: int, config_path: Path, post_dir: Path,
                      objective: float) -> Path:
    """Tar logs + post-processing output + the config for this iteration."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tar_path = BO_DIR / f"iter_{it:03d}_{stamp}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(LOG_BASE,     arcname=f"iter_{it:03d}/logs")
        tar.add(post_dir,     arcname=f"iter_{it:03d}/post")
        tar.add(config_path,  arcname=f"iter_{it:03d}/Config.toml")
        meta = BO_DIR / f"_meta_{it}.json"
        meta.write_text(json.dumps({"iter": it, "objective": objective}, indent=2))
        tar.add(meta, arcname=f"iter_{it:03d}/meta.json")
        meta.unlink()
    print(f"    archived → {tar_path.name}", flush=True)
    return tar_path


def append_result(row: dict) -> None:
    df = pd.DataFrame([row])
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main optimization loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calls", type=int, default=60,
                    help="total BO evaluations (>= n-initial)")
    ap.add_argument("--n-initial", type=int, default=12,
                    help="random exploration points before the GP kicks in")
    ap.add_argument("--max-parallel", type=int, default=10,
                    help="max simultaneous array tasks (sbatch %%N)")
    ap.add_argument("--poll", type=int, default=60, help="squeue poll seconds")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    iteration = {"n": 0}

    @use_named_args(SPACE)
    def objective_fn(**params):
        it = iteration["n"]
        iteration["n"] += 1
        print(f"\n{'='*70}\n[iter {it}] params: {params}\n{'='*70}", flush=True)

        cfg = BO_DIR / f"config_iter_{it:03d}.toml"
        generate_config(params, cfg)

        job_id = submit_array(cfg, args.max_parallel)
        print(f"    submitted array job {job_id}", flush=True)
        wait_for_job(job_id, poll_s=args.poll)

        post_dir = BO_DIR / f"post_iter_{it:03d}"
        df = run_post_process(post_dir)
        score = compute_objective(df)
        print(f"    objective = {score:.4f}", flush=True)

        archive_iteration(it, cfg, post_dir, score)
        row = dict(params)
        row.update({"iter": it, "job_id": job_id, "objective": score,
                    "beta": round(1 - params["alfa"], 4)})
        append_result(row)
        return score

    print(f"Starting BO: {args.n_calls} calls "
          f"({args.n_initial} random), max {args.max_parallel} parallel tasks.",
          flush=True)
    res = gp_minimize(
        objective_fn, SPACE,
        n_calls=args.n_calls, n_initial_points=args.n_initial,
        random_state=args.seed, verbose=True,
    )

    best = dict(zip([d.name for d in SPACE], res.x))
    best["beta"] = round(1 - best["alfa"], 4)
    print(f"\n{'#'*70}\nBEST objective = {res.fun:.4f}\nBEST params = "
          f"{json.dumps(best, indent=2)}\n{'#'*70}")
    (BO_DIR / "best_params.json").write_text(json.dumps(
        {"objective": res.fun, "params": best}, indent=2))


if __name__ == "__main__":
    main()
