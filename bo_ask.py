#!/usr/bin/env python3
"""
bo_ask.py — runs INSIDE the container.

Loads (or creates) the persistent skopt Optimizer, asks for the next parameter
vector, writes a candidate Config.toml, and records the pending point so that
bo_tell.py can close the loop. Each invocation is independent and stateful via
an on-disk pickle, so the host-side bash loop can call it one iteration at a time
and resume after any interruption.

Usage (host calls this via apptainer exec):
    python3 bo_ask.py --state DIR --template Config.toml --out config_iter.toml
"""
from __future__ import annotations
import argparse
import json
import pickle
import re
import numpy as np
from pathlib import Path

from skopt import Optimizer
from skopt.space import Real, Integer

# Search space (must match bo_tell.py)
SPACE = [
    Real(0.3, 0.9,      name="alfa"),
    Integer(200, 1000,  name="gossip_interval_ms"),
    Integer(1, 4,       name="neighbor_k"),
    Real(100.0, 500.0,  name="t_base_ms"),
    Integer(8, 10,      name="cluster_resolution"),
    Integer(13, 15,     name="spot_resolution"),
]
DIM_NAMES = [d.name for d in SPACE]
N_INITIAL = 12
SEED = 42


# Default (baseline) parameters — used as the very first evaluation so the BO
# trace starts from the known operating point and convergence is measured
# against it.
DEFAULT_PARAMS = {
    "alfa": 0.7,
    "gossip_interval_ms": 500,
    "neighbor_k": 2,
    "t_base_ms": 200.0,
    "cluster_resolution": 9,
    "spot_resolution": 14,
}


def get_optimizer(state_dir: Path) -> Optimizer:
    pkl = state_dir / "optimizer.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return Optimizer(
        dimensions=SPACE,
        base_estimator="GP",
        n_initial_points=N_INITIAL,
        random_state=SEED,
        acq_func="EI",
    )


def save_optimizer(opt: Optimizer, state_dir: Path) -> None:
    with open(state_dir / "optimizer.pkl", "wb") as f:
        pickle.dump(opt, f)


def write_config(params: dict, template: Path, dest: Path) -> dict:
    alfa    = round(float(params["alfa"]), 4)
    beta    = round(1.0 - alfa, 4)
    cluster = int(params["cluster_resolution"])
    spot    = int(params["spot_resolution"])
    if spot <= cluster:           # enforce spot finer than cluster
        spot = cluster + 1
        params = {**params, "spot_resolution": spot}

    text = template.read_text()

    def sub(pat, rep, s):
        new, n = re.subn(pat, rep, s, count=1)
        if n == 0:
            raise RuntimeError(f"pattern not found in template: {pat}")
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
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="directory where config_b{i}.toml files are written")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="how many candidate configs to propose at once "
                         "(1 = sequential BO; >1 = parallel constant-liar batch)")
    args = ap.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    opt = get_optimizer(args.state)
    n = max(1, args.batch_size)

    # Iteration 0 special-case: the very first proposed point is forced to the
    # default config (integrated baseline). Only applies when the optimizer has
    # no observations yet and nothing is pending.
    is_first = (len(opt.Xi) == 0) and not (args.state / "pending.json").exists()

    # Ask for the batch. constant-liar ("cl_min") makes the N points within a
    # batch differ from each other by temporarily assuming a pessimistic result
    # for the points already chosen in this same batch.
    if n == 1:
        xs = [[DEFAULT_PARAMS[d.name] for d in SPACE]] if is_first else [opt.ask()]
    else:
        xs = opt.ask(n_points=n, strategy="cl_min")
        if is_first:
            # replace the first proposed point with the default baseline
            xs[0] = [DEFAULT_PARAMS[d.name] for d in SPACE]

    pend_points = []
    for i, x in enumerate(xs):
        params = dict(zip(DIM_NAMES, x))
        dest = args.out_dir / f"config_b{i}.toml"
        params = write_config(params, args.template, dest)
        x_native = [int(v) if isinstance(v, (int, np.integer)) else float(v)
                    for v in x]
        p_native = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                    for k, v in params.items()}
        pend_points.append({"x": x_native, "params": p_native,
                            "config": str(dest)})
        print(f"[ask] b{i}: {json.dumps(p_native)}")

    # one pending file holding the whole batch; tell consumes it in order
    (args.state / "pending.json").write_text(
        json.dumps({"batch": pend_points}, indent=2))
    save_optimizer(opt, args.state)



if __name__ == "__main__":
    main()
