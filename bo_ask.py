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
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)

    opt = get_optimizer(args.state)

    # Iteration 0: use the default config as the integrated baseline, so the BO
    # trace's first point is the known operating point. skopt still 'asks' it
    # (we override x), and bo_tell will tell() the measured objective back.
    is_first = (len(opt.Xi) == 0) and not (args.state / "pending.json").exists()
    if is_first:
        x = [DEFAULT_PARAMS[d.name] for d in SPACE]
    else:
        x = opt.ask()
    params = dict(zip(DIM_NAMES, x))
    params = write_config(params, args.template, args.out)

    # record pending point (x as asked) so tell can match it exactly
    pending = {"x": x, "params": {k: (float(v) if isinstance(v, float) else int(v))
                                  for k, v in params.items()}}
    (args.state / "pending.json").write_text(json.dumps(pending, indent=2))
    save_optimizer(opt, args.state)

    print(json.dumps(pending["params"]))  # host can log this


if __name__ == "__main__":
    main()
