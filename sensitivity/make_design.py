#!/usr/bin/env python3
"""Phase-0/Phase-A design generator for the GeoGrid sensitivity study.

Builds an LHS design over the 5-D primary parameter space, always includes the
paper baseline as an explicitly flagged configuration, writes one leaderless
``Config.toml`` per configuration, and records everything in ``design.json``.

By default it runs the Phase-0 preflight first and refuses to generate anything
if a check FAILs (use ``--skip-preflight`` only for local dry-runs, e.g. with a
local leaderless template). It never touches the simulator or the campaign data.

Usage:

    python3 make_design.py --base ~/vanet-parking --sif-suffix <T> \
        --n-configs 10 --out out

    # local dry-run against a checked-out leaderless config
    python3 make_design.py --template ../../vanet-parking/Config.toml \
        --skip-preflight --out /tmp/sens_design
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Symmetric Latin Hypercube of *n* points in the unit cube, shape (n, d).

    Classical stratified construction: one jittered sample per stratum per
    dimension, strata independently permuted. Deterministic for a fixed rng.
    """
    if n < 1:
        return np.empty((0, d))
    u = rng.random((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        u[:, j] = (perm + u[:, j]) / n
    return u


def build_configs(n_configs: int, seed: int) -> list[dict]:
    """Return the ordered list of configurations: baseline first, then LHS."""
    rng = np.random.default_rng(seed)
    dims = list(spec.PRIMARY_PARAMS)
    # n_configs - 1 LHS points; the baseline is the explicit first point.
    u = lhs(max(0, n_configs - 1), len(dims), rng)

    configs: list[dict] = []
    baseline_params = spec.normalise(spec.BASELINE)
    configs.append({"params": baseline_params, "is_baseline": True})
    for row in u:
        raw = {p.name: p.lo + float(x) * (p.hi - p.lo)
               for p, x in zip(dims, row)}
        configs.append({"params": spec.normalise(raw), "is_baseline": False})

    for c in configs:
        errs = spec.constraint_violations(c["params"])
        if errs:
            raise RuntimeError(f"constraint violation in design: {errs}")
    return configs


def report_design(records: list[dict], out: Path) -> Counter:
    """Write ``design_matrix.csv`` and print the post-snapping matrix + a
    duplicate-level report, so the design can be inspected *before* spending
    the 25-run battery. Returns the exact-duplicate count among LHS points.
    """
    params = [p.name for p in spec.PRIMARY_PARAMS]
    path = out / "design_matrix.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config_id", "is_baseline"] + params)
        for r in records:
            w.writerow([r["config_id"], int(r["is_baseline"])]
                       + [r["params"][k] for k in params])

    print(f"\ndesign matrix ({len(records)} x {len(params)}) -> {path}")
    header = "config".ljust(9) + "base " + "".join(p.rjust(12) for p in params)
    print(header)
    for r in records:
        row = r["config_id"].ljust(9) + ("yes  " if r["is_baseline"] else "no   ")
        row += "".join(str(r["params"][k]).rjust(12) for k in params)
        print(row)

    lhs = [r for r in records if not r["is_baseline"]]
    print("\nunique levels (LHS points only, baseline excluded):")
    for k in params:
        vals = [r["params"][k] for r in lhs]
        counts = Counter(vals)
        dup = {v: n for v, n in counts.items() if n > 1}
        note = f"  duplicates: {dup}" if dup else ""
        print(f"  {k:<20} {len(counts)} unique / {len(vals)}{note}")

    seen: Counter = Counter(tuple(r["params"][k] for k in params) for r in lhs)
    dup_rows = {row: n for row, n in seen.items() if n > 1}
    if dup_rows:
        print(f"\nWARNING: {len(dup_rows)} duplicated LHS row(s): {dup_rows}")
    else:
        print("\nno duplicated LHS rows")
    return Counter({k: v for k, v in dup_rows.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="~/vanet-parking")
    ap.add_argument("--template", default=None,
                    help="leaderless Config.toml used as the run template")
    ap.add_argument("--mappa", default=None)
    ap.add_argument("--sumo-cfg-dir", default=None)
    ap.add_argument("--sif-dir", default=None)
    ap.add_argument("--sif-suffix", default=None)
    ap.add_argument("--n-configs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip Phase-0 checks (local dry-run only)")
    ap.add_argument("--force", action="store_true",
                    help="generate even if preflight has FAILs")
    args = ap.parse_args()

    base = Path(args.base).expanduser()
    if args.template:
        template = Path(args.template).expanduser()
    else:
        template = spec.find_template([
            base / "configs_next" / "next_leaderless_k2.toml",
            base / "configs_next" / "next_leaderless.toml",
            base / "data" / "Config.toml",
        ])
    if template is None or not template.is_file():
        print(f"ERROR: no leaderless template found (base={base}); "
              f"pass --template", file=sys.stderr)
        return 2

    if not args.skip_preflight:
        import preflight
        ns = argparse.Namespace(
            base=args.base, mappa=args.mappa, sumo_cfg_dir=args.sumo_cfg_dir,
            sif_dir=args.sif_dir, sif_suffix=args.sif_suffix, template=str(template),
            json=None)
        checks = preflight.run(ns)
        n_fail = sum(c.status == preflight.FAIL for c in checks)
        for c in checks:
            print(f"  [{c.status}] {c.name}: {c.detail}")
        if n_fail and not args.force:
            print(f"ERROR: preflight has {n_fail} FAIL(s); aborting "
                  f"(use --force to override)", file=sys.stderr)
            return 1

    template_text = template.read_text()
    configs = build_configs(args.n_configs, args.seed)

    out = args.out.expanduser()
    cfg_dir = out / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    start, stop = spec.DENSITY_ARRAY_RANGE[spec.DENSITY]
    records = []
    for i, c in enumerate(configs):
        cid = f"sens_c{i:02d}"
        toml_path = cfg_dir / f"{cid}.toml"
        toml_path.write_text(spec.render_config(template_text, c["params"]))
        records.append({
            "config_id": cid,
            "is_baseline": c["is_baseline"],
            "params": {k: (int(v) if float(v).is_integer() and k in
                           ("trajectory_window", "gossip_interval_ms")
                           else round(float(v), 4))
                       for k, v in c["params"].items()},
            "beta": spec.beta_of(c["params"]["alfa"]),
            "config_toml": str(toml_path.relative_to(out)),
            "log_subdir": cid,
            "zmq_offset_base": spec.ZMQ_OFFSET_GLOBAL_BASE + i * spec.ZMQ_OFFSET_BLOCK,
            "array": [start, stop],
        })

    design = {
        "study": "geogrid_sensitivity_nominal",
        "phase": "A",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "design": "lhs",
        "rng_seed": args.seed,
        "n_configs": len(records),
        "density": spec.DENSITY,
        "density_label": spec.DENSITY_LABEL[spec.DENSITY],
        "occupancies": list(spec.OCCUPANCIES),
        "seeds": list(spec.SEEDS),
        "runs_per_config": spec.RUNS_PER_CONFIG,
        "total_runs": spec.RUNS_PER_CONFIG * len(records),
        "template": str(template),
        "parameters": {p.name: {"toml_key": p.toml_key, "lo": p.lo, "hi": p.hi,
                                "kind": p.kind, "snap": p.snap, "unit": p.unit}
                       for p in spec.PRIMARY_PARAMS},
        "baseline": {"config_id": records[0]["config_id"],
                     "params": records[0]["params"],
                     "beta": records[0]["beta"]},
        "fixed": {f"{s}.{k}": v for (s, k), v in spec.FIXED_VALUES.items()},
        "configs": records,
    }
    design_path = out / "design.json"
    design_path.write_text(json.dumps(design, indent=2))

    report_design(records, out)

    print(f"\nWrote {len(records)} configs -> {cfg_dir}")
    print(f"design.json -> {design_path}")
    print(f"runs: {len(records)} x {spec.RUNS_PER_CONFIG} = {design['total_runs']}")
    for r in records:
        tag = " (baseline)" if r["is_baseline"] else ""
        p = r["params"]
        print(f"  {r['config_id']}{tag}: alfa={p['alfa']}, t_base={p['t_base_ms']}, "
              f"Dmax={p['max_distance_m']}, ntr={p['trajectory_window']}, "
              f"Tgossip={p['gossip_interval_ms']}, offset={r['zmq_offset_base']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
