#!/usr/bin/env python3
"""Phase-0 preflight checks for the GeoGrid sensitivity study.

Turns the four previously "assumed" facts into automatic checks that must pass
(or be explicitly waived) before the first ``sbatch``:

  1. the leaderless (gossip) SIF image suffix exists;
  2. the 25 nominal scenario cells map to the expected array indices;
  3. the per-seed route and parking layouts exist for seeds 1..5;
  4. the run template is a valid leaderless baseline with the paper's fixed
     values (k=2, H3 resolutions, TTL, ...), and the 5 knobs respect their
     constraints.

Usage (standalone, read-only):

    python3 preflight.py --base ~/vanet-parking --sif-suffix <T>
    python3 preflight.py --base . --sumo-cfg-dir <local>/mappa/sumo_cfg_seeds \
        --mappa <local>/mappa --template <leaderless Config.toml>

Exit code is 0 only if no check is FAIL; WARN never fails the run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _check_template(template: Path) -> list[Check]:
    checks: list[Check] = []
    if not template or not template.is_file():
        return [Check("template", FAIL,
                      f"leaderless template not found (tried: {template})")]
    cfg = spec.read_toml(template)
    checks.append(Check("template", PASS, f"{template}"))

    backend = spec.effective_value(cfg, "coordinator", "backend")
    checks.append(Check(
        "backend=leaderless", PASS if backend == "leaderless" else FAIL,
        f"coordinator.backend = {backend!r}"))

    bad = []
    for (section, key), expected in spec.FIXED_VALUES.items():
        got = spec.effective_value(cfg, section, key)
        if got != expected:
            bad.append(f"{section}.{key}={got!r} (expected {expected!r})")
    checks.append(Check("fixed values", PASS if not bad else FAIL,
                        "; ".join(bad) if bad else
                        f"{len(spec.FIXED_VALUES)} fixed values match Table III"))

    nae = spec.effective_value(cfg, "gossip", "anti_entropy_every_n_rounds")
    checks.append(Check("N_ae present", PASS if nae is not None else WARN,
                        f"gossip.anti_entropy_every_n_rounds = {nae}"))

    # The five knobs must be absent-or-present and inside bounds. They are
    # overwritten by make_design regardless, so a missing one is only a warning.
    missing = [p.toml_key for p in spec.PRIMARY_PARAMS
               if f"\n{p.toml_key}" not in f"\n{template.read_text()}"
               and f" {p.toml_key}" not in template.read_text()]
    checks.append(Check("knob keys present", PASS if not missing else WARN,
                        ("missing (will be appended): " + ", ".join(missing))
                        if missing else "all 5 knob keys present"))
    return checks


def _check_sif(sif_dir: Path, suffix: str | None) -> list[Check]:
    checks: list[Check] = []
    if not sif_dir.is_dir():
        return [Check("SIF dir", FAIL, f"{sif_dir} is not a directory")]
    available = sorted(p.name for p in sif_dir.glob("*.sif"))
    if not suffix:
        cands = [n for n in available if "core" in n]
        return [Check("SIF suffix", WARN,
                      "not provided; choose from available core images: "
                      + ", ".join(cands or ["<none>"]))]
    trio = [f"van3twin_{suffix}.sif", f"bridge_{suffix}.sif",
            f"core_{suffix}.sif"]
    missing = [n for n in trio if not (sif_dir / n).is_file()]
    checks.append(Check("SIF images", PASS if not missing else FAIL,
                        "found " + ", ".join(trio) if not missing else
                        "missing " + ", ".join(missing)))
    return checks


def _parse_sumo_cfg(path: Path):
    """Return (route-files value, additional-files value, seed) from a .sumo.cfg."""
    root = ET.parse(path).getroot()
    inp = root.find("./input")
    route = inp.find("route-files").get("value") if inp is not None and \
        inp.find("route-files") is not None else None
    addl = inp.find("additional-files").get("value") if inp is not None and \
        inp.find("additional-files") is not None else None
    seed_el = root.find("./random_number/seed")
    seed = seed_el.get("value") if seed_el is not None else None
    return route, addl, seed


def _vehicles_for_cfg(path: Path) -> int | None:
    """Count <vehicle> entries in the route file referenced by one .sumo.cfg."""
    route, _addl, _seed = _parse_sumo_cfg(path)
    if not route:
        return None
    # route-files value is relative to the .sumo.cfg location (e.g. ../scenari_seed/..)
    for candidate in route.split(","):
        rp = (path.parent / candidate.strip()).resolve()
        if rp.is_file():
            return len(re.findall(r"<vehicle\b", rp.read_text(errors="replace")))
    return None


_CFG_NAME_RE = re.compile(
    r"combination_(?P<dens>[a-z]+)_routes_parking_occupied_"
    r"(?P<occ>\d+)_seed(?P<seed>\d+)")


def _check_cfg_content(sumo_cfg_dir: Path, block: list[str]) -> Check:
    """Content check of the nominal block: embedded seed/occupancy must match
    the filename and the referenced route file must carry the nominal fleet."""
    errors: list[str] = []
    counts: set[int] = set()
    for name in block:
        m = _CFG_NAME_RE.match(name)
        if not m:
            errors.append(f"unparseable name: {name}")
            continue
        occ, seed = m.group("occ"), m.group("seed")
        cfg_path = sumo_cfg_dir / name
        try:
            route, addl, seedval = _parse_sumo_cfg(cfg_path)
        except (ET.ParseError, OSError) as exc:
            errors.append(f"{name}: {exc}")
            continue
        if seedval is None or str(int(seedval)) != str(int(seed)):
            errors.append(f"{name}: embedded seed {seedval!r} != filename {seed}")
        if f"parking_occupied_{occ}_seed{seed}" not in (addl or ""):
            errors.append(f"{name}: parking file mismatch for occ={occ}, seed={seed}")
        n = _vehicles_for_cfg(cfg_path)
        if n is None:
            errors.append(f"{name}: route file not found ({route!r})")
        else:
            counts.add(n)
    expected = spec.FLEET_SIZE[spec.DENSITY]
    ok = not errors and counts == {expected}
    detail = (f"25 cfgs, fleet={sorted(counts)} (expected {{{expected}}})"
              if not errors else "; ".join(errors[:3]))
    return Check("nominal cfg content", PASS if ok else FAIL, detail)


def _check_fleet_ladder(sumo_cfg_dir: Path, files: list[str]) -> Check:
    """Verify the whole density ladder: block -> density label -> real fleet
    size, so a shifted index mapping is caught even if one block coincidentally
    matches. ``caos`` is informational (centralised-only, unused by GeoGrid)."""
    observed, mismatches, info_mismatch = {}, [], []
    for density, (start, stop) in spec.DENSITY_ARRAY_RANGE.items():
        block = files[start:stop + 1]
        counts = {_vehicles_for_cfg(sumo_cfg_dir / n) for n in block}
        expected = spec.FLEET_SIZE[density]
        observed[density] = sorted(c for c in counts if c is not None)
        if counts != {expected}:
            (info_mismatch if density in spec.FLEET_SIZE_INFORMATIONAL
             else mismatches).append(f"{density}: {observed[density]} != {expected}")
    detail = ", ".join(f"{d}={observed[d]}" for d in spec.DENSITY_ARRAY_RANGE)
    if mismatches:
        return Check("fleet scale ladder", FAIL, detail + "  [" + "; ".join(mismatches) + "]")
    if info_mismatch:
        return Check("fleet scale ladder", WARN,
                     detail + "  [informational: " + "; ".join(info_mismatch) + "]")
    return Check("fleet scale ladder", PASS, detail)


def _check_sumo_cfg(sumo_cfg_dir: Path) -> list[Check]:
    checks: list[Check] = []
    if not sumo_cfg_dir.is_dir():
        return [Check("sumo_cfg_seeds", FAIL, f"{sumo_cfg_dir} not a directory")]
    files = sorted(p.name for p in sumo_cfg_dir.glob("*.sumo.cfg"))
    checks.append(Check("sumo_cfg_seeds", PASS if len(files) == 100 else WARN,
                        f"{len(files)} scenario files (expected 100)"))

    start, stop = spec.DENSITY_ARRAY_RANGE[spec.DENSITY]
    block = files[start:stop + 1]
    if len(block) != spec.RUNS_PER_CONFIG:
        checks.append(Check("nominal block size", FAIL,
                            f"array {start}-{stop} has {len(block)} files, "
                            f"expected {spec.RUNS_PER_CONFIG}"))
        return checks

    bad_names = [n for n in block if not n.startswith(f"combination_{spec.DENSITY}_")]
    names_ok = not bad_names
    checks.append(Check("nominal array block", PASS if names_ok else FAIL,
                        (f"array {start}-{stop} all '{spec.DENSITY}'") if names_ok
                        else f"unexpected files: {bad_names[:3]}"))

    cells = set()
    parse_ok = True
    for n in block:
        m = _CFG_NAME_RE.match(n)
        if not m or m.group("dens") != spec.DENSITY:
            parse_ok = False
            continue
        cells.add((int(m.group("occ")), int(m.group("seed"))))
    expected = {(o, s) for o in spec.OCCUPANCIES for s in spec.SEEDS}
    checks.append(Check("25-cell mapping", PASS if (parse_ok and cells == expected)
                        else FAIL,
                        f"{len(cells)} unique (occ, seed) cells"
                        if cells == expected else
                        f"cell mismatch (got {len(cells)}, expected 25)"))

    # Content, not just names/counts: embedded seed/occupancy and real fleet.
    checks.append(_check_cfg_content(sumo_cfg_dir, block))
    # Whole-ladder property, so a shifted index mapping cannot pass by accident.
    checks.append(_check_fleet_ladder(sumo_cfg_dir, files))
    return checks


def _check_seed_inputs(mappa: Path) -> list[Check]:
    checks: list[Check] = []
    routes = mappa / "scenari_seed" / "usati"
    parked = mappa / "parked_seed"
    missing_routes = [f"{t}_routes_seed{s}.rou.xml"
                      for t in ("minimo", "normale", "trafficato")
                      for s in spec.SEEDS
                      if not (routes / f"{t}_routes_seed{s}.rou.xml").is_file()]
    checks.append(Check("seed route files", PASS if not missing_routes else FAIL,
                        "all present" if not missing_routes
                        else "missing " + ", ".join(missing_routes[:5])))
    missing_park = [f"parking_occupied_{o}_seed{s}.add.xml"
                    for o in spec.OCCUPANCIES for s in spec.SEEDS
                    if not (parked / f"parking_occupied_{o}_seed{s}.add.xml").is_file()]
    checks.append(Check("seed parking files", PASS if not missing_park else FAIL,
                        "all present" if not missing_park
                        else "missing " + ", ".join(missing_park[:5])))
    return checks


def _check_constraints() -> list[Check]:
    checks: list[Check] = []
    errs = spec.constraint_violations(spec.BASELINE)
    checks.append(Check("baseline valid", PASS if not errs else FAIL,
                        "baseline satisfies all constraints" if not errs
                        else "; ".join(errs)))
    # bounds sanity: lo < hi and baseline inside
    bad = [p.name for p in spec.PRIMARY_PARAMS
           if not (p.lo < p.hi and p.lo <= spec.BASELINE[p.name] <= p.hi)]
    checks.append(Check("bounds contain baseline", PASS if not bad else FAIL,
                        "ok" if not bad else f"bad bounds: {bad}"))
    checks.append(Check("beta complement", PASS, f"beta = {spec.beta_of(spec.BASELINE['alfa'])}"))
    return checks


def run(args: argparse.Namespace) -> list[Check]:
    base = Path(args.base).expanduser()
    mappa = Path(args.mappa).expanduser() if args.mappa else base / "data" / "mappa"
    sumo_cfg_dir = (Path(args.sumo_cfg_dir).expanduser() if args.sumo_cfg_dir
                    else mappa / "sumo_cfg_seeds")
    sif_dir = Path(args.sif_dir).expanduser() if args.sif_dir else base / "sif-offset"

    if args.template:
        template = Path(args.template).expanduser()
    else:
        template = spec.find_template([
            base / "configs_next" / "next_leaderless_k2.toml",
            base / "configs_next" / "next_leaderless.toml",
            base / "data" / "Config.toml",
        ])

    checks: list[Check] = []
    checks += _check_template(template)
    checks += _check_sif(sif_dir, args.sif_suffix)
    checks += _check_sumo_cfg(sumo_cfg_dir)
    checks += _check_seed_inputs(mappa)
    checks += _check_constraints()
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("VANET_BASE", "~/vanet-parking"),
                    help="simulator base dir (default: $VANET_BASE or ~/vanet-parking)")
    ap.add_argument("--mappa", default=None, help="override <base>/data/mappa")
    ap.add_argument("--sumo-cfg-dir", default=None,
                    help="override <mappa>/sumo_cfg_seeds")
    ap.add_argument("--sif-dir", default=None, help="override <base>/sif-offset")
    ap.add_argument("--sif-suffix", default=None,
                    help="leaderless SIF suffix, e.g. gossip (see run_campaign_seeds.slurm)")
    ap.add_argument("--template", default=None,
                    help="leaderless Config.toml to use as run template")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the reports as JSON to this path")
    args = ap.parse_args()

    checks = run(args)
    width = max(len(c.name) for c in checks)
    print("\nPhase-0 preflight")
    print("=" * (width + 60))
    for c in checks:
        print(f"  [{c.status}] {c.name:<{width}}  {c.detail}")
    n_fail = sum(c.status == FAIL for c in checks)
    n_warn = sum(c.status == WARN for c in checks)
    print("=" * (width + 60))
    print(f"  {len(checks)} checks: {n_fail} FAIL, {n_warn} WARN")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(c) for c in checks], indent=2))
        print(f"  report -> {args.json}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
