#!/usr/bin/env python3
"""Generate LaTeX macros from evidence/probe_gossip_rounds.jsonl.

The probe records one "gossip round published" event per round per vehicle,
each carrying the `gps_tick` (position-update tick) at which it fired. With
simulated-time timers, a 500 ms gossip period over a 0.1 s position-update
tick must land exactly 5 ticks after the previous round of the same vehicle.

For each vehicle, consecutive published rounds are sorted by `gps_tick` and
the adjacent differences are collected. The denominator is *all* adjacent
intervals (not only the ones equal to 5). Reported:

    gossipProbeVehicles / gossipProbeRounds / gossipProbeIntervals
    gossipProbeExactFiveCount / gossipProbeExactFiveShare
    gossipProbeMultipleFiveShare

Usage:
    python3 gossip_probe_macros.py --evidence DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_tables import NumberRegistry


def parse_probe(path: Path) -> dict:
    vehicles: dict = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        fields = obj.get("fields", {})
        if fields.get("message") != "gossip round published":
            continue
        vehicles.setdefault(fields["vehicle_id"], []).append(fields["gps_tick"])

    intervals = []
    for ticks in vehicles.values():
        ticks.sort()
        intervals.extend(b - a for a, b in zip(ticks, ticks[1:]))

    n = len(intervals)
    exact = sum(1 for d in intervals if d == 5)
    multiple = sum(1 for d in intervals if d % 5 == 0)
    return {
        "vehicles": len(vehicles),
        "rounds": sum(len(t) for t in vehicles.values()),
        "intervals": n,
        "exact_five": exact,
        "multiple_five": multiple,
        "exact_five_share": (100.0 * exact / n) if n else 0.0,
        "multiple_five_share": (100.0 * multiple / n) if n else 0.0,
    }


def build_macros(stats: dict, numbers: NumberRegistry) -> None:
    numbers.add("gossipProbeVehicles", str(stats["vehicles"]))
    numbers.add("gossipProbeRounds", str(stats["rounds"]))
    numbers.add("gossipProbeIntervals", str(stats["intervals"]))
    numbers.add("gossipProbeExactFiveCount", str(stats["exact_five"]))
    numbers.add("gossipProbeExactFiveShare", f"{stats['exact_five_share']:.2f}")
    numbers.add("gossipProbeMultipleFiveShare", f"{stats['multiple_five_share']:.2f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    stats = parse_probe(args.evidence / "probe_gossip_rounds.jsonl")
    numbers = NumberRegistry()
    build_macros(stats, numbers)
    args.out.mkdir(parents=True, exist_ok=True)
    numbers.write(args.out / "gossip_numbers.tex")
    print(
        f"gossip probe: {stats['exact_five']}/{stats['intervals']} exact-5 "
        f"({stats['exact_five_share']:.2f}%), {stats['multiple_five']}/"
        f"{stats['intervals']} multiple-5 ({stats['multiple_five_share']:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
