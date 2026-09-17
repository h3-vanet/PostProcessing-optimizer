#!/usr/bin/env python3
"""Generate LaTeX macros from evidence/caos_crashes.txt.

The file is a concatenation of per-run ``== <path>`` headers, each followed
by the run's stdout lines. One header is one run (duplicate executions of the
same scenario count separately). For each campaign (first path component) the
parser counts runs into three mutually exclusive classes:

    SIGSEGV                     -- the observed failed-to-die signal
    SIGABRT or the "Layer 2 source or destination Id shouldn't be 0" abort
    neither

Macros (letters-only names):

    caos<Campaign>Total / caos<Campaign>Segv / caos<Campaign>Abort / caos<Campaign>Neither
    caosAllTotal / caosAllSegv / caosAllAbort / caosAllNeither

Usage:
    python3 caos_crash_macros.py --evidence DIR --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_tables import NumberRegistry


def campaign_slug(name: str) -> str:
    slug = "".join(part.capitalize() for part in name.split("_") if part)
    slug = "".join(ch for ch in slug if ch.isalpha())
    return slug or "Unknown"


def parse_crashes(path: Path) -> dict:
    runs = []
    current = None
    for line in path.read_text().splitlines():
        if line.startswith("== "):
            if current is not None:
                runs.append(current)
            current = {"header": line[3:].strip(), "body": []}
        elif current is not None and line.strip():
            current["body"].append(line)
    if current is not None:
        runs.append(current)

    per_campaign: dict = {}
    for run in runs:
        campaign = run["header"].split("/")[0]
        body = "\n".join(run["body"])
        segv = "SIGSEGV" in body
        abort = "SIGABRT" in body or "shouldn't be 0" in body
        rec = per_campaign.setdefault(
            campaign, {"total": 0, "segv": 0, "abort": 0, "neither": 0}
        )
        rec["total"] += 1
        if segv:
            rec["segv"] += 1
        if abort:
            rec["abort"] += 1
        if not segv and not abort:
            rec["neither"] += 1
    return per_campaign


def build_macros(per_campaign: dict, numbers: NumberRegistry) -> None:
    totals = {"total": 0, "segv": 0, "abort": 0, "neither": 0}
    for campaign in sorted(per_campaign):
        rec = per_campaign[campaign]
        slug = campaign_slug(campaign)
        numbers.add(f"caos{slug}Total", str(rec["total"]))
        numbers.add(f"caos{slug}Segv", str(rec["segv"]))
        numbers.add(f"caos{slug}Abort", str(rec["abort"]))
        numbers.add(f"caos{slug}Neither", str(rec["neither"]))
        for key in totals:
            totals[key] += rec[key]
    numbers.add("caosAllTotal", str(totals["total"]))
    numbers.add("caosAllSegv", str(totals["segv"]))
    numbers.add("caosAllAbort", str(totals["abort"]))
    numbers.add("caosAllNeither", str(totals["neither"]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    per_campaign = parse_crashes(args.evidence / "caos_crashes.txt")
    numbers = NumberRegistry()
    build_macros(per_campaign, numbers)
    args.out.mkdir(parents=True, exist_ok=True)
    numbers.write(args.out / "caos_numbers.tex")
    for campaign in sorted(per_campaign):
        rec = per_campaign[campaign]
        print(
            f"{campaign}: total={rec['total']} segv={rec['segv']} "
            f"abort={rec['abort']} neither={rec['neither']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
