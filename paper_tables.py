#!/usr/bin/env python3
"""Regenerate GeoGrid paper LaTeX tables and prose-number macros from
post-processed campaign metrics and run configs.

Standalone script: reads summary_all_experiments.csv (one per series) and
config_used.toml (one per run, under campaign trees) and writes:

    out/tables/<nn>_<name>.tex   one booktabs table per question
    out/numbers.tex              \\newcommand macros for every quoted number
    out/check.txt                run counts, config consistency, skips

Usage:
    python3 paper_tables.py --metrics DIR [DIR ...] --configs DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

EXPECT_TOLERANCE = 0.05

VALID_NR_SIM_T_REACHED = 179

DENSITY_LABELS = {
    "minimo": "sparse",
    "normale": "nominal",
    "trafficato": "congested",
    "caos": "caos",
}
DENSITY_ORDER = ["sparse", "nominal", "congested", "caos"]

# series name -> (arm label, k, display)
SERIES_ARM = {
    "post_simtime_ll": ("GeoGrid k=1", 1),
    "post_simtime_ll_k2": ("GeoGrid k=2", 2),
    "post_simtime_br": ("Broker", None),
    "post_ll": ("GeoGrid k=1", 1),
    "post_br": ("Broker", None),
    "rsu_simtime_9": ("Broker", None),
    "rsu_simtime_16": ("Broker", None),
    "rsu_simtime_25": ("Broker", None),
    "mcs5_simtime_br": ("Broker", None),
    "mcs5_simtime_ll_k2": ("GeoGrid k=2", 2),
}

# series name -> config tree name
SERIES_CONFIG_TREE = {
    "post_simtime_ll": "campaign_simtime_ll",
    "post_simtime_ll_k2": "campaign_simtime_ll_k2",
    "post_simtime_br": "campaign_simtime_br",
    "rsu_simtime_9": "rsu_simtime_9",
    "rsu_simtime_16": "rsu_simtime_16",
    "rsu_simtime_25": "rsu_simtime_25",
    "mcs5_simtime_br": "mcs5_simtime_br",
    "mcs5_simtime_ll_k2": "mcs5_simtime_ll_k2",
}

# Broker-arm series: their metrics directory name gets --broker-suffix appended.
BROKER_SERIES = {"post_simtime_br", "rsu_simtime_9", "rsu_simtime_16", "rsu_simtime_25", "mcs5_simtime_br"}

# Parameters to report in the parameters table: (dotted key, display name, only-for-arms)
PARAMETER_KEYS = [
    ("gossip.gossip_interval_ms", "gossip\\_interval\\_ms", None),
    ("gossip.neighbor_k", "neighbor\\_k", None),
    ("h3.cluster_resolution", "cluster\\_resolution", None),
    ("h3.spot_resolution", "spot\\_resolution", None),
    ("crdt.slot_ttl_secs", "slot\\_ttl\\_secs", None),
    ("assignment.backoff.max_distance_m", "max\\_distance\\_m", None),
    ("coordinator.backend", "backend", None),
]

# Keys the core actually uses but may be absent from config_used.toml (the
# core falls back to these defaults internally): (dotted key, display name,
# default value, only-for-arms).
KNOWN_KEYS = [
    ("gossip.sim_tick_secs", "sim\\_tick\\_secs", 0.1, None),
    ("broker.rtt_ms", "rtt\\_ms", 0, {"Broker"}),
    ("broker.claim_ttl_secs", "claim\\_ttl\\_secs", 120, {"Broker"}),
    ("broker.state_log_interval_secs", "state\\_log\\_interval\\_secs", 30, {"Broker"}),
]

# TOML keys the core silently ignores (not KNOWN_KEYS, never shown in a
# table, reported to check.txt only).
IGNORED_BY_CORE_KEYS = {
    "broker.claim_ttl_sim_s",
    "broker.lease_sim_s",
    "broker.uplink_timeout_sim_s",
    "broker.uplink_max_retries",
    "broker.uplink_backoff_multiplier",
}


def effective_value(cfg: dict, key: str, default):
    """(value, used_default) for a KNOWN_KEYS entry given a flat config dict."""
    if key in cfg:
        return cfg[key], False
    return default, True


def broker_rtt_note(configs: dict) -> str:
    """Caption fragment naming the effective broker rtt_ms, for any table
    that includes the Broker arm."""
    cfg = configs.get("campaign_simtime_br", {})
    key, _, default, _ = next(k for k in KNOWN_KEYS if k[0] == "broker.rtt_ms")
    value, used_default = effective_value(cfg, key, default)
    suffix = " (default)" if used_default else ""
    return f"broker rtt\\_ms = {value}{suffix}"


# --------------------------------------------------------------------------- #
# Reporting / number-collection infrastructure
# --------------------------------------------------------------------------- #


class ConfigMismatchError(Exception):
    """Raised when a parameter differs across runs of the same config tree."""


@dataclass
class Report:
    series_runs: dict = field(default_factory=dict)  # series -> total rows
    series_valid: dict = field(default_factory=dict)  # series -> valid rows
    series_invalid_by_density: dict = field(default_factory=dict)  # series -> {density: n}
    warnings: list = field(default_factory=list)
    skipped_tables: list = field(default_factory=list)  # (name, reason)
    config_notes: list = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def skip_table(self, name: str, reason: str) -> None:
        self.skipped_tables.append((name, reason))

    def note_series(self, series: str, df: pd.DataFrame) -> None:
        self.series_runs[series] = len(df)
        valid = df[df["valid"]]
        self.series_valid[series] = len(valid)
        invalid = df[~df["valid"]]
        counts = invalid.groupby("density").size().to_dict()
        self.series_invalid_by_density[series] = counts

    def write(self, path: Path) -> None:
        lines = []
        lines.append("=== Runs per series ===")
        for series in sorted(self.series_runs):
            total = self.series_runs[series]
            valid = self.series_valid[series]
            lines.append(f"{series}: {valid}/{total} valid")
            invalid = self.series_invalid_by_density.get(series, {})
            for density in DENSITY_ORDER:
                if density in invalid:
                    lines.append(f"    invalid[{density}] = {invalid[density]}")
        lines.append("")
        lines.append("=== Config consistency ===")
        if self.config_notes:
            lines.extend(self.config_notes)
        else:
            lines.append("(no config trees checked)")
        lines.append("")
        lines.append("=== Skipped tables ===")
        if self.skipped_tables:
            for name, reason in self.skipped_tables:
                lines.append(f"{name}: {reason}")
        else:
            lines.append("(none)")
        lines.append("")
        lines.append("=== Warnings ===")
        if self.warnings:
            lines.extend(self.warnings)
        else:
            lines.append("(none)")
        path.write_text("\n".join(lines) + "\n")


class NumberRegistry:
    def __init__(self) -> None:
        self._numbers: dict[str, str] = {}

    def add(self, macro_name: str, value: str) -> None:
        if not macro_name.isalpha():
            raise ValueError(f"macro name must be letters only: {macro_name!r}")
        if macro_name in self._numbers and self._numbers[macro_name] != value:
            raise ValueError(f"duplicate macro name with different value: {macro_name!r}")
        self._numbers[macro_name] = value

    def write(self, path: Path) -> None:
        lines = [
            f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(self._numbers.items())
        ]
        path.write_text("\n".join(lines) + "\n")

    def as_dict(self) -> dict[str, str]:
        return dict(self._numbers)


@dataclass
class TableResult:
    name: str
    latex: str


def fmt_mean_sd(values) -> tuple[float, float, int]:
    values = [v for v in values if v is not None and not pd.isna(v)]
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return (mean, sd, n)


def render_latex(header, rows, caption: str, label: str) -> str:
    ncols = len(header)
    assert ncols <= 6, f"table {label} has {ncols} columns, max is 6"
    colspec = "l" * ncols
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


_ARM_MACRO_NAMES = {
    "GeoGrid k=1": "GeoGridKOne",
    "GeoGrid k=2": "GeoGridKTwo",
    "Broker": "Broker",
}


def slug(text: str) -> str:
    """Letters-only token for macro names. Arm labels are mapped explicitly
    so 'GeoGrid k=1' and 'GeoGrid k=2' don't collide once digits are dropped."""
    if text in _ARM_MACRO_NAMES:
        return _ARM_MACRO_NAMES[text]
    return "".join(ch for ch in text if ch.isalpha())


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def discover_series(
    metrics_dirs: list[Path], report: Report, broker_suffix: str = ""
) -> dict[str, pd.DataFrame]:
    series = {}
    for name in list(SERIES_ARM.keys()):
        dirname = name + broker_suffix if name in BROKER_SERIES else name
        found = None
        for metrics_dir in metrics_dirs:
            candidate = metrics_dir / dirname / "summary_all_experiments.csv"
            if candidate.exists():
                found = candidate
                break
        if found is None:
            where = f"'{dirname}'" if dirname != name else f"'{name}'"
            report.warn(f"series '{name}' (looked for {where}) not found in any --metrics dir; its tables are skipped")
            continue
        df = pd.read_csv(found)
        df["density"] = df["traffic"].map(DENSITY_LABELS)
        if df["density"].isna().any():
            unknown = sorted(set(df.loc[df["density"].isna(), "traffic"]))
            report.warn(f"series '{name}': unknown traffic values {unknown}, dropping those rows")
            df = df.dropna(subset=["density"])
        df["valid"] = df["nr_sim_t_reached"] >= VALID_NR_SIM_T_REACHED
        report.note_series(name, df)
        series[name] = df
    return series


def _flatten_toml(d: dict, prefix: str = "") -> dict:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_toml(v, key))
        else:
            flat[key] = v
    return flat


def load_configs(configs_dir: Path, report: Report) -> dict[str, dict]:
    trees = {}
    seen_tree_names = set(SERIES_CONFIG_TREE.values())
    for tree_name in sorted(seen_tree_names):
        tree_dir = configs_dir / tree_name
        if not tree_dir.exists():
            report.warn(f"config tree '{tree_name}' not found under {configs_dir}")
            continue
        toml_files = sorted(tree_dir.glob("**/config_used.toml"))
        if not toml_files:
            report.warn(f"config tree '{tree_name}' has no config_used.toml files")
            continue
        flat_configs = []
        for f in toml_files:
            with f.open("rb") as fh:
                raw = tomllib.load(fh)
            flat_configs.append((f, _flatten_toml(raw)))

        merged: dict = {}
        for f, flat in flat_configs:
            for key, value in flat.items():
                if key in merged:
                    prev_file, prev_value = merged[key]
                    if prev_value != value:
                        raise ConfigMismatchError(
                            f"tree '{tree_name}': key '{key}' differs "
                            f"({prev_value!r} in {prev_file}, {value!r} in {f})"
                        )
                else:
                    merged[key] = (f, value)

        # Consistency check on EFFECTIVE values (TOML value if present, else
        # the KNOWN_KEYS default) — catches e.g. one run stating a value
        # explicitly while another silently relies on a differing default.
        for key, display, default, _only_arms in KNOWN_KEYS:
            effective_values = {
                effective_value(flat, key, default)[0] for _f, flat in flat_configs
            }
            if len(effective_values) > 1:
                raise ConfigMismatchError(
                    f"tree '{tree_name}': effective key '{key}' differs across runs: "
                    f"{sorted(effective_values)}"
                )

        ignored_present = sorted(
            {key for _f, flat in flat_configs for key in flat if key in IGNORED_BY_CORE_KEYS}
        )
        if ignored_present:
            report.config_notes.append(
                f"{tree_name}: ignored by core (not used by the running system): "
                + ", ".join(ignored_present)
            )

        trees[tree_name] = {k: v for k, (_, v) in merged.items()}
        report.config_notes.append(
            f"{tree_name}: {len(toml_files)} config_used.toml file(s), all keys consistent"
        )
    return trees


# --------------------------------------------------------------------------- #
# Table builders
# --------------------------------------------------------------------------- #


def build_parameters(configs: dict, numbers: NumberRegistry, report: Report):
    rows = []
    series_for_arm = {
        "GeoGrid k=1": "campaign_simtime_ll",
        "GeoGrid k=2": "campaign_simtime_ll_k2",
        "Broker": "campaign_simtime_br",
    }
    for arm, tree_name in series_for_arm.items():
        cfg = configs.get(tree_name)
        if cfg is None:
            report.skip_table(f"parameters[{arm}]", f"config tree '{tree_name}' unavailable")
            continue
        for key, display, only_arms in PARAMETER_KEYS:
            if only_arms is not None and arm not in only_arms:
                continue
            if key not in cfg:
                continue
            rows.append((arm, display, cfg[key]))
        for key, display, default, only_arms in KNOWN_KEYS:
            if only_arms is not None and arm not in only_arms:
                continue
            value, used_default = effective_value(cfg, key, default)
            suffix = " (default)" if used_default else ""
            rows.append((arm, display, f"{value}{suffix}"))
    if not rows:
        report.skip_table("parameters", "no config trees available")
        return None
    header = ["Arm", "Parameter", "Value"]
    caption = "Configuration parameters used for each arm (source: config\\_used.toml)."
    latex = render_latex(header, rows, caption, "tab:parameters")
    return TableResult("parameters", latex)


def _agg_park_rate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    valid = df[df["valid"]]
    return valid.groupby(group_cols)["C1_park_rate"].apply(list).reset_index()


def build_park_rate_density(series: dict, configs: dict, numbers: NumberRegistry, report: Report):
    arms = [
        ("GeoGrid k=1", "post_simtime_ll"),
        ("GeoGrid k=2", "post_simtime_ll_k2"),
        ("Broker", "post_simtime_br"),
    ]
    available = [(arm, s) for arm, s in arms if s in series]
    if not available:
        report.skip_table("park_rate_density", "no series available")
        return None
    header = ["Density"] + [arm for arm, _ in available]
    rows = []
    for density in DENSITY_ORDER:
        row = [density]
        for arm, s in available:
            df = series[s]
            values = df[(df["valid"]) & (df["density"] == density)]["C1_park_rate"].tolist()
            mean, sd, n = fmt_mean_sd(values)
            if n == 0:
                row.append("--")
                continue
            row.append(f"{mean:.1f}$\\pm${sd:.1f} (n={n})")
            macro = "parkRate" + slug(density) + slug(arm)
            numbers.add(macro, f"{mean:.1f}")
        rows.append(row)
    caption = (
        "Park rate (\\%) by density for each arm, mean $\\pm$ SD (n runs); "
        "SD is mixed occupancy+seed within each density."
    )
    if any(arm == "Broker" for arm, _ in available):
        caption += f" {broker_rtt_note(configs)}."
    latex = render_latex(header, rows, caption, "tab:park_rate_density")
    return TableResult("park_rate_density", latex)


def build_gap_to_broker(series: dict, configs: dict, numbers: NumberRegistry, report: Report):
    if "post_simtime_br" not in series:
        report.skip_table("gap_to_broker", "post_simtime_br unavailable")
        return None
    if "post_simtime_ll" not in series and "post_simtime_ll_k2" not in series:
        report.skip_table("gap_to_broker", "no GeoGrid series available")
        return None
    header = ["Density", "Gap k=1 (pts)", "Gap k=2 (pts)", "Share closed (\\%)"]
    rows = []
    br = series["post_simtime_br"]
    for density in DENSITY_ORDER:
        broker_vals = br[(br["valid"]) & (br["density"] == density)]["C1_park_rate"].tolist()
        broker_mean, _, broker_n = fmt_mean_sd(broker_vals)
        row = [density]
        gap_k1 = gap_k2 = None
        if broker_n == 0:
            rows.append([density, "--", "--", "--"])
            continue
        for label, s_name in (("k1", "post_simtime_ll"), ("k2", "post_simtime_ll_k2")):
            if s_name not in series:
                row.append("--")
                continue
            df = series[s_name]
            vals = df[(df["valid"]) & (df["density"] == density)]["C1_park_rate"].tolist()
            mean, _, n = fmt_mean_sd(vals)
            if n == 0:
                row.append("--")
                continue
            gap = broker_mean - mean
            if label == "k1":
                gap_k1 = gap
            else:
                gap_k2 = gap
            row.append(f"{gap:.1f}")
            label_macro = "KOne" if label == "k1" else "KTwo"
            numbers.add("gap" + slug(density) + label_macro, f"{gap:.1f}")
        if gap_k1 is not None and gap_k2 is not None and gap_k1 != 0:
            share = (gap_k1 - gap_k2) / gap_k1 * 100
            row.append(f"{share:.1f}")
            numbers.add("shareClosed" + slug(density), f"{share:.1f}")
        else:
            row.append("--")
        rows.append(row)
    caption = (
        "Gap to Broker in percentage points by density for GeoGrid k=1 and k=2, "
        "and the share of the k=1 gap closed by k=2. SD omitted (derived quantity). "
        f"{broker_rtt_note(configs)}."
    )
    latex = render_latex(header, rows, caption, "tab:gap_to_broker")
    return TableResult("gap_to_broker", latex)


def build_occupancy_drop(series: dict, configs: dict, numbers: NumberRegistry, report: Report):
    arms = [
        ("GeoGrid k=1", "post_simtime_ll"),
        ("GeoGrid k=2", "post_simtime_ll_k2"),
        ("Broker", "post_simtime_br"),
    ]
    available = [(arm, s) for arm, s in arms if s in series]
    if not available:
        report.skip_table("occupancy_drop", "no series available")
        return None
    header = ["Arm", "Density", "Drop 30->95 (pts)"]
    rows = []
    for arm, s in available:
        df = series[s]
        for density in DENSITY_ORDER:
            sub = df[(df["valid"]) & (df["density"] == density)]
            v30 = sub[sub["occupancy"] == 30]["C1_park_rate"].tolist()
            v95 = sub[sub["occupancy"] == 95]["C1_park_rate"].tolist()
            m30, _, n30 = fmt_mean_sd(v30)
            m95, _, n95 = fmt_mean_sd(v95)
            if n30 == 0 or n95 == 0:
                rows.append([arm, density, "--"])
                continue
            drop = m30 - m95
            rows.append([arm, density, f"{drop:.1f}"])
            numbers.add("occDrop" + slug(density) + slug(arm), f"{drop:.1f}")
    caption = (
        "Park-rate drop (percentage points) from 30\\% to 95\\% occupancy, "
        "per arm per density. n=5 seeds per (density, occupancy) cell; SD is "
        "seed-only for each endpoint (30\\% and 95\\%)."
    )
    if any(arm == "Broker" for arm, _ in available):
        caption += f" {broker_rtt_note(configs)}."
    latex = render_latex(header, rows, caption, "tab:occupancy_drop")
    return TableResult("occupancy_drop", latex)


def build_worst_cells(series: dict, configs: dict, numbers: NumberRegistry, report: Report):
    arms = [
        ("GeoGrid k=1", "post_simtime_ll"),
        ("GeoGrid k=2", "post_simtime_ll_k2"),
        ("Broker", "post_simtime_br"),
    ]
    available = [(arm, s) for arm, s in arms if s in series]
    if not available:
        report.skip_table("worst_cells", "no series available")
        return None
    header = ["Arm", "Density", "Occupancy", "Park rate (\\%)"]
    rows = []
    for arm, s in available:
        df = series[s]
        valid = df[df["valid"]]
        if valid.empty:
            rows.append([arm, "--", "--", "--"])
            continue
        cell_means = (
            valid.groupby(["density", "occupancy"])["C1_park_rate"].mean().reset_index()
        )
        worst = cell_means.loc[cell_means["C1_park_rate"].idxmin()]
        rows.append(
            [arm, worst["density"], int(worst["occupancy"]), f"{worst['C1_park_rate']:.1f}"]
        )
        numbers.add("worst" + slug(arm), f"{worst['C1_park_rate']:.1f}")
    caption = "Minimum mean park rate per arm and the (density, occupancy) cell where it occurs."
    if any(arm == "Broker" for arm, _ in available):
        caption += f" {broker_rtt_note(configs)}."
    latex = render_latex(header, rows, caption, "tab:worst_cells")
    return TableResult("worst_cells", latex)


def build_winners_parked(series: dict, numbers: NumberRegistry, report: Report):
    arms = [("GeoGrid k=1", "post_simtime_ll"), ("GeoGrid k=2", "post_simtime_ll_k2")]
    available = [(arm, s) for arm, s in arms if s in series]
    if not available:
        report.skip_table("winners_parked", "no GeoGrid series available")
        return None
    required = {"A2_vehicles_with_slot", "A2_assigned_uniq", "C1_parked"}
    for arm, s in available:
        missing = required - set(series[s].columns)
        if missing:
            report.skip_table("winners_parked", f"series '{s}' missing columns {sorted(missing)}")
            return None
    header = ["Arm", "Density", "Winners", "Claims", "Parked", "Claims/winner"]
    rows = []
    for arm, s in available:
        df = series[s]
        for density in DENSITY_ORDER:
            sub = df[(df["valid"]) & (df["density"] == density)]
            if sub.empty:
                rows.append([arm, density, "--", "--", "--", "--"])
                continue
            winners, _, _ = fmt_mean_sd(sub["A2_vehicles_with_slot"].tolist())
            claims, _, _ = fmt_mean_sd(sub["A2_assigned_uniq"].tolist())
            parked, _, _ = fmt_mean_sd(sub["C1_parked"].tolist())
            ratio = claims / winners if winners else float("nan")
            rows.append(
                [arm, density, f"{winners:.1f}", f"{claims:.1f}", f"{parked:.1f}", f"{ratio:.2f}"]
            )
    caption = (
        "Winners (A2\\_vehicles\\_with\\_slot) and claims (A2\\_assigned\\_uniq), vehicles "
        "parked (C1\\_parked), and claims per winner (ratio of means: "
        "mean(A2\\_assigned\\_uniq) / mean(A2\\_vehicles\\_with\\_slot)), by density, "
        "GeoGrid arms only. Means over valid runs."
    )
    latex = render_latex(header, rows, caption, "tab:winners_parked")
    return TableResult("winners_parked", latex)


def build_channel(series: dict, configs: dict, numbers: NumberRegistry, report: Report):
    arms = [
        ("GeoGrid k=1", "post_simtime_ll"),
        ("GeoGrid k=2", "post_simtime_ll_k2"),
        ("Broker", "post_simtime_br"),
    ]
    available = [(arm, s) for arm, s in arms if s in series]
    if not available:
        report.skip_table("channel", "no series available")
        return None
    header = ["Arm", "Density", "PLR (\\%)", "CP tx bytes/vehicle"]
    rows = []
    for arm, s in available:
        df = series[s]
        for density in DENSITY_ORDER:
            sub = df[(df["valid"]) & (df["density"] == density)]
            if sub.empty:
                rows.append([arm, density, "--", "--"])
                continue
            plr, _, _ = fmt_mean_sd(sub["nr_plr_pct"].tolist())
            tx, _, _ = fmt_mean_sd(sub["D1_cp_tx_bytes_per_vehicle_mean"].tolist())
            rows.append([arm, density, f"{plr:.2f}", f"{tx:.0f}"])
    caption = "Packet loss ratio and control-plane tx bytes per vehicle, by density per arm."
    if any(arm == "Broker" for arm, _ in available):
        caption += f" {broker_rtt_note(configs)}."
    latex = render_latex(header, rows, caption, "tab:channel")
    return TableResult("channel", latex)


def build_rsu_sensitivity(series: dict, configs: dict, numbers: NumberRegistry, report: Report):
    counts = [(4, "post_simtime_br"), (9, "rsu_simtime_9"), (16, "rsu_simtime_16"), (25, "rsu_simtime_25")]
    header = ["RSU count", "Park rate (\\%)", "E1 (\\%)", "PLR (\\%)"]
    rows = []
    any_present = False
    for count, s in counts:
        if s not in series:
            report.skip_table(f"rsu_sensitivity[{count}]", f"series '{s}' not yet available")
            continue
        df = series[s]
        sub = df[(df["valid"]) & (df["density"] == "nominal")]
        if sub.empty:
            rows.append([count, "--", "--", "--"])
            continue
        any_present = True
        park, _, _ = fmt_mean_sd(sub["C1_park_rate"].tolist())
        e1, _, _ = fmt_mean_sd(sub["E1_rsu_coverage_pct"].tolist())
        plr, _, _ = fmt_mean_sd(sub["nr_plr_pct"].tolist())
        rows.append([count, f"{park:.1f}", f"{e1:.1f}", f"{plr:.2f}"])
        count_word = {4: "Four", 9: "Nine", 16: "Sixteen", 25: "TwentyFive"}[count]
        numbers.add(f"rsu{count_word}ParkRate", f"{park:.1f}")
    if not any_present:
        report.skip_table("rsu_sensitivity", "no RSU-count series available")
        return None
    caption = (
        "RSU count sensitivity at nominal density (Broker arm): park rate, RSU "
        "coverage (E1), and PLR. RSU count 9/16/25 rows are omitted when their "
        f"campaign is not yet available. {broker_rtt_note(configs)}."
    )
    latex = render_latex(header, rows, caption, "tab:rsu_sensitivity")
    return TableResult("rsu_sensitivity", latex)


def build_mcs(series: dict, configs: dict, numbers: NumberRegistry, report: Report):
    combos = [
        ("Broker", "MCS14", "post_simtime_br"),
        ("Broker", "MCS5", "mcs5_simtime_br"),
        ("GeoGrid k=2", "MCS14", "post_simtime_ll_k2"),
        ("GeoGrid k=2", "MCS5", "mcs5_simtime_ll_k2"),
    ]
    header = ["Arm", "MCS", "Park rate (\\%)", "PLR (\\%)", "E1 (\\%)"]
    rows = []
    any_present = False
    for arm, mcs, s in combos:
        if s not in series:
            report.skip_table(f"mcs[{arm},{mcs}]", f"series '{s}' not yet available")
            continue
        df = series[s]
        sub = df[(df["valid"]) & (df["density"] == "nominal")]
        if sub.empty:
            rows.append([arm, mcs, "--", "--", "--"])
            continue
        any_present = True
        park, _, _ = fmt_mean_sd(sub["C1_park_rate"].tolist())
        plr, _, _ = fmt_mean_sd(sub["nr_plr_pct"].tolist())
        if "E1_rsu_coverage_pct" in sub.columns:
            e1, _, n_e1 = fmt_mean_sd(sub["E1_rsu_coverage_pct"].tolist())
            e1_str = f"{e1:.1f}" if n_e1 else "--"
        else:
            e1_str = "--"
        rows.append([arm, mcs, f"{park:.1f}", f"{plr:.2f}", e1_str])
    if not any_present:
        report.skip_table("mcs", "no MCS series available")
        return None
    caption = (
        "MCS sensitivity at nominal density: MCS14 (baseline) vs MCS5, broker and "
        "GeoGrid k=2. MCS5 rows are omitted when that campaign is not yet available. "
        f"{broker_rtt_note(configs)}."
    )
    latex = render_latex(header, rows, caption, "tab:mcs")
    return TableResult("mcs", latex)


def build_timer_fix_robustness(series: dict, configs: dict, numbers: NumberRegistry, report: Report):
    pairs = [("Broker", "post_br", "post_simtime_br"), ("GeoGrid k=1", "post_ll", "post_simtime_ll")]
    available = [(arm, pre, post) for arm, pre, post in pairs if pre in series and post in series]
    if not available:
        report.skip_table("timer_fix_robustness", "pre-fix or sim-time series unavailable")
        return None
    header = ["Arm", "Density", "Pre-fix (\\%)", "Sim-time (\\%)", "Diff (pts)"]
    rows = []
    for arm, pre_s, post_s in available:
        pre_df, post_df = series[pre_s], series[post_s]
        for density in DENSITY_ORDER:
            pre_vals = pre_df[(pre_df["valid"]) & (pre_df["density"] == density)]["C1_park_rate"].tolist()
            post_vals = post_df[(post_df["valid"]) & (post_df["density"] == density)]["C1_park_rate"].tolist()
            pre_mean, _, pre_n = fmt_mean_sd(pre_vals)
            post_mean, _, post_n = fmt_mean_sd(post_vals)
            if pre_n == 0 or post_n == 0:
                rows.append([arm, density, "--", "--", "--"])
                continue
            diff = post_mean - pre_mean
            rows.append([arm, density, f"{pre_mean:.1f}", f"{post_mean:.1f}", f"{diff:.1f}"])
            numbers.add("robust" + slug(density) + slug(arm), f"{diff:.1f}")
    caption = (
        "Pre-fix vs sim-time park rate by density, broker and GeoGrid k=1. "
        "Diff = sim-time $-$ pre-fix, in percentage points."
    )
    if any(arm == "Broker" for arm, _, _ in available):
        caption += f" {broker_rtt_note(configs)}."
    latex = render_latex(header, rows, caption, "tab:timer_fix_robustness")
    return TableResult("timer_fix_robustness", latex)


def build_run_validity(series: dict, numbers: NumberRegistry, report: Report):
    if not series:
        report.skip_table("run_validity", "no series available")
        return None
    header = ["Series", "Total", "Valid", "Invalid (by density)"]
    rows = []
    for name in sorted(series):
        df = series[name]
        total = len(df)
        valid = int(df["valid"].sum())
        invalid = df[~df["valid"]]
        counts = invalid.groupby("density").size().to_dict()
        counts_str = ", ".join(f"{d}={counts[d]}" for d in DENSITY_ORDER if d in counts) or "--"
        rows.append([name.replace("_", "\\_"), total, valid, counts_str])
    caption = (
        "Runs per series, valid runs (nr\\_sim\\_t\\_reached $\\geq$ 179), and invalid "
        "runs broken down by density."
    )
    latex = render_latex(header, rows, caption, "tab:run_validity")
    return TableResult("run_validity", latex)


TABLE_BUILDERS = [
    ("01_parameters", lambda series, configs, numbers, report: build_parameters(configs, numbers, report)),
    ("02_park_rate_density", lambda series, configs, numbers, report: build_park_rate_density(series, configs, numbers, report)),
    ("03_gap_to_broker", lambda series, configs, numbers, report: build_gap_to_broker(series, configs, numbers, report)),
    ("04_occupancy_drop", lambda series, configs, numbers, report: build_occupancy_drop(series, configs, numbers, report)),
    ("05_worst_cells", lambda series, configs, numbers, report: build_worst_cells(series, configs, numbers, report)),
    ("06_winners_parked", lambda series, configs, numbers, report: build_winners_parked(series, numbers, report)),
    ("07_channel", lambda series, configs, numbers, report: build_channel(series, configs, numbers, report)),
    ("08_rsu_sensitivity", lambda series, configs, numbers, report: build_rsu_sensitivity(series, configs, numbers, report)),
    ("09_mcs", lambda series, configs, numbers, report: build_mcs(series, configs, numbers, report)),
    ("10_timer_fix_robustness", lambda series, configs, numbers, report: build_timer_fix_robustness(series, configs, numbers, report)),
    ("11_run_validity", lambda series, configs, numbers, report: build_run_validity(series, numbers, report)),
]


def run(
    metrics_dirs: list[Path], configs_dir: Path, out_dir: Path, broker_suffix: str = ""
) -> tuple[Report, NumberRegistry]:
    report = Report()
    numbers = NumberRegistry()

    series = discover_series(metrics_dirs, report, broker_suffix=broker_suffix)
    configs = load_configs(configs_dir, report)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for out_name, builder in TABLE_BUILDERS:
        result = builder(series, configs, numbers, report)
        if result is None:
            continue
        (tables_dir / f"{out_name}.tex").write_text(result.latex)

    numbers.write(out_dir / "numbers.tex")
    report.write(out_dir / "check.txt")
    return report, numbers


def check_expectations(numbers: NumberRegistry, expected: dict[str, float]) -> list[str]:
    """Return a list of human-readable mismatch descriptions (empty if all match)."""
    actual_numbers = numbers.as_dict()
    mismatches = []
    for macro, expected_value in expected.items():
        if macro not in actual_numbers:
            mismatches.append(f"{macro}: expected {expected_value}, but macro was not produced")
            continue
        actual = float(actual_numbers[macro])
        if abs(actual - float(expected_value)) > EXPECT_TOLERANCE:
            mismatches.append(f"{macro}: expected {expected_value}, got {actual}")
    return mismatches


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True, type=Path, help="metrics dirs")
    parser.add_argument("--configs", required=True, type=Path, help="configs dir")
    parser.add_argument("--out", required=True, type=Path, help="output dir")
    parser.add_argument(
        "--broker-suffix",
        default="",
        help="suffix appended to broker series directory names (post_simtime_br, "
        "rsu_simtime_{9,16,25}, mcs5_simtime_br) when looking them up under --metrics",
    )
    parser.add_argument(
        "--expect",
        type=Path,
        default=None,
        help="JSON file of {macro_name: expected_value}; exits non-zero on any "
        f"mismatch beyond tolerance {EXPECT_TOLERANCE}",
    )
    args = parser.parse_args(argv)

    _report, numbers = run(args.metrics, args.configs, args.out, broker_suffix=args.broker_suffix)

    if args.expect is not None:
        expected = json.loads(args.expect.read_text())
        mismatches = check_expectations(numbers, expected)
        if mismatches:
            for m in mismatches:
                print(f"MISMATCH: {m}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
