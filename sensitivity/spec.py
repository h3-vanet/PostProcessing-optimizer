#!/usr/bin/env python3
"""Shared specification for the GeoGrid condition-specific sensitivity study.

This module is the single source of truth for:

* the parameters that may be varied (5-D primary space + N_ae ancillary),
* their bounds, types and snapping rules,
* the paper baseline (recognizable, always included in the design),
* the values that are held fixed (asserted against the run template),
* the hard constraints enforced before any simulator run,
* helpers to render a candidate ``Config.toml`` from the run template.

It is intentionally dependency-light (stdlib only) so it can run inside the
``ppbo`` container or on the login node without extra packages.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Parameter definitions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Param:
    """One controllable parameter.

    ``name`` is the short key used in ``design.json``; ``toml_key`` is the
    actual key written into ``Config.toml``. ``kind`` controls sampling and
    rendering:

    * ``continuous``   — real-valued, sampled uniformly in ``[lo, hi]``;
    * ``integer``      — sampled real, rounded to the nearest integer;
    * ``integer_snap`` — sampled real, snapped to a multiple of ``snap``.
    """

    name: str
    toml_key: str
    lo: float
    hi: float
    kind: str
    snap: float | None = None
    unit: str = ""


# Primary Bayesian space (5 effective dimensions). beta is derived, not sampled.
PRIMARY_PARAMS: tuple[Param, ...] = (
    Param("alfa", "alfa", 0.10, 0.90, "continuous"),
    Param("t_base_ms", "t_base_ms", 100.0, 500.0, "continuous", unit="ms"),
    Param("max_distance_m", "max_distance_m", 250.0, 750.0, "continuous", unit="m"),
    Param("trajectory_window", "trajectory_window", 1.0, 5.0, "integer", unit="cells"),
    Param("gossip_interval_ms", "gossip_interval_ms", 200.0, 1000.0,
          "integer_snap", snap=100.0, unit="ms"),
)

# Ancillary factor, studied separately at nominal (not part of the 5-D surrogate).
ANCILLARY_PARAMS: tuple[Param, ...] = (
    Param("anti_entropy_every_n_rounds", "anti_entropy_every_n_rounds",
          10.0, 200.0, "integer", unit="rounds"),
)
ANCILLARY_LEVELS: tuple[int, ...] = (10, 50, 200)

# Paper baseline, Table III of the evaluation section. k=2 is the main config.
BASELINE: dict[str, float] = {
    "alfa": 0.3098,
    "t_base_ms": 307.7,
    "max_distance_m": 500.0,
    "trajectory_window": 3.0,
    "gossip_interval_ms": 500.0,
}
BASELINE_ANCILLARY: dict[str, float] = {"anti_entropy_every_n_rounds": 50.0}

# (section, toml_key) -> expected effective value in the leaderless template.
# The primary study holds all of these fixed; preflight asserts each one.
FIXED_VALUES: dict[tuple[str, str], object] = {
    ("h3", "cluster_resolution"): 10,
    ("h3", "spot_resolution"): 14,
    ("h3", "hysteresis_threshold"): 3,
    ("gossip", "neighbor_k"): 2,
    ("assignment.backoff", "max_wait_s"): 300.0,
    ("assignment.general", "max_pending_slots"): 15,
    ("crdt", "slot_ttl_secs"): 3600,
    ("crdt", "max_entries"): 10000,
    ("vehicle", "max_position_history"): 10,
    ("coordinator", "backend"): "leaderless",
}

# Design constants fixed by the paper's methodology.
DENSITY = "normale"                 # nominal operating regime (primary)
DENSITY_LABEL = {"minimo": "sparse", "normale": "nominal",
                 "trafficato": "congested", "caos": "extreme"}
OCCUPANCIES: tuple[int, ...] = (30, 50, 70, 85, 95)
SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
RUNS_PER_CONFIG = len(OCCUPANCIES) * len(SEEDS)  # 25

# Index block in the sorted ``sumo_cfg_seeds`` listing (100 files, one scheme).
# task id == index - 1; run with sbatch --array=<start>-<stop>.
DENSITY_ARRAY_RANGE = {
    "caos": (0, 24),
    "minimo": (25, 49),
    "normale": (50, 74),
    "trafficato": (75, 99),
}

# Expected fleet size per density (number of <vehicle> in the seed route file),
# from the paper's density table: sparse 45, nominal 120, congested 240,
# extreme 449-450. ``caos`` is centralised-arm-only / not used by the GeoGrid
# study, so it is checked informationally only.
FLEET_SIZE = {"caos": 450, "minimo": 45, "normale": 120, "trafficato": 240}
FLEET_SIZE_INFORMATIONAL = {"caos"}

# Disjoint ZMQ offset blocks: 25 tasks x PORT_STRIDE(50) = 1250 -> block 2000.
ZMQ_OFFSET_GLOBAL_BASE = 14000
ZMQ_OFFSET_BLOCK = 2000
PORT_STRIDE = 50


def beta_of(alfa: float) -> float:
    """beta is always the complement of alpha (hard constraint)."""
    return round(1.0 - float(alfa), 4)


def snap_value(p: Param, v: float) -> float:
    """Apply the parameter's type rule and clamp to its bounds."""
    v = min(max(float(v), p.lo), p.hi)
    if p.kind == "integer":
        return float(round(v))
    if p.kind == "integer_snap":
        assert p.snap, f"{p.name} needs a snap step"
        return float(round(v / p.snap) * p.snap)
    return float(v)


def normalise(params: dict[str, float],
              params_spec: tuple[Param, ...] = PRIMARY_PARAMS) -> dict[str, float]:
    """Snap/clamp every parameter and re-derive beta."""
    out = {p.name: snap_value(p, params[p.name]) for p in params_spec}
    if "alfa" in out:
        out["alfa"] = round(out["alfa"], 4)
    return out


def constraint_violations(params: dict[str, float],
                          params_spec: tuple[Param, ...] = PRIMARY_PARAMS
                          ) -> list[str]:
    """Return a list of human-readable constraint violations (empty = valid).

    These mirror the mathematical/semantic constraints of the simulator:
    positivity, integer domains, the alpha/beta complement, the trajectory
    window vs. the position-history buffer, and the gossip tick granularity.
    """
    errs: list[str] = []
    for p in params_spec:
        v = params.get(p.name)
        if v is None:
            errs.append(f"missing parameter {p.name}")
            continue
        if not (p.lo <= v <= p.hi):
            errs.append(f"{p.name}={v} outside [{p.lo}, {p.hi}]")
        if p.kind in ("integer", "integer_snap") and float(v) != int(v):
            errs.append(f"{p.name}={v} must be an integer")
        if p.kind == "integer_snap" and p.snap and (
                round(v / p.snap) * p.snap != v):
            errs.append(f"{p.name}={v} not a multiple of {p.snap}")

    # beta is always derived, never carried in ``params``; render_config()
    # validates the rendered TOML against its alfa. Nothing to check here.
    # n_tr must fit inside the fixed max_position_history buffer (10). With the
    # current bounds (hi=5) this is defensive and cannot trigger, but it guards
    # against a future widening of the trajectory_window range.
    if params.get("trajectory_window", 0) > FIXED_VALUES[
            ("vehicle", "max_position_history")]:
        errs.append("trajectory_window exceeds max_position_history")
    # gossip interval must be at least one position-update tick (0.1 s = 100 ms).
    if params.get("gossip_interval_ms", 0) < 100:
        errs.append("gossip_interval_ms below one sim tick (100 ms)")
    return errs


# ─────────────────────────────────────────────────────────────────────────────
# Config.toml parsing / rendering
# ─────────────────────────────────────────────────────────────────────────────

_KEY_RE = (r"(?m)^(?P<indent>\s*){key}(?P<eq>\s*=\s*)"
           r"(?P<val>[^#\n]*?)(?P<tail>\s*(?:#.*)?)$")


def render_config(template_text: str, params: dict[str, float],
                  params_spec: tuple[Param, ...] = PRIMARY_PARAMS) -> str:
    """Return *template_text* with the given parameters substituted.

    Only the named keys are touched; indentation and trailing comments are
    preserved. Raises ``RuntimeError`` if a key is missing from the template,
    so a wrong template fails loudly instead of silently keeping an old value.
    """
    text = template_text
    replacements: list[tuple[str, str]] = []
    for p in params_spec:
        replacements.append((p.toml_key, _format_value(p, params[p.name])))
    if "alfa" in params:
        replacements.append(("beta", f"{beta_of(params['alfa']):.4f}"))

    for key, value in replacements:
        pat = re.compile(_KEY_RE.format(key=re.escape(key)))
        new_text, n = pat.subn(
            lambda m: f"{m.group('indent')}{key}{m.group('eq')}{value}{m.group('tail')}",
            text, count=1)
        if n == 0:
            raise RuntimeError(f"key '{key}' not found in the config template")
        text = new_text

    # Real guard: re-parse the rendered text and verify every written value
    # (including the derived beta) actually landed. This is what catches a
    # template whose formatting defeats the regex; the old dead `beta` check
    # in constraint_violations did not.
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"rendered Config.toml is not valid TOML: {exc}") from exc
    expected = {key: float(value) for key, value in replacements}
    for key, exp in expected.items():
        found = _find_all(parsed, key)
        if not any(isinstance(v, (int, float)) and abs(float(v) - exp) < 1e-6
                   for v in found):
            raise RuntimeError(
                f"rendered {key}={found!r} does not match expected {exp}")
    return text


def _find_all(node, key):
    """Recursively collect every value stored under *key* in a parsed TOML."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                out.append(v)
            else:
                out.extend(_find_all(v, key))
    return out


def _format_value(p: Param, v: float) -> str:
    if p.kind in ("integer", "integer_snap"):
        return str(int(round(v)))
    if p.name in ("alfa", "t_base_ms"):
        return f"{v:.4f}" if p.name == "alfa" else f"{v:.1f}"
    return f"{v:.1f}"


def read_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def effective_value(cfg: dict, section: str, key: str):
    """Read ``key`` from a (possibly dotted) TOML section, or ``None``."""
    node = cfg
    for part in section.split("."):
        node = node.get(part, {})
        if not isinstance(node, dict):
            return None
    return node.get(key)


def find_template(candidates: list[Path]) -> Path | None:
    """Return the first candidate whose backend is ``leaderless``."""
    for cand in candidates:
        try:
            cfg = read_toml(cand)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if effective_value(cfg, "coordinator", "backend") == "leaderless":
            return cand
    return None
