#!/usr/bin/env python3
"""Regression tests for the sensitivity study.

Focus on the invariants whose violation is easy to miss in an end-to-end run:

* ``render_config`` round-trips and the derived ``beta`` really lands in the
  rendered TOML (the old no-op check would not catch a broken template);
* the design always contains the baseline and only valid configurations;
* Phase-B candidates are generated in **raw parameter units** (the earlier bug
  fitted on raw units and predicted on unit-cube candidates, silently turning
  uncertainty-driven selection into pure maximin);
* paired targets stay aligned with the feature matrix (full-length arrays with
  NaN outside each response's rows).

Run with:  python3 -m pytest sensitivity/test_sensitivity.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402


TEMPLATE = """\
[h3]
cluster_resolution = 10
spot_resolution = 14
hysteresis_threshold = 3
[gossip]
anti_entropy_every_n_rounds = 50
gossip_interval_ms = 500
neighbor_k = 2
[assignment.backoff]
alfa = 0.7
beta = 0.3
t_base_ms = 200.0
max_distance_m = 500.0
trajectory_window = 3
max_wait_s = 300.0
[assignment.general]
max_pending_slots = 15
[crdt]
slot_ttl_secs = 3600
max_entries = 10000
[vehicle]
max_position_history = 10
[coordinator]
backend = "leaderless"
"""


def test_render_config_writes_derived_beta():
    params = spec.normalise(spec.BASELINE)
    out = spec.render_config(TEMPLATE, params)
    cfg = __import__("tomllib").loads(out)
    assert cfg["assignment"]["backoff"]["alfa"] == pytest.approx(params["alfa"])
    assert cfg["assignment"]["backoff"]["beta"] == pytest.approx(
        spec.beta_of(params["alfa"]))
    assert cfg["coordinator"]["backend"] == "leaderless"


def test_render_config_rejects_broken_template():
    # a template missing the beta key must fail loudly, not keep the old value
    with pytest.raises(RuntimeError):
        spec.render_config("[assignment.backoff]\nalfa = 0.7\n", spec.normalise(spec.BASELINE))


def test_build_configs_includes_baseline_and_is_valid():
    import make_design
    configs = make_design.build_configs(10, seed=42)
    assert len(configs) == 10
    assert configs[0]["is_baseline"] is True
    assert configs[0]["params"]["alfa"] == pytest.approx(spec.BASELINE["alfa"])
    for c in configs:
        assert spec.constraint_violations(c["params"]) == []
    # no duplicated LHS rows
    rows = [tuple(c["params"][p.name] for p in spec.PRIMARY_PARAMS)
            for c in configs if not c["is_baseline"]]
    assert len(set(rows)) == len(rows)


def test_suggest_phase_b_candidates_are_raw_units():
    """Regression for the scale bug: candidates must be in raw parameter units."""
    import surrogate_sensitivity as ss
    rng = np.random.default_rng(0)
    X = np.tile([spec.BASELINE[p.name] for p in spec.PRIMARY_PARAMS], (6, 1))
    X = X + rng.normal(scale=0.01, size=X.shape)
    targets = {"r": X[:, 0] * 10.0}  # arbitrary finite target
    out = ss.suggest_phase_b(X, targets, ["r"], 4, seed=0,
                             mode="uncertainty", n_estimators=30, n_jobs=1)
    assert len(out) == 4
    for rec in out:
        for p in spec.PRIMARY_PARAMS:
            assert p.lo - 1e-6 <= rec[p.name] <= p.hi + 1e-6


def test_paired_targets_align_with_features():
    import pandas as pd
    import surrogate_sensitivity as ss
    n_cfg, n_cell = 4, 3
    rows = []
    for c in range(n_cfg):
        for cell in range(n_cell):
            row = {"config_id": f"sens_c{c:02d}", "traffic": "normale",
                   "occupancy": 30 + cell, "seed": cell + 1, "valid": True}
            for j, p in enumerate(spec.PRIMARY_PARAMS):
                row[p.name] = spec.BASELINE[p.name] + 0.01 * (c + 1) * (j + 1)
            row["C1_park_rate"] = 80.0 + c - cell
            rows.append(row)
    df = pd.DataFrame(rows)
    X, features, targets, gcfg, gcell, meta = ss.build_paired_from_frame(
        df, ["C1_park_rate"], "sens_c00")
    for y in targets.values():
        assert y.shape[0] == X.shape[0]
    assert len(gcfg) == X.shape[0]
    finite = targets["C1_park_rate"][gcfg != "sens_c00"]
    assert np.isfinite(finite).any()


def test_null_band_detects_signal_and_ignores_other_features():
    """A parameter with a real effect must clear the response-permutation band;
    an irrelevant one must not dominate it."""
    import surrogate_sensitivity as ss
    rng = np.random.default_rng(0)
    n, d = 24, 5
    X = rng.random((n, d))
    groups = np.array([f"c{i}" for i in range(8) for _ in range(3)])
    features = [f"x{i}" for i in range(d)]
    y = 10.0 * X[:, 0] + rng.normal(scale=0.1, size=n)
    imp = ss.oof_permutation_importance(X, y, groups, features, seed=0,
                                        n_estimators=50, n_repeats=3, n_jobs=1)
    band = ss.null_band_importance(X, y, groups, features, seed=0,
                                   n_estimators=50, n_jobs=1, n_perm=20)
    assert imp["x0"][0] > band["x0"]
    assert imp["x0"][0] > imp["x1"][0]


def test_grouped_importance_nonzero_for_config_constant_features():
    """Regression: with configurations held out whole, the parameters are
    constant inside a test folder, so a within-fold permutation would be
    structurally zero. The grouped (global-across-configs) estimator must be
    nonzero for a parameter the response actually depends on."""
    import surrogate_sensitivity as ss
    rng = np.random.default_rng(3)
    n_cfg = 8
    X = np.tile(np.array([spec.BASELINE[p.name]
                          for p in spec.PRIMARY_PARAMS]), (n_cfg, 1))
    X[:, 0] = rng.uniform(0.1, 0.9, n_cfg)          # alfa varies per config
    y = 50.0 * X[:, 0]                              # strong reliance on alfa
    features = [p.name for p in spec.PRIMARY_PARAMS]
    groups = np.array([f"c{i}" for i in range(n_cfg)])
    imp = ss.oof_permutation_importance(X, y, groups, features, seed=0,
                                        n_estimators=50, n_repeats=3, n_jobs=1)
    assert imp["alfa"][0] > 0.0
    assert imp["alfa"][0] > imp["gossip_interval_ms"][0]


def test_assert_cell_coverage_accepts_grid_and_rejects_aggregated():
    """The paired/CRN design needs one row per (occupancy, seed); a 5-row
    aggregated summary must be rejected before it silently produces wrong
    numbers."""
    import pandas as pd
    import assemble_dataset as ad
    rows = [{"occupancy": o, "seed": s, "traffic": "normale"}
            for o in spec.OCCUPANCIES for s in spec.SEEDS]
    good = pd.DataFrame(rows)
    ad._assert_cell_coverage(good, "sens_c00")  # must not raise

    aggregated = pd.DataFrame([{"occupancy": o, "seed": 1, "traffic": "normale"}
                               for o in spec.OCCUPANCIES])
    with pytest.raises(ValueError):
        ad._assert_cell_coverage(aggregated, "sens_c00")

    duplicate = good.copy()
    duplicate.loc[0, "seed"] = duplicate.loc[1, "seed"]  # duplicate, one missing
    with pytest.raises(ValueError):
        ad._assert_cell_coverage(duplicate, "sens_c01")


def test_build_decision_reports_supporting_numbers():
    import surrogate_sensitivity as ss
    no_signal = {r: {"n_params_above_null": 0, "params_above_null": []}
                 for r in ["a", "b"]}
    d = ss.build_decision(no_signal, null_perm=100)
    assert d["phase_b_recommended"] is True
    assert d["n_responses_with_signal"] == 0
    assert d["primary_grouping"] == "loco_oof_primary"

    some = {"a": {"n_params_above_null": 1, "params_above_null": ["alfa"]},
            "b": {"n_params_above_null": 0, "params_above_null": []}}
    d2 = ss.build_decision(some, null_perm=100)
    assert d2["phase_b_recommended"] is False
    assert d2["responses_with_signal"] == ["a"]
    assert d2["signal_by_response"] == {"a": ["alfa"]}

    assert ss.build_decision(no_signal, null_perm=0)["phase_b_recommended"] is None
