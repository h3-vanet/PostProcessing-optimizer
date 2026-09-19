#!/usr/bin/env python3
"""Surrogate-based sensitivity analysis for the GeoGrid nominal study.

Modes
-----
``aggregated`` (default)
    One row per parameter configuration (``dataset_aggregated.csv``), target
    ``<response>_mean``. Smallest/cheapest; CV is leave-one-configuration-out.

``paired``
    One row per (configuration, scenario cell), target
    ``response - response(baseline)`` on the same cell. Removes the scenario
    variance that the common-random-number design creates; this is the primary
    analysis. Requires ``dataset_per_run.csv``.

``per_run``
    One row per run, with ``occupancy`` and one-hot ``seed`` as covariates, to
    check that parameter effects are comparable in size to occupancy (the one
    interpretable reference scale). Diagnostic.

Honest evaluation, in every mode
--------------------------------
* CV is **leave-one-configuration-out** (LOCO): the model must interpolate in
  parameter space to a configuration it has never seen. Reported as ``r2_loco``.
* ``r2_locell`` (leave-one-scenario-cell-out, all configurations present in
  training) is reported separately as a *distinct* quantity, never as a single
  ``r2_cv``. In paired mode it is close to trivial by construction and its only
  use is to show the gap.
* Permutation importance is computed **out-of-fold** and measures predictive
  reliance, not memorisation. Because the parameters are constant inside a
  configuration, the feature column is permuted **globally across
  configurations** and re-predicted with the already-fitted held-out-group
  models; a within-test-fold permutation would be structurally zero whenever a
  whole configuration (or a single configuration) is held out.
* The primary grouping is ``loco_oof_primary``; ``locell_oof_diagnostic`` is
  diagnostic only and must not be read as a correction of the primary number.
  In paired mode ``loco_oof_primary ~ 0`` on the knobs means the response
  surface is not estimable with this many configurations -- the signal that
  Phase B (more configurations) is needed, not leakage.
* A parameter is called material only if its primary importance exceeds an
  explicit **95% null band** built by permuting the response relative to the
  features (no dependence by construction). ``r2_loco`` is reported as support,
  never as the decision.

All feature matrices are kept in **raw parameter units** for fit and predict
(ExtraTrees are scale-invariant; the only requirement is consistency). Candidate
points for Phase B are generated in raw units too, which is what fixes the
earlier training/prediction scale mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec  # noqa: E402

PARAM_NAMES = [p.name for p in spec.PRIMARY_PARAMS]
PARAMS = {p.name: p for p in spec.PRIMARY_PARAMS}
DEFAULT_RESPONSES = [
    "C1_park_rate", "A5_jain", "A5b_overlap_rate_pct_d010",
    "A1_p50_ms", "D1_cp_tx_bytes_per_vehicle_mean", "nr_plr_pct",
]
CELL_KEYS = ["traffic", "occupancy", "seed"]


def _versions() -> dict:
    """Effective versions of the analysis stack, recorded in the summary."""
    import importlib.metadata as md
    out = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:  # pragma: no cover - package absent
            out[pkg] = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────


def _cell_id(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in CELL_KEYS if c in df.columns]
    return df[cols].astype(str).agg("|".join, axis=1)


def build_aggregated(path: Path, responses: list[str]):
    df = pd.read_csv(path).sort_values("config_id").reset_index(drop=True)
    X = df[PARAM_NAMES].to_numpy(float)
    features = list(PARAM_NAMES)
    targets, groups_cfg = {}, df["config_id"].to_numpy()
    for r in responses:
        if f"{r}_mean" in df.columns:
            targets[r] = pd.to_numeric(df[f"{r}_mean"], errors="coerce").to_numpy(float)
    return X, features, targets, groups_cfg, None, {"n_config": len(df)}


def build_paired(path: Path, responses: list[str], baseline_id: str):
    return build_paired_from_frame(pd.read_csv(path), responses, baseline_id)


def build_paired_from_frame(df: pd.DataFrame, responses: list[str],
                            baseline_id: str):
    df = df.copy()
    df["_cell"] = _cell_id(df)
    base = df[df["config_id"] == baseline_id]
    if base.empty:
        raise SystemExit(f"baseline config {baseline_id!r} not found in {path}")
    base_map = {}
    for r in responses:
        if r in df.columns:
            base_map[r] = base.groupby("_cell")[r].mean()
    frames = []
    for r in responses:
        if r not in df.columns or r not in base_map:
            continue
        sub = df[["config_id", "_cell"] + PARAM_NAMES + ["valid"]].copy()
        sub["response"] = r
        sub["value"] = pd.to_numeric(df[r], errors="coerce")
        sub["baseline"] = sub["_cell"].map(base_map[r])
        sub["delta"] = sub["value"] - sub["baseline"]
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if long.empty or PARAM_NAMES[0] not in long.columns:
        return (np.empty((0, len(PARAM_NAMES))), list(PARAM_NAMES), {},
                np.array([]), np.array([]), {"n_rows": 0, "baseline": baseline_id})
    X = long[PARAM_NAMES].to_numpy(float)
    features = list(PARAM_NAMES)
    resp_arr = long["response"].to_numpy()
    delta_arr = pd.to_numeric(long["delta"], errors="coerce").to_numpy(float)
    # y is full-length with NaN outside the response's own rows, so it stays
    # aligned with X/groups; the per-response mask selects the relevant rows.
    targets = {r: np.where(resp_arr == r, delta_arr, np.nan)
               for r in responses if r in set(resp_arr)}
    groups_cfg = long["config_id"].to_numpy()
    groups_cell = long["_cell"].to_numpy()
    return X, features, targets, groups_cfg, groups_cell, {
        "n_rows": len(long), "baseline": baseline_id}


def build_per_run(path: Path, responses: list[str]):
    df = pd.read_csv(path)
    df["_cell"] = _cell_id(df)
    X = df[PARAM_NAMES + ["occupancy"]].to_numpy(float)
    features = list(PARAM_NAMES) + ["occupancy"]
    seed_dummies = pd.get_dummies(df["seed"], prefix="seed").astype(float)
    X = np.hstack([X, seed_dummies.to_numpy(float)])
    features += list(seed_dummies.columns)
    targets = {r: pd.to_numeric(df[r], errors="coerce").to_numpy(float)
               for r in responses if r in df.columns}
    return X, features, targets, df["config_id"].to_numpy(), \
        df["_cell"].to_numpy(), {"n_rows": len(df)}


# ─────────────────────────────────────────────────────────────────────────────
# Honest cross-validation / importance
# ─────────────────────────────────────────────────────────────────────────────


def _logo_splits(X, y, groups):
    from sklearn.model_selection import LeaveOneGroupOut
    return list(LeaveOneGroupOut().split(X, y, groups))


def oof_predictions(X, y, groups, seed, n_estimators, n_jobs):
    from sklearn.ensemble import ExtraTreesRegressor
    preds = np.full(len(y), np.nan)
    for tr, te in _logo_splits(X, y, groups):
        model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed,
                                    n_jobs=n_jobs).fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
    return preds


def oof_permutation_importance(X, y, groups, feature_names, seed,
                               n_estimators, n_repeats, n_jobs):
    """Grouped out-of-fold permutation importance.

    The parameters are constant within a configuration, so the usual
    within-test-fold permutation is degenerate whenever the CV holds out whole
    configurations (a single configuration leaves nothing to permute). Instead
    we permute the feature column **globally across configurations** and
    re-predict each held-out group with the model that was fitted without it.
    This yields a valid counterfactual ("what if this configuration had a
    different value of parameter f?") and is non-zero when the surrogate relies
    on f to generalise.

    Fold models are fitted once; each repeat only re-predicts, so the cost is
    dominated by the number of LOCO folds.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import mean_squared_error
    rng = np.random.default_rng(seed)
    folds = _logo_splits(X, y, groups)
    models, oof = [], np.full(len(y), np.nan)
    for tr, te in folds:
        model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed,
                                    n_jobs=n_jobs).fit(X[tr], y[tr])
        models.append(model)
        oof[te] = model.predict(X[te])
    base = float(mean_squared_error(y, oof))

    acc = {f: [] for f in feature_names}
    for fi, feat in enumerate(feature_names):
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, fi] = rng.permutation(Xp[:, fi])
            preds = np.full(len(y), np.nan)
            for (tr, te), model in zip(folds, models):
                preds[te] = model.predict(Xp[te])
            acc[feat].append(float(mean_squared_error(y, preds)) - base)
    return {f: (float(np.mean(v)), float(np.std(v))) for f, v in acc.items()}


def null_band_importance(X, y, groups, feature_names, seed, n_estimators,
                         n_jobs, n_perm=100, q=0.95):
    """95% null band for the OOF permutation importance.

    ``importances_std`` from permutation_importance measures permutation
    variability, not the uncertainty of the estimate over few training points,
    so ``mean - std > 0`` is optimistic. Instead we build an explicit null:
    shuffle the response relative to the features (no dependence by
    construction), recompute the OOF importance, and take the ``q`` quantile of
    the resulting distribution per feature. A parameter is called material only
    if its observed primary importance exceeds this band.

    Cost is ``n_perm`` full LOCO passes (each with one permutation repeat), so
    it is computed only for the primary grouping and can be disabled with
    ``n_perm=0``.
    """
    if n_perm <= 0:
        return {f: float("nan") for f in feature_names}
    rng = np.random.default_rng(seed + 101)
    null = {f: [] for f in feature_names}
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        imp = oof_permutation_importance(X, y_perm, groups, feature_names,
                                         seed, n_estimators, n_repeats=1,
                                         n_jobs=n_jobs)
        for feat, (m, _s) in imp.items():
            null[feat].append(m)
    return {f: float(np.quantile(v, q)) for f, v in null.items()}


def reliable_r2(y, preds):
    from sklearn.metrics import r2_score
    finite = np.isfinite(preds) & np.isfinite(y)
    if finite.sum() < 3:
        return float("nan")
    return float(r2_score(y[finite], preds[finite]))


def pdp_and_interactions(X, y, feature_names, seed, n_estimators, n_jobs, grid,
                         interaction_indices=None):
    """Descriptive PDP (full-data fit) and pairwise H^2 with an estimability
    flag. H^2 is reported but not trusted unless each axis has enough distinct
    values and there are enough observations. ``interaction_indices`` limits the
    H^2 pairs (default: all features)."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.inspection import partial_dependence
    model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed,
                                n_jobs=n_jobs).fit(X, y)
    pdp_rows, interaction_rows = [], []
    n_unique = [len(np.unique(X[:, j])) for j in range(X.shape[1])]
    if interaction_indices is None:
        interaction_indices = list(range(X.shape[1]))

    def pdp1(j):
        res = partial_dependence(model, X, features=[j],
                                 grid_resolution=grid, kind="average")
        key = "grid_values" if "grid_values" in res else "values"
        return (np.asarray(res[key][0], float).ravel(),
                np.asarray(res["average"][0], float).ravel())

    cache = {j: pdp1(j) for j in range(X.shape[1])}
    for j, name in enumerate(feature_names):
        for xv, yv in zip(*cache[j]):
            pdp_rows.append({"parameter": name, "x": float(xv), "pdp": float(yv)})

    for a in interaction_indices:
        for b in interaction_indices:
            if b <= a:
                continue
            res = partial_dependence(model, X, features=[a, b],
                                     grid_resolution=grid, kind="average")
            key = "grid_values" if "grid_values" in res else "values"
            za = np.asarray(res["average"][0], float)
            joint = za - za.mean()
            marg = ((cache[a][1] - cache[a][1].mean())[:, None]
                    + (cache[b][1] - cache[b][1].mean())[None, :])
            denom = float(np.sum(joint ** 2))
            h2 = float(np.sum((joint - marg) ** 2) / denom) if denom else 0.0
            interaction_rows.append({
                "param_a": feature_names[a], "param_b": feature_names[b],
                "H2": round(h2, 6),
                "n_unique_a": n_unique[a], "n_unique_b": n_unique[b],
                "estimable": bool(min(n_unique[a], n_unique[b]) >= 8 and len(y) >= 30),
            })
    return pdp_rows, interaction_rows


# ─────────────────────────────────────────────────────────────────────────────
# Phase-B candidate generation (raw units throughout)
# ─────────────────────────────────────────────────────────────────────────────


def generate_candidates(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = len(PARAM_NAMES)
    out = np.empty((0, d))
    rounds = 0
    while len(out) < 4000 and rounds < 20:
        rounds += 1
        u = rng.random((2000, d))
        snapped = np.column_stack([
            np.array([spec.snap_value(p, p.lo + u[i, j] * (p.hi - p.lo))
                      for i in range(len(u))])
            for j, p in enumerate(spec.PRIMARY_PARAMS)])
        ok = np.array([
            not spec.constraint_violations(
                {n_: snapped[i, j] for j, n_ in enumerate(PARAM_NAMES)})
            for i in range(len(snapped))])
        out = np.vstack([out, snapped[ok]])
    return out[:4000]


def _unit(X: np.ndarray) -> np.ndarray:
    lo = np.array([PARAMS[n].lo for n in PARAM_NAMES], float)
    hi = np.array([PARAMS[n].hi for n in PARAM_NAMES], float)
    return (X - lo) / (hi - lo)


def suggest_phase_b(X_obs: np.ndarray, targets: dict, responses: list[str],
                    n_points: int, seed: int, mode: str,
                    n_estimators: int, n_jobs: int) -> list[dict]:
    """Return N raw-unit parameter dicts.

    ``maximin``: maximin spacing over the candidate cloud, seeded with the
    already-observed configurations, so the batch fills the largest holes.
    ``uncertainty``: uncertainty-weighted maximin (ExtraTrees inter-tree
    variance), only meaningful with enough configurations to trust a surrogate.
    """
    from sklearn.ensemble import ExtraTreesRegressor

    # Guard against the earlier scale bug: fit and predict must be raw units.
    lo = np.array([PARAMS[n].lo for n in PARAM_NAMES], float)
    hi = np.array([PARAMS[n].hi for n in PARAM_NAMES], float)
    assert np.all(X_obs >= lo - 1e-9) and np.all(X_obs <= hi + 1e-9), \
        "X_obs is not in raw parameter units"

    cand = generate_candidates(4000, seed)
    assert np.all(cand >= lo - 1e-9) and np.all(cand <= hi + 1e-9), \
        "candidates are not in raw parameter units"

    cand_u = _unit(cand)
    obs_u = _unit(X_obs)

    var_norm = np.zeros(len(cand))
    if mode == "uncertainty":
        for r in responses:
            y = targets.get(r)
            if y is None:
                continue
            mask = np.isfinite(y)
            if mask.sum() < 4:
                continue
            model = ExtraTreesRegressor(n_estimators=n_estimators,
                                        random_state=seed, n_jobs=n_jobs)
            model.fit(X_obs[mask], y[mask])
            var_norm += np.stack([t.predict(cand)
                                  for t in model.estimators_]).var(axis=0)
        if var_norm.max() > 0:
            var_norm /= var_norm.max()

    # seeds of the greedy maximin: observed points plus already-chosen ones.
    chosen_u = [obs_u[j] for j in range(len(obs_u))]
    selected: list[int] = []
    for _ in range(n_points):
        best_i, best_score = None, -np.inf
        for i in range(len(cand)):
            if i in selected:
                continue
            dmin = min(np.linalg.norm(cand_u[i] - c) for c in chosen_u)
            score = dmin if mode == "maximin" else dmin * (1.0 + var_norm[i])
            if score > best_score:
                best_score, best_i = score, i
        selected.append(int(best_i))
        chosen_u.append(cand_u[best_i])

    return [spec.normalise({n: cand[i, j] for j, n in enumerate(PARAM_NAMES)})
            for i in selected]


def write_phase_b(design: dict, suggested: list[dict], template: Path,
                  out: Path, mode: str) -> Path:
    base_offset = max(c["zmq_offset_base"] for c in design["configs"]) + \
        spec.ZMQ_OFFSET_BLOCK
    cfg_dir = out / "configs_phase_b"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    text = template.read_text()
    start, stop = spec.DENSITY_ARRAY_RANGE[spec.DENSITY]
    records = []
    for i, params in enumerate(suggested):
        cid = f"sens_b{i:02d}"
        p = spec.normalise(params)
        (cfg_dir / f"{cid}.toml").write_text(spec.render_config(text, p))
        records.append({
            "config_id": cid, "is_baseline": False,
            "params": {k: (int(v) if k in ("trajectory_window",
                                           "gossip_interval_ms") else round(v, 4))
                       for k, v in p.items()},
            "beta": spec.beta_of(p["alfa"]),
            "config_toml": str((cfg_dir / f"{cid}.toml").relative_to(out)),
            "log_subdir": cid,
            "zmq_offset_base": base_offset + i * spec.ZMQ_OFFSET_BLOCK,
            "array": [start, stop],
        })
    design_b = dict(design)
    design_b.update({"phase": "B",
                     "design": "maximin_extension" if mode == "maximin"
                               else "uncertainty_driven_refinement",
                     "phase_b_mode": mode,
                     "n_configs": len(records),
                     "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "configs": records})
    path = out / "design_phase_b.json"
    path.write_text(json.dumps(design_b, indent=2))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def build_decision(summary: dict, null_perm: int) -> dict:
    """Derive the Phase-B recommendation from the primary importances.

    * ``phase_b_recommended`` is True when no response has any of the 5 knobs
      exceeding the null band, i.e. with the current configurations the
      response surface is not estimable and more configurations are needed.
    * The supporting numbers are reported alongside the boolean, so the
      recommendation can be contested by looking at the detail.
    """
    with_signal = [r for r in summary if summary[r].get("n_params_above_null")]
    return {
        "phase_b_recommended": (None if null_perm <= 0 else len(with_signal) == 0),
        "criterion": ("primary (loco_oof_primary) permutation importance exceeds the "
                      "95% null band from response permutation"),
        "null_perm": null_perm,
        "n_responses": len(summary),
        "n_responses_with_signal": len(with_signal),
        "responses_with_signal": with_signal,
        "signal_by_response": {r: summary[r]["params_above_null"] for r in with_signal},
        "primary_grouping": "loco_oof_primary",
        "note": ("in paired mode the primary grouping generalises to unseen "
                 "configurations; locell_oof_diagnostic is not a generalisation "
                 "test and must not be read as one"),
    }


def main() -> int:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("ERROR: scikit-learn is required for surrogate_sensitivity.py.\n"
              "Install with:  pip install scikit-learn", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("aggregated", "paired", "per_run"),
                    default="paired")
    ap.add_argument("--dataset", type=Path,
                    help="dataset_aggregated.csv (mode=aggregated)")
    ap.add_argument("--per-run-dataset", type=Path,
                    help="dataset_per_run.csv (mode=paired|per_run)")
    ap.add_argument("--baseline-config", default="sens_c00",
                    help="baseline config id for paired differences")
    ap.add_argument("--design", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--responses", default=",".join(DEFAULT_RESPONSES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--null-perm", type=int, default=100,
                    help="response permutations for the 95%% null band; 0 disables "
                         "the band (and the phase-B recommendation)")
    ap.add_argument("--suggest-phase-b", type=int, default=0, metavar="N")
    ap.add_argument("--phase-b-mode", choices=("maximin", "uncertainty"),
                    default="maximin")
    ap.add_argument("--template", type=Path, default=None)
    args = ap.parse_args()

    responses = [r for r in args.responses.split(",") if r]
    design = json.loads(args.design.read_text())
    out = (args.out or args.design.parent).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "aggregated":
        if not args.dataset:
            print("ERROR: --dataset required for mode=aggregated", file=sys.stderr)
            return 2
        X, features, targets, gcfg, gcell, meta = build_aggregated(args.dataset, responses)
    else:
        if not args.per_run_dataset:
            print("ERROR: --per-run-dataset required for paired/per_run",
                  file=sys.stderr)
            return 2
        if args.mode == "paired":
            X, features, targets, gcfg, gcell, meta = build_paired(
                args.per_run_dataset, responses, args.baseline_config)
        else:
            X, features, targets, gcfg, gcell, meta = build_per_run(
                args.per_run_dataset, responses)
    if X.shape[0] < 4:
        print("ERROR: too few observations for a surrogate", file=sys.stderr)
        return 1

    # Config-level data for the Phase-B proposal and for r2_loco when possible.
    print(f"mode={args.mode}  features={features}  rows={X.shape[0]}  meta={meta}")

    imp_rows, pdp_rows, inter_rows, summary = [], [], [], {}
    for r in responses:
        y = targets.get(r)
        if y is None:
            print(f"  [skip] {r}: not in dataset")
            continue
        mask = np.isfinite(y)
        if mask.sum() < 4:
            print(f"  [skip] {r}: only {int(mask.sum())} finite")
            continue
        Xr, yr = X[mask], y[mask]
        gcfg_r = gcfg[mask]

        oof = oof_predictions(Xr, yr, gcfg_r, args.seed, args.n_estimators, args.n_jobs)
        r2_loco = reliable_r2(yr, oof)
        r2_locell = float("nan")
        if gcell is not None and args.mode != "aggregated":
            gcell_r = gcell[mask]
            if len(np.unique(gcell_r)) > 1:
                oof_c = oof_predictions(Xr, yr, gcell_r, args.seed,
                                        args.n_estimators, args.n_jobs)
                r2_locell = reliable_r2(yr, oof_c)

        # PRIMARY importance: leave-one-configuration-out. In paired mode the
        # model must interpolate in parameter space to a configuration it never
        # saw; loco_oof ~ 0 there means the response surface is not estimable
        # with this many configurations, which is the Phase-B signal -- NOT
        # leakage (paired X carries only the 5 knobs, no cell columns).
        imp = oof_permutation_importance(Xr, yr, gcfg_r, features, args.seed,
                                         args.n_estimators, args.n_repeats,
                                         args.n_jobs)
        band = null_band_importance(Xr, yr, gcfg_r, features, args.seed,
                                    args.n_estimators, args.n_jobs,
                                    n_perm=args.null_perm)
        for feat, (m, s) in imp.items():
            b = band.get(feat)
            imp_rows.append({
                "response": r, "parameter": feat,
                "importance_mean": round(m, 6),
                "importance_std": round(s, 6),
                "null_band_q95": None if b is None or np.isnan(b) else round(b, 6),
                "exceeds_null": None if b is None or np.isnan(b) else bool(m > b),
                "grouping": "loco_oof_primary"})

        # DIAGNOSTIC importance: leave-one-cell-out. Holds out a scenario cell
        # with all configurations still in training, and the scenario variance
        # has already been removed by the paired difference -- so parameters
        # "emerging" here is near-trivial and does NOT show generalisation to
        # new knob values. Never read as a correction of the primary number.
        if gcell is not None and args.mode != "aggregated":
            gcell_r = gcell[mask]
            if len(np.unique(gcell_r)) > 1:
                imp_c = oof_permutation_importance(
                    Xr, yr, gcell_r, features, args.seed, args.n_estimators,
                    args.n_repeats, args.n_jobs)
                for feat, (m, s) in imp_c.items():
                    imp_rows.append({"response": r, "parameter": feat,
                                     "importance_mean": round(m, 6),
                                     "importance_std": round(s, 6),
                                     "null_band_q95": None, "exceeds_null": None,
                                     "grouping": "locell_oof_diagnostic"})

        pdp, inter = pdp_and_interactions(
            Xr, yr, features, args.seed, args.n_estimators, args.n_jobs,
            args.grid,
            interaction_indices=[features.index(n) for n in PARAM_NAMES
                                 if n in features])
        for row in pdp:
            row["response"] = r
        for row in inter:
            row["response"] = r
        pdp_rows += pdp
        inter_rows += inter

        # Only the 5 knobs are judged for the Phase-B decision; occupancy/seed
        # covariates in per_run mode are not protocol parameters.
        has_band = args.null_perm > 0
        above = [f for f in PARAM_NAMES if f in imp
                 and imp[f][0] > band.get(f, np.nan)] if has_band else []
        summary[r] = {"r2_loco": None if np.isnan(r2_loco) else round(r2_loco, 4),
                      "r2_locell": None if np.isnan(r2_locell) else round(r2_locell, 4),
                      "n_rows": int(mask.sum()),
                      "n_config": int(len(np.unique(gcfg_r))),
                      "primary_grouping": "loco_oof_primary",
                      "diagnostic_grouping": "locell_oof_diagnostic"
                                             if gcell is not None
                                             and args.mode != "aggregated" else None,
                      "n_params_above_null": len(above) if has_band else None,
                      "params_above_null": above if has_band else None,
                      "mode": args.mode}
        print(f"  {r}: r2_loco={summary[r]['r2_loco']} "
              f"r2_locell={summary[r]['r2_locell']} "
              f"above_null={above if has_band else 'n/a'} (n={int(mask.sum())})",
              flush=True)

    decision = build_decision(summary, args.null_perm)
    result = {"mode": args.mode, "versions": _versions(),
              "responses": summary, "decision": decision}
    pd.DataFrame(imp_rows).to_csv(out / "sensitivity_importance.csv", index=False)
    pd.DataFrame(pdp_rows).to_csv(out / "sensitivity_pdp.csv", index=False)
    pd.DataFrame(inter_rows).to_csv(out / "sensitivity_interactions.csv", index=False)
    (out / "sensitivity_summary.json").write_text(json.dumps(result, indent=2))
    print(f"\nphase_b_recommended={decision['phase_b_recommended']} "
          f"({decision['n_responses_with_signal']}/{decision['n_responses']} "
          f"responses with signal)")
    print(f"Wrote importance/PDP/interactions -> {out}")

    if args.suggest_phase_b:
        if not args.template or not args.template.is_file():
            print("ERROR: --template required for --suggest-phase-b", file=sys.stderr)
            return 2
        # Config-level targets for the proposal.
        cfg_df = pd.DataFrame(X[:, :len(PARAM_NAMES)], columns=PARAM_NAMES)
        cfg_df["config_id"] = gcfg
        X_cfg = cfg_df.groupby("config_id")[PARAM_NAMES].first().to_numpy(float)
        cfg_targets = {
            r: np.asarray(
                cfg_df.assign(v=np.asarray(targets[r]))
                .groupby("config_id")["v"].mean().to_numpy())
            for r in responses if r in targets}
        suggested = suggest_phase_b(X_cfg, cfg_targets, responses,
                                    args.suggest_phase_b, args.seed,
                                    args.phase_b_mode, args.n_estimators, args.n_jobs)
        path = write_phase_b(design, suggested, args.template, out, args.phase_b_mode)
        print(f"Phase-B candidates ({args.phase_b_mode}) -> {path}")
        for s in suggested:
            print(f"  alfa={s['alfa']:.3f}, t_base={s['t_base_ms']:.1f}, "
                  f"Dmax={s['max_distance_m']:.1f}, ntr={int(s['trajectory_window'])}, "
                  f"Tgossip={int(s['gossip_interval_ms'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
