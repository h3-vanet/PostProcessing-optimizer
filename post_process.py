#!/usr/bin/env python3
"""
VANET-Parking Post-Processor — Three-Source Analysis
=====================================================
Statistical analysis and visualization of simulation logs.

Usage:
    python post_process.py                          # batch: process all experiments
    python post_process.py --exp <experiment_name>   # single experiment
    python post_process.py --help                    # full help
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

LOG_BASE = Path("./logs")
OUT_DIR  = Path("./output")
FIG_DIR  = OUT_DIR / "figures"
WARMUP_END = 0  # t_s restarts from 0 each run (0–300s); no warm-up to skip.
                # Old logs kept the Rust process alive across runs with epoch-
                # relative timestamps and needed 65_000 ms; new logs restart the
                # container per run, so the full 0–300s window is valid.

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_palette("tab10")


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _find_log(base: Path, subdir: str, pattern: str) -> Path:
    """Find the most recent log in *subdir* under *base*."""
    candidates = sorted((base / subdir).glob(pattern)) if (base / subdir).exists() else []
    if not candidates:
        candidates = sorted(base.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No log found in {base / subdir} for pattern {pattern!r}")
    return candidates[-1]


def _find_latest_run(exp_dir: Path) -> Path | None:
    """Return the latest timestamped run directory under *exp_dir*."""
    runs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    return runs[-1] if runs else None


def _parse_experiment_name(name: str) -> dict:
    """Extract traffic and occupancy from a directory name.

    e.g. ``combination_caos_routes_parking_occupied_30`` →
    ``{'traffic': 'caos', 'occupancy': 30}``.
    """
    core = name.replace("combination_", "")
    parts = core.split("_")
    traffic = parts[0] if parts else name
    try:
        occupancy = int(parts[-1])
    except (ValueError, IndexError):
        # Name doesn't follow the expected ..._occupied_<N> convention.
        occupancy = float("nan")
        print(f"  [warn] could not parse occupancy from '{name}' — set to NaN")
    return {"traffic": traffic, "occupancy": occupancy}


# ═══════════════════════════════════════════════════════════════════
#  1. PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_core_log(path: Path) -> tuple[pd.DataFrame, datetime]:
    """Parse vanet-parking NDJSON log → (DataFrame, t0)."""
    records: list[dict] = []
    errors = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors += 1
                continue
            flat = {"ts": obj["timestamp"], "level": obj["level"]}
            flat.update(obj.get("fields", {}))
            flat["target"] = obj.get("target", "")
            records.append(flat)

    print(f"  [core]    {len(records):>6,} rows — {errors} errors")
    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    t0 = df["ts"].min()
    df["t_s"] = (df["ts"] - t0).dt.total_seconds()
    return df, t0


def parse_bridge_log(path: Path, t0_core: datetime | None) -> pd.DataFrame:
    """Parse bridge plain-text log → DataFrame."""
    records: list[dict] = []
    patterns: dict[str, re.Pattern] = {
        "assignment_won": re.compile(
            r"\[bridge\] AssignmentWon received: (.+)"),
        "slot_occupied": re.compile(
            r"\[bridge\] slot (\S+) occupied by (\S+) — total occupancy: (\d+)"),
        "spot_observed": re.compile(
            r"\[bridge\] ParkingSpotObserved — (\d+) nearby slots at \(([^,]+),([^)]+)\)"),
        "heartbeat": re.compile(
            r"\[bridge\] heartbeat — (\d+) msgs — active vehicles: (\d+)"),
        "vehicle_entry": re.compile(
            r"\[bridge\] active vehicles after entry: (\d+)"),
        "vehicle_exit": re.compile(
            r"\[bridge\] active vehicles after exit: (\d+)"),
        "remove_vehicle": re.compile(
            r"\[bridge\] RemoveVehicle → vin=(\S+) \(parked on (\S+)\)"),
    }

    errors = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            matched = False
            for evt_type, pat in patterns.items():
                m = pat.match(line)
                if not m:
                    continue
                rec: dict = {"evt": evt_type, "raw": line, "lineno": lineno}
                if evt_type == "assignment_won":
                    try:
                        d = ast.literal_eval(m.group(1))
                        rec.update(d)
                    except Exception:
                        errors += 1
                elif evt_type == "slot_occupied":
                    rec["slot_id"] = m.group(1)
                    rec["sumo_id"] = m.group(2)
                    rec["total_occupancy"] = int(m.group(3))
                elif evt_type == "spot_observed":
                    rec["n_slots"] = int(m.group(1))
                    rec["lat"] = float(m.group(2))
                    rec["lng"] = float(m.group(3))
                elif evt_type == "heartbeat":
                    rec["n_msgs"] = int(m.group(1))
                    rec["active_vehicles"] = int(m.group(2))
                elif evt_type in ("vehicle_entry", "vehicle_exit"):
                    rec["active_vehicles"] = int(m.group(1))
                elif evt_type == "remove_vehicle":
                    rec["sumo_id"] = m.group(1)
                    rec["slot_id"] = m.group(2)
                records.append(rec)
                matched = True
                break
            if not matched and line.startswith("[bridge]"):
                errors += 1

    print(f"  [bridge]  {len(records):>6,} events — {errors} unparsed")
    return pd.DataFrame(records)


def parse_van3twin_log(path: Path) -> pd.DataFrame:
    """Parse van3twin plain-text log → DataFrame."""
    records: list[dict] = []
    patterns: dict[str, re.Pattern] = {
        "vehicle_entered": re.compile(r"\[vehicle\] entered sumo_id=(\S+)"),
        "vehicle_exited":  re.compile(r"\[vehicle\] exited.*sumo_id=(\S+)"),
        "vehicle_exited_zmq": re.compile(
            r"\[vehicle\] exited ZMQ sumo_id=(\S+) u64=(\S+)"),
        "cmd_remove":      re.compile(r"\[cmd\] RemoveVehicle vehicle=(\S+)"),
        "poly_sent":       re.compile(
            r"\[poly\] sent (\d+) polygons from (.+)"),
        "zmq_bound":       re.compile(
            r"\[zmq\] (\S+) (?:socket )?(?:bound|connected) on (.+)"),
    }

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            for evt_type, pat in patterns.items():
                m = pat.match(line)
                if not m:
                    continue
                rec: dict = {"evt": evt_type, "raw": line, "lineno": lineno}
                if evt_type == "vehicle_entered":
                    rec["sumo_id"] = m.group(1)
                elif evt_type == "vehicle_exited":
                    rec["sumo_id"] = m.group(1)
                elif evt_type == "vehicle_exited_zmq":
                    rec["sumo_id"] = m.group(1)
                    rec["vehicle_id"] = m.group(2)
                elif evt_type == "cmd_remove":
                    rec["sumo_id"] = m.group(1)
                elif evt_type == "poly_sent":
                    rec["n_polygons"] = int(m.group(1))
                    rec["source"] = m.group(2).strip()
                records.append(rec)
                break

    print(f"  [van3twin] {len(records):>5,} events")
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════
#  2. CORE EVENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_core_events(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split core DataFrame into per-event DataFrames."""
    msg = df["message"]
    evts = {
        "spawned":          df[msg == "vehicle entered"].copy(),
        "vehicle_spawned":  df[msg == "vehicle spawned"].copy(),
        "actor_init":       df[msg == "VehicleActor initialized"].copy(),
        "actor_started":    df[msg == "VehicleActor started"].copy(),
        "exited":           df[msg == "vehicle exited — tasks cancelled"].copy(),
        "shutdown":         df[msg == "VehicleActor shutdown"].copy(),

        "gossip_init":      df[msg == "GossipEngine initialized"].copy(),
        "gossip_started":   df[msg == "GossipEngine loop started"].copy(),
        "gossip_shutdown":  df[msg == "GossipEngine shutdown via token"].copy(),

        "handover":         df[msg == "cell handover confirmed"].copy(),
        "handover_snapshot":df[msg == "cell handover — force snapshot on next round"].copy(),

        "timer_start":      df[msg == "assignment timer started"].copy(),
        "timer_expired":    df[msg == "timer expired — AttemptAssignment emitted"].copy(),
        "attempt":          df[msg == "slot assignment attempt"].copy(),
        "won":              df[msg == "slot assigned successfully"].copy(),
        "cancelled":        df[msg == "slot occupied — timer cancelled"].copy(),
        "ignored":          df[msg == "pending slot limit reached — slot ignored"].copy(),
        "out_of_range":     df[msg == "slot beyond max distance — not participating"].copy(),
        "gossip_change":    df[msg == "slot state received via gossip"].copy(),
    }

    print("\n  [core events]")
    for name, sub in evts.items():
        if len(sub):
            print(f"    {name:25s}: {len(sub):>6,}")
    return evts


# ═══════════════════════════════════════════════════════════════════
#  3. METRICS COMPUTATION  —  A. Core (Rust)
# ═══════════════════════════════════════════════════════════════════

def compute_core_metrics(evts: dict[str, pd.DataFrame],
                         df_full: pd.DataFrame) -> dict:
    """Compute all core (Rust) metrics, return dict of DataFrames."""
    results: dict = {}

    # ── Initial cells ──────────────────────────────────────────────
    gi = evts["gossip_init"].copy()
    if "cell" in gi.columns:
        results["initial_cells"] = gi[["t_s", "vehicle_id", "cell"]].dropna()

    # ── Ignored detail ─────────────────────────────────────────────
    ign = evts["ignored"].copy()
    if not ign.empty:
        for col in ("pending", "limit"):
            if col in ign.columns:
                ign[col] = pd.to_numeric(ign[col], errors="coerce")
        cols = [c for c in ("t_s", "vehicle_id", "slot_id", "pending", "limit")
                if c in ign.columns]
        results["ignored_detail"] = ign[cols].copy()

    # ── Handover snapshot ──────────────────────────────────────────
    hs = evts["handover_snapshot"].copy()
    if not hs.empty:
        cols = [c for c in ("t_s", "vehicle_id", "old_cell", "new_cell")
                if c in hs.columns]
        results["handover_snapshot"] = hs[cols].copy()

    # ── A1: Timer distribution ─────────────────────────────────────
    ts = evts["timer_start"].copy()
    if "timer_ms" in ts.columns:
        ts["timer_ms"] = pd.to_numeric(ts["timer_ms"], errors="coerce")

        if "wait_time_s" in ts.columns:
            ts["wait_time_s"] = pd.to_numeric(ts["wait_time_s"],
                                              errors="coerce").fillna(0)
        else:
            ts["wait_time_s"] = 0

        if "distance_m" in ts.columns:
            ts["distance_m"] = pd.to_numeric(ts["distance_m"],
                                             errors="coerce").fillna(0)
        else:
            ts["distance_m"] = 0

        cols = [c for c in ("t_s", "timer_ms", "slot_id",
                            "vehicle_id", "wait_time_s", "distance_m")
                if c in ts.columns]
        results["timer_dist"] = ts[cols].dropna()

    # ── A2: Funnel after warmup ────────────────────────────────────
    ts_a   = evts["timer_start"][evts["timer_start"]["t_s"] >= WARMUP_END]
    won_a  = evts["won"][evts["won"]["t_s"] >= WARMUP_END]
    att_a  = evts["attempt"][evts["attempt"]["t_s"] >= WARMUP_END]
    can_a  = evts["cancelled"][evts["cancelled"]["t_s"] >= WARMUP_END]
    oor_a  = evts["out_of_range"][evts["out_of_range"]["t_s"] >= WARMUP_END]
    ign_a  = evts["ignored"][evts["ignored"]["t_s"] >= WARMUP_END]

    n_timer   = len(ts_a)
    n_won     = len(won_a)
    n_attempt = len(att_a)
    n_cancel  = len(can_a)
    n_oor     = len(oor_a)
    n_ign     = len(ign_a)

    results["summary"] = pd.DataFrame([{
        "timers_started":     n_timer,
        "attempts":           n_attempt,
        "assigned":           n_won,
        "cancelled":          n_cancel,
        "out_of_range":       n_oor,
        "ignored":            n_ign,
        "success_rate_pct":      round(n_won / n_timer * 100, 2) if n_timer else 0,
        "cancellation_rate_pct": round(n_cancel / n_timer * 100, 2) if n_timer else 0,
        "oor_rate_pct":          round(n_oor / n_timer * 100, 2) if n_timer else 0,
        "active_period_s":      WARMUP_END,
    }])

    # ── A3: Assignment latency ─────────────────────────────────────
    if "elapsed_ms" in evts["won"].columns:
        won_lat = evts["won"].copy()
        won_lat["elapsed_ms"] = pd.to_numeric(won_lat["elapsed_ms"],
                                              errors="coerce")
        results["assignment_latency"] = won_lat[
            ["t_s", "slot_id", "vehicle_id", "elapsed_ms"]].dropna()

    # ── A4: Contention per slot ────────────────────────────────────
    if "slot_id" in ts_a.columns and not ts_a.empty:
        contention = (ts_a.groupby("slot_id").size()
                      .reset_index(name="n_timers"))
        if "slot_id" in won_a.columns:
            won_count = (won_a.groupby("slot_id").size()
                         .reset_index(name="n_won"))
            contention = (contention
                          .merge(won_count, on="slot_id", how="left")
                          .fillna(0))
            contention["n_won"] = contention["n_won"].astype(int)
        results["contention"] = contention

    # ── A5: Jain fairness index ────────────────────────────────────
    shut = evts["shutdown"].copy()
    if "n_slots_won" in shut.columns:
        shut["n_slots_won"] = pd.to_numeric(shut["n_slots_won"],
                                            errors="coerce").fillna(0)
        x = shut["n_slots_won"].astype(float)
        sq_sum = (x ** 2).sum()
        jain = ((x.sum() ** 2) / (len(x) * sq_sum)
                if sq_sum > 0 else float("nan"))
        results["fairness"] = shut[["vehicle_id", "n_slots_won"]].copy()
        results["fairness"]["jain_index"] = jain
        print(f"  [A5] Jain fairness index: {jain:.4f}")

    # ── A5b: Double assignment (same slot won by >1 distinct vehicle) ──
    # A slot should be won once; if it's won by multiple distinct vehicles in
    # the active window, that's a double assignment (contention not resolved).
    # This is one of the BO objective terms, so it must be in the summary.
    won_a = evts["won"][evts["won"]["t_s"] >= WARMUP_END].copy()
    if "slot_id" in won_a.columns and not won_a.empty:
        if "vehicle_id" in won_a.columns:
            winners = (won_a.groupby("slot_id")["vehicle_id"]
                       .nunique().reset_index(name="n_winners"))
        else:
            winners = (won_a.groupby("slot_id").size()
                       .reset_index(name="n_winners"))
        n_slots_won = len(winners)
        n_double    = int((winners["n_winners"] > 1).sum())
        results["double_assignment"] = pd.DataFrame([{
            "n_slots_assigned": n_slots_won,
            "n_double":         n_double,
            "double_rate_pct":  round(n_double / n_slots_won * 100, 2)
                                if n_slots_won else 0.0,
        }])
        print(f"  [A5b] double assignment: {n_double}/{n_slots_won} slots "
              f"({(n_double/n_slots_won*100) if n_slots_won else 0:.1f}%)")

    # ── A6: Propagation latency ────────────────────────────────────
    won_df  = evts["won"].copy()
    gos_occ = (evts["gossip_change"][
        evts["gossip_change"].get("new_status", pd.Series(dtype=str)) == "Occupied"
    ].copy() if "new_status" in evts["gossip_change"].columns else pd.DataFrame())

    prop_rows: list[dict] = []
    if not won_df.empty and not gos_occ.empty:
        won_df["slot_id"]  = won_df["slot_id"].astype(str)
        gos_occ["slot_id"] = gos_occ["slot_id"].astype(str)
        common = set(won_df["slot_id"]) & set(gos_occ["slot_id"])
        for sid in common:
            # Use EVERY win of this slot, not just the first — slots can turn
            # over multiple times in a run, and iloc[0] would discard the rest.
            slot_wins = won_df[won_df["slot_id"] == sid].sort_values("t_s")
            slot_gos  = gos_occ[gos_occ["slot_id"] == sid]
            for _, w in slot_wins.iterrows():
                others = slot_gos[
                    (slot_gos["t_s"] >= w["t_s"])
                    & (slot_gos["vehicle_id"] != w["vehicle_id"])
                ]
                if not others.empty:
                    prop_rows.append({
                        "slot_id":         sid,
                        "t_won_s":         w["t_s"],
                        "prop_latency_ms": (others["t_s"].min() - w["t_s"]) * 1000,
                    })
    results["propagation"] = pd.DataFrame(prop_rows)
    if prop_rows:
        p = pd.DataFrame(prop_rows)["prop_latency_ms"]
        print(f"  [A6] propagation latency — p50: {p.median():.0f} ms, "
              f"p95: {p.quantile(0.95):.0f} ms")

    return results


# ═══════════════════════════════════════════════════════════════════
#  3. METRICS COMPUTATION  —  B. Bridge  /  C. Van3twin
# ═══════════════════════════════════════════════════════════════════

def compute_bridge_metrics(df_bridge: pd.DataFrame) -> dict:
    """Compute bridge metrics."""
    results: dict = {}

    occ = df_bridge[df_bridge["evt"] == "slot_occupied"].copy()
    if not occ.empty:
        results["occupancy_timeline"] = occ[
            ["lineno", "slot_id", "sumo_id", "total_occupancy"]].copy()
        print(f"  [B1] slots occupied (ground truth): {len(occ)}")
        print(f"       final occupancy: {occ['total_occupancy'].max()}")

        slots_per_veh = (occ.groupby("sumo_id")["slot_id"]
                         .count()
                         .reset_index(name="n_slots_parked"))
        results["slots_per_vehicle_gt"] = slots_per_veh
        print(f"  [B2] parked vehicles: {len(slots_per_veh)}")
        print(f"       avg slots/vehicle: {slots_per_veh['n_slots_parked'].mean():.2f}")

    aw = df_bridge[df_bridge["evt"] == "assignment_won"].copy()
    if not aw.empty:
        results["assignment_won_bridge"] = aw[
            ["slot_id", "slot_lat", "slot_lng", "vehicle_id"]].copy()
        print(f"  [B3] AssignmentWon (bridge): {len(aw)}")

    sp = df_bridge[df_bridge["evt"] == "spot_observed"].copy()
    if not sp.empty:
        sp["n_slots"] = pd.to_numeric(sp["n_slots"], errors="coerce")
        results["spots_observed"] = sp[["n_slots", "lat", "lng"]].dropna()
        print(f"  [B4] ParkingSpotObserved — mean={sp['n_slots'].mean():.1f}, "
              f"max={sp['n_slots'].max()}")

    rv = df_bridge[df_bridge["evt"] == "remove_vehicle"].copy()
    if not rv.empty:
        results["parked_vehicles"] = rv[["sumo_id", "slot_id"]].copy()
        print(f"  [B5] removed vehicles (parked): {len(rv)}")

    hb = df_bridge[df_bridge["evt"] == "heartbeat"].copy()
    if not hb.empty:
        hb["active_vehicles"] = pd.to_numeric(hb["active_vehicles"],
                                              errors="coerce")
        hb["n_msgs"] = pd.to_numeric(hb["n_msgs"], errors="coerce")
        results["active_vehicles_timeline"] = hb[
            ["lineno", "active_vehicles", "n_msgs"]].dropna()

    return results


def compute_van3twin_metrics(df_van3: pd.DataFrame) -> dict:
    """Compute van3twin metrics."""
    results: dict = {}

    entered = df_van3[df_van3["evt"] == "vehicle_entered"]["sumo_id"].tolist()
    exited  = df_van3[df_van3["evt"] == "vehicle_exited"]["sumo_id"].tolist()
    removed = df_van3[df_van3["evt"] == "cmd_remove"]["sumo_id"].tolist()

    ne = len(entered)
    nr = len(removed)
    results["sumo_summary"] = pd.DataFrame([{
        "vehicles_entered":    ne,
        "vehicles_exited":     len(exited),
        "vehicles_parked":     nr,
        "vehicles_not_parked": len(set(entered) - set(removed)),
        "park_rate_pct": round(nr / ne * 100, 2) if ne else 0,
    }])
    print(f"  [C1] entered: {ne}, parked: {nr}, "
          f"park rate: {nr/ne*100:.1f}%" if ne else "  [C1] no vehicles entered")

    zmq_exit = df_van3[df_van3["evt"] == "vehicle_exited_zmq"].copy()
    if not zmq_exit.empty:
        results["sumo_to_u64"] = zmq_exit[["sumo_id", "vehicle_id"]].dropna()
        print(f"  [C2] sumo→u64 mapping: {len(zmq_exit)} pairs")

    poly = df_van3[df_van3["evt"] == "poly_sent"].copy()
    if not poly.empty:
        poly["n_polygons"] = pd.to_numeric(poly["n_polygons"], errors="coerce")
        results["polygons"] = poly[["n_polygons", "source"]].dropna()
        print(f"  [C3] total polygons sent: {poly['n_polygons'].sum()}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  4. PLOTTING  —  A. Core
# ═══════════════════════════════════════════════════════════════════

def plot_timer_distribution(timer_dist: pd.DataFrame, out: Path):
    """A1 — Timer backoff distribution."""
    if timer_dist.empty or "timer_ms" not in timer_dist.columns:
        return
    el = timer_dist["timer_ms"]
    if el.isna().all():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(el, bins=60, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Timer backoff (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"A1 — Timer backoff distribution  "
                 f"(median={el.median():.0f}ms)")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}s"))
    fig.tight_layout()
    fig.savefig(out / "A1_timer_distribution.png")
    plt.close(fig)
    print("    [fig] A1_timer_distribution.png")


def plot_assignment_funnel(summary: pd.DataFrame, out: Path):
    """A2 — Funnel bar chart."""
    if summary.empty:
        return
    row    = summary.iloc[0]
    labels = ["Timers started", "Attempts", "Slots assigned"]
    values = [row["timers_started"], row["attempts"], row["assigned"]]
    colors = ["#4c72b0", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of events")
    ax.set_title(f"A2 — Assignment funnel  "
                 f"(success rate: {row['success_rate_pct']:.1f}%)")
    fig.tight_layout()
    fig.savefig(out / "A2_assignment_funnel.png")
    plt.close(fig)
    print("    [fig] A2_assignment_funnel.png")


def plot_assignment_latency(latency: pd.DataFrame, out: Path):
    """A3 — Assignment latency histogram."""
    if latency.empty or "elapsed_ms" not in latency.columns:
        return
    el = latency["elapsed_ms"]
    if el.isna().all():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(el, bins=50, edgecolor="white", linewidth=0.3)
    ax.axvline(el.median(), color="red", linewidth=1,
               linestyle="--", label=f"Median {el.median():.0f}ms")
    ax.set_xlabel("Assignment latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"A3 — Assignment latency  (p50={el.median():.0f}ms, "
                 f"p95={el.quantile(0.95):.0f}ms)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "A3_assignment_latency.png")
    plt.close(fig)
    print("    [fig] A3_assignment_latency.png")


def plot_slot_contention(contention: pd.DataFrame, out: Path):
    """A4 — Slot contention distribution."""
    if contention.empty or "n_timers" not in contention.columns:
        print("    [fig] A4_slot_contention.png — skipped (empty)")
        return
    max_val = contention["n_timers"].max()
    if pd.isna(max_val) or max_val < 1:
        print("    [fig] A4_slot_contention.png — skipped (no data)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(contention["n_timers"],
            bins=range(1, int(max_val) + 2),
            edgecolor="white", linewidth=0.3, align="left")
    ax.set_xlabel("Vehicles competing per slot")
    ax.set_ylabel("Number of slots")
    ax.set_title("A4 — Slot contention")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out / "A4_slot_contention.png")
    plt.close(fig)
    print("    [fig] A4_slot_contention.png")


def plot_fairness(fairness: pd.DataFrame, out: Path):
    """A5 — Slots won per vehicle with Jain index."""
    if fairness.empty or "n_slots_won" not in fairness.columns:
        return
    x = fairness["n_slots_won"]
    if x.isna().all():
        return
    jain = fairness["jain_index"].iloc[0] if "jain_index" in fairness.columns else float("nan")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(x, bins=range(0, int(x.max()) + 2),
            edgecolor="white", linewidth=0.3, align="left")
    ax.set_xlabel("Slots assigned per vehicle")
    ax.set_ylabel("Number of vehicles")
    ax.set_title(f"A5 — Fairness  (Jain index = {jain:.4f})")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out / "A5_fairness.png")
    plt.close(fig)
    print("    [fig] A5_fairness.png")


def plot_propagation_latency(propagation: pd.DataFrame, out: Path):
    """A6 — Gossip propagation latency histogram."""
    if propagation.empty or "prop_latency_ms" not in propagation.columns:
        return
    pl = propagation["prop_latency_ms"]
    if pl.isna().all():
        return
    pl_short = pl[pl < 60_000]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pl_short / 1000, bins=30, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Propagation latency (s)")
    ax.set_ylabel("Number of slots")
    ax.set_title(f"A6 — Gossip propagation latency  "
                 f"(p50={pl.median()/1000:.1f}s, n={len(pl)})")
    fig.tight_layout()
    fig.savefig(out / "A6_propagation_latency.png")
    plt.close(fig)
    print("    [fig] A6_propagation_latency.png")


# ═══════════════════════════════════════════════════════════════════
#  4. PLOTTING  —  B. Bridge
# ═══════════════════════════════════════════════════════════════════

def plot_occupancy_timeline(occupancy: pd.DataFrame, out: Path):
    """B1 — Cumulative occupancy over time."""
    if occupancy.empty or "total_occupancy" not in occupancy.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(occupancy["lineno"], occupancy["total_occupancy"],
            linewidth=1.2, color="#2c7bb6")
    ax.fill_between(occupancy["lineno"], occupancy["total_occupancy"], alpha=0.15)
    ax.set_xlabel("Bridge event sequence")
    ax.set_ylabel("Occupied slots (total)")
    ax.set_title(f"B1 — Occupancy timeline  "
                 f"(final: {occupancy['total_occupancy'].max()})")
    fig.tight_layout()
    fig.savefig(out / "B1_occupancy_timeline.png")
    plt.close(fig)
    print("    [fig] B1_occupancy_timeline.png")


def plot_slots_per_vehicle(slots_per_veh: pd.DataFrame, out: Path):
    """B2 — Distribution of slots per vehicle (ground truth)."""
    if slots_per_veh.empty or "n_slots_parked" not in slots_per_veh.columns:
        return
    spv = slots_per_veh["n_slots_parked"]
    if spv.isna().all():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(spv, bins=range(1, int(spv.max()) + 2),
            edgecolor="white", linewidth=0.3, align="left")
    ax.set_xlabel("Slots occupied per vehicle (ground truth)")
    ax.set_ylabel("Number of vehicles")
    ax.set_title(f"B2 — Slots per vehicle  (mean={spv.mean():.2f})")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out / "B2_slots_per_vehicle_gt.png")
    plt.close(fig)
    print("    [fig] B2_slots_per_vehicle_gt.png")


def plot_spots_observed(spots: pd.DataFrame, out: Path):
    """B3 — Nearby slots observed per GPS update."""
    if spots.empty or "n_slots" not in spots.columns:
        return
    so = spots["n_slots"]
    if so.isna().all():
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(so, bins=range(0, int(so.max()) + 2),
            edgecolor="white", linewidth=0.3, align="left")
    ax.set_xlabel("Available nearby slots")
    ax.set_ylabel("Observations")
    ax.set_title(f"B3 — Parking spots observed per GPS update  "
                 f"(mean={so.mean():.1f})")
    fig.tight_layout()
    fig.savefig(out / "B3_spots_observed.png")
    plt.close(fig)
    print("    [fig] B3_spots_observed.png")


# ═══════════════════════════════════════════════════════════════════
#  4. PLOTTING  —  C. Van3twin
# ═══════════════════════════════════════════════════════════════════

def plot_vehicle_outcome(sumo_summary: pd.DataFrame, out: Path):
    """C1 — Vehicle outcome bar chart."""
    if sumo_summary.empty:
        return
    sm = sumo_summary.iloc[0]
    labels = ["Entered", "Parked", "Not parked"]
    values = [sm["vehicles_entered"], sm["vehicles_parked"],
              sm["vehicles_not_parked"]]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of vehicles")
    ax.set_title(f"C1 — SUMO vehicle outcome  "
                 f"(park rate: {sm['park_rate_pct']:.1f}%)")
    fig.tight_layout()
    fig.savefig(out / "C1_sumo_vehicle_outcome.png")
    plt.close(fig)
    print("    [fig] C1_sumo_vehicle_outcome.png")


# ═══════════════════════════════════════════════════════════════════
#  5. CSV EXPORT
# ═══════════════════════════════════════════════════════════════════

def save_csvs(core_m: dict, bridge_m: dict, van3_m: dict, out_dir: Path):
    """Save all non-empty DataFrames as CSV."""
    all_dfs = {
        "A0_initial_cells":        core_m.get("initial_cells"),
        "A0_ignored_detail":       core_m.get("ignored_detail"),
        "A0_handover_snapshot":    core_m.get("handover_snapshot"),
        "A1_timer_dist":           core_m.get("timer_dist"),
        "A2_summary":              core_m.get("summary"),
        "A3_assignment_latency":   core_m.get("assignment_latency"),
        "A4_contention":           core_m.get("contention"),
        "A5_fairness":             core_m.get("fairness"),
        "A5b_double_assignment":   core_m.get("double_assignment"),
        "A6_propagation":          core_m.get("propagation"),
        "B1_occupancy_timeline":   bridge_m.get("occupancy_timeline"),
        "B2_slots_per_vehicle_gt": bridge_m.get("slots_per_vehicle_gt"),
        "B3_spots_observed":       bridge_m.get("spots_observed"),
        "B4_assignment_won_bridge":bridge_m.get("assignment_won_bridge"),
        "B5_parked_vehicles":      bridge_m.get("parked_vehicles"),
        "B6_active_timeline":      bridge_m.get("active_vehicles_timeline"),
        "C1_sumo_summary":         van3_m.get("sumo_summary"),
        "C2_sumo_to_u64":          van3_m.get("sumo_to_u64"),
        "C3_polygons":             van3_m.get("polygons"),
    }
    for name, df in all_dfs.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            p = out_dir / f"{name}.csv"
            df.to_csv(p, index=False)
            print(f"    [csv] {p.name}  ({len(df):,} rows)")


# ═══════════════════════════════════════════════════════════════════
#  6. SINGLE-EXPERIMENT PIPELINE
# ═══════════════════════════════════════════════════════════════════

def process_single_experiment(exp_dir: Path, out_base: Path) -> dict | None:
    """Parse, compute, save CSVs/figures for *one* experiment.

    Returns a summary dict for aggregation, or ``None`` on failure.
    """
    exp_name = exp_dir.name
    out_dir  = out_base / exp_name
    fig_dir  = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _find_latest_run(exp_dir)
    if run_dir is None:
        print(f"  SKIP  {exp_name}  —  no run directories")
        return None

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {exp_name}")
    print(f"  run: {run_dir.name}")
    print(f"{sep}")

    try:
        log_core   = _find_log(run_dir, "core",    "vanet-parking.log.*")
        log_bridge = _find_log(run_dir, "bridge",  "bridge.log.*")
        log_van3   = _find_log(run_dir, "van3twin", "van3twin.log.*")
    except FileNotFoundError as e:
        print(f"  SKIP  {exp_name}  —  {e}")
        return None

    # Parse
    df_core, _   = parse_core_log(log_core)
    df_bridge    = parse_bridge_log(log_bridge, None)
    df_van3      = parse_van3twin_log(log_van3)

    # Events & metrics
    evts    = extract_core_events(df_core)
    core_m  = compute_core_metrics(evts, df_core)
    bridge_m = compute_bridge_metrics(df_bridge)
    van3_m  = compute_van3twin_metrics(df_van3)

    # CSV
    print("  --- CSV ---")
    save_csvs(core_m, bridge_m, van3_m, out_dir)

    # Plots
    print("  --- Figures ---")
    for key, data, fn in (
        ("timer_dist",         core_m,   plot_timer_distribution),
        ("summary",            core_m,   plot_assignment_funnel),
        ("assignment_latency", core_m,   plot_assignment_latency),
        ("contention",         core_m,   plot_slot_contention),
        ("fairness",           core_m,   plot_fairness),
        ("propagation",        core_m,   plot_propagation_latency),
        ("occupancy_timeline", bridge_m, plot_occupancy_timeline),
        ("slots_per_vehicle_gt", bridge_m, plot_slots_per_vehicle),
        ("spots_observed",     bridge_m, plot_spots_observed),
        ("sumo_summary",       van3_m,   plot_vehicle_outcome),
    ):
        if key in data and isinstance(data[key], pd.DataFrame) and not data[key].empty:
            fn(data[key], fig_dir)

    # ── Collect summary ────────────────────────────────────────────
    meta = _parse_experiment_name(exp_name)
    summary: dict = {"experiment": exp_name, **meta}

    if "summary" in core_m:
        s = core_m["summary"].iloc[0]
        for col in s.index:
            summary[f"A2_{col}"] = s[col]
        # Expose timers_started explicitly as n_timer for reliability checks:
        # success_rate from few timers (low-contention scenarios) is noisy.
        summary["A2_n_timer"] = int(s.get("timers_started", 0))

    if "double_assignment" in core_m and not core_m["double_assignment"].empty:
        d = core_m["double_assignment"].iloc[0]
        summary["A5b_double_rate_pct"] = d["double_rate_pct"]
        summary["A5b_n_double"]        = int(d["n_double"])
        summary["A5b_n_slots"]         = int(d["n_slots_assigned"])

    if "fairness" in core_m and "jain_index" in core_m["fairness"].columns:
        summary["A5_jain"] = core_m["fairness"]["jain_index"].iloc[0]

    if "propagation" in core_m and not core_m["propagation"].empty:
        p = core_m["propagation"]["prop_latency_ms"]
        summary["A6_p50_ms"] = p.median()
        summary["A6_p95_ms"] = p.quantile(0.95)
        summary["A6_n"]      = len(p)

    if "contention" in core_m:
        c = core_m["contention"]
        if "n_timers" in c.columns:
            summary["A4_mean"]    = c["n_timers"].mean()
            summary["A4_max"]     = c["n_timers"].max()
            summary["A4_n_slots"] = len(c)
        else:
            summary["A4_mean"]    = float("nan")
            summary["A4_max"]     = float("nan")
            summary["A4_n_slots"] = 0

    if "assignment_latency" in core_m:
        el = core_m["assignment_latency"]["elapsed_ms"]
        if not el.empty:
            summary["A3_p50_ms"] = el.median()
            summary["A3_p95_ms"] = el.quantile(0.95)
        else:
            summary["A3_p50_ms"] = float("nan")
            summary["A3_p95_ms"] = float("nan")

    if "sumo_summary" in van3_m:
        s = van3_m["sumo_summary"].iloc[0]
        summary["C1_entered"]   = int(s["vehicles_entered"])
        summary["C1_parked"]    = int(s["vehicles_parked"])
        summary["C1_park_rate"] = s["park_rate_pct"]

    if "timer_dist" in core_m:
        t = core_m["timer_dist"]["timer_ms"]
        summary["A1_p50_ms"]  = t.median()
        summary["A1_mean_ms"] = t.mean()

    return summary


# ═══════════════════════════════════════════════════════════════════
#  7. BATCH  +  AGGREGATION
# ═══════════════════════════════════════════════════════════════════

def run_batch(log_base: Path, out_base: Path):
    """Process all experiment directories, then aggregate results."""
    experiments = sorted(
        d for d in log_base.iterdir()
        if d.is_dir() and d.name != "general_slurm"
    )
    print(f"Found {len(experiments)} experiment directories\n")

    all_summaries: list[dict] = []
    for exp_dir in experiments:
        try:
            s = process_single_experiment(exp_dir, out_base)
            if s is not None:
                all_summaries.append(s)
        except Exception as e:
            print(f"  ERROR  {exp_dir.name}  —  {e}")
            import traceback
            traceback.print_exc()

    if not all_summaries:
        print("\nNo experiments processed successfully.")
        return

    # ── Save raw summary ───────────────────────────────────────────
    df_summary = pd.DataFrame(all_summaries)
    summary_path = out_base / "summary_all_experiments.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"  Aggregated {len(df_summary)} experiments  →  {summary_path}")
    print(f"{'=' * 60}")

    # ── Reliability warning: success_rate from few timers is noisy ──
    if "A2_n_timer" in df_summary.columns:
        LOW_SAMPLE = 30  # below this, success_rate% is statistically weak
        weak = df_summary[df_summary["A2_n_timer"] < LOW_SAMPLE]
        if not weak.empty:
            print(f"\n  [reliability] {len(weak)} experiment(s) have < {LOW_SAMPLE} "
                  f"timers — their success_rate is statistically unreliable:")
            for _, r in weak.iterrows():
                print(f"    {r['experiment']:50s}  n_timer={int(r['A2_n_timer'])}")

    # ── Aggregate: pivot tables + comparison charts ────────────────
    agg_dir  = out_base / "aggregated"
    fig_agg  = agg_dir / "figures"
    piv_dir  = agg_dir / "pivots"
    agg_dir.mkdir(parents=True, exist_ok=True)
    fig_agg.mkdir(parents=True, exist_ok=True)
    piv_dir.mkdir(parents=True, exist_ok=True)

    metric_defs = [
        ("A2_success_rate_pct",  "Success rate (%)"),
        ("A2_n_timer",           "Timers started (sample size for success rate)"),
        ("A5_jain",              "Jain fairness index"),
        ("A5b_double_rate_pct",  "Double assignment rate (%)"),
        ("A4_mean",              "Mean contention (vehicles/slot)"),
        ("A3_p50_ms",            "Assignment latency p50 (ms)"),
        ("A3_p95_ms",            "Assignment latency p95 (ms)"),
        ("A6_p50_ms",            "Propagation latency p50 (ms)"),
        ("A6_p95_ms",            "Propagation latency p95 (ms)"),
        ("C1_park_rate",         "Park rate (%)"),
        ("A1_p50_ms",            "Timer backoff p50 (ms)"),
    ]

    print(f"\n{'=' * 60}")
    print("  AGGREGATED SUMMARY  (traffic × occupancy)")
    print(f"{'=' * 60}")

    for metric, label in metric_defs:
        if metric not in df_summary.columns:
            continue
        try:
            pivot = df_summary.pivot_table(
                index="traffic", columns="occupancy",
                values=metric, aggfunc="first",
            )
            pivot.index.name = None
            print(f"\n  {label}  [{metric}]")
            print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))
            pivot.to_csv(piv_dir / f"{metric}_pivot.csv")
        except Exception as e:
            print(f"  Skipping {metric}: {e}")

    # ── Comparison line charts ─────────────────────────────────────
    chart_defs = [
        ("A2_success_rate_pct", "Success rate by traffic and occupancy",
         "Success rate (%)"),
        ("A5_jain", "Jain fairness index by traffic and occupancy",
         "Jain index"),
        ("A5b_double_rate_pct", "Double assignment rate by traffic and occupancy",
         "Double assignment (%)"),
        ("A4_mean", "Mean slot contention by traffic and occupancy",
         "Mean vehicles/slot"),
        ("A3_p50_ms", "Median assignment latency by traffic and occupancy",
         "Latency p50 (ms)"),
        ("C1_park_rate", "Park rate by traffic and occupancy",
         "Park rate (%)"),
        ("A1_p50_ms", "Median timer backoff by traffic and occupancy",
         "Timer backoff p50 (ms)"),
    ]

    for metric, title, ylabel in chart_defs:
        if metric not in df_summary.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for traffic in sorted(df_summary["traffic"].unique()):
            subset = (df_summary[df_summary["traffic"] == traffic]
                      .sort_values("occupancy"))
            ax.plot(subset["occupancy"].astype(str), subset[metric],
                    marker="o", label=traffic)
        ax.set_xlabel("Parking occupancy (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_agg / f"{metric}_comparison.png")
        plt.close(fig)
        print(f"  [fig] {metric}_comparison.png")

    print(f"\n{'=' * 60}")
    print(f"  Done — pivot tables → {piv_dir.resolve()}")
    print(f"  Done — comparison charts → {fig_agg.resolve()}")
    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════
#  8. CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VANET-Parking Post-Processor — three-source analysis",
    )
    parser.add_argument(
        "--exp", "-e", type=str, default=None,
        help="Process a single experiment by name (e.g. "
             "combination_caos_routes_parking_occupied_30). "
             "Omit to process all experiments.",
    )
    parser.add_argument(
        "--log-base", type=str, default=str(LOG_BASE),
        help=f"Log directory root (default: {LOG_BASE})",
    )
    parser.add_argument(
        "--out", "-o", type=str, default=str(OUT_DIR),
        help=f"Output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    log_base = Path(args.log_base)
    out_base = Path(args.out)

    if args.exp:
        exp_dir = log_base / args.exp
        if not exp_dir.is_dir():
            print(f"ERROR: experiment directory not found: {exp_dir}")
            sys.exit(1)
        s = process_single_experiment(exp_dir, out_base)
        if s:
            print(f"\n{'=' * 60}")
            print(f"  Summary for {args.exp}")
            for k, v in s.items():
                print(f"    {k:25s}: {v}")
            print(f"{'=' * 60}\n")
    else:
        run_batch(log_base, out_base)


if __name__ == "__main__":
    main()