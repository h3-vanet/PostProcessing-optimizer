"""One-off generator for tests/fixtures/*. Not run by pytest; re-run manually
if the synthetic dataset needs to change:

    python3 tests/generate_fixtures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"

DENSITIES = ["minimo", "normale", "trafficato", "caos"]
OCCUPANCIES = [30, 95]
SEEDS = [1, 2]

COLUMNS = [
    "experiment",
    "traffic",
    "occupancy",
    "seed",
    "nr_sim_t_reached",
    "C1_entered",
    "C1_parked",
    "C1_park_rate",
    "A2_vehicles_with_slot",
    "A2_assigned_uniq",
    "A1_p50_ms",
    "A3_p50_ms",
    "A3_p95_ms",
    "A5_jain",
    "A5b_overlap_rate_pct",
    "A5b_overlap_rate_pct_d010",
    "nr_plr_pct",
    "D1_cp_tx_bytes_per_vehicle_mean",
]
BROKER_COLUMNS = COLUMNS + ["E1_rsu_coverage_pct"]

# base park rate (seed1, occ30) per density, for each series; seed2 = base+2,
# occ95 = base - 10 (seed1), +2 for seed2 as well.
BASE = {
    "post_simtime_ll": {"minimo": 90, "normale": 85, "trafficato": 75, "caos": 60},
    "post_simtime_ll_k2": {"minimo": 94, "normale": 90, "trafficato": 82, "caos": 70},
    "post_simtime_br": {"minimo": 97, "normale": 95, "trafficato": 90, "caos": 82},
    "post_ll": {"minimo": 80, "normale": 75, "trafficato": 65, "caos": 50},
    "post_br": {"minimo": 92, "normale": 90, "trafficato": 85, "caos": 77},
}

# (winners=A2_vehicles_with_slot, claims=A2_assigned_uniq), constant across
# density/occupancy/seed for simplicity: ratio = mean(claims)/mean(winners).
WINNERS_CLAIMS = {
    "post_simtime_ll": (40, 50),  # ratio 1.25
    "post_simtime_ll_k2": (50, 55),  # ratio 1.10
}

PLR = {"post_simtime_ll": 5.0, "post_simtime_ll_k2": 4.0, "post_simtime_br": 3.0, "post_ll": 6.0, "post_br": 3.5}
TX_BYTES = {"post_simtime_ll": 1000, "post_simtime_ll_k2": 1200, "post_simtime_br": 800, "post_ll": 1100, "post_br": 850}

# leaderless caos rows are invalid (25/25 in the real campaign); model it here
# for post_simtime_ll only, to exercise validity filtering + run_validity table.
INVALID_CELLS = {"post_simtime_ll": {"caos"}}

E1_BASE = {"post_simtime_br": 40}


def rows_for_series(series: str, invalid_densities: set | None = None) -> list[dict]:
    rows = []
    base = BASE[series]
    is_broker = series in ("post_simtime_br", "post_br")
    if invalid_densities is None:
        invalid_densities = INVALID_CELLS.get(series, set())
    for density in DENSITIES:
        for occ in OCCUPANCIES:
            occ_base = base[density] if occ == 30 else base[density] - 10
            for seed in SEEDS:
                park_rate = occ_base + (2 if seed == 2 else 0)
                nr_sim_t_reached = 100 if density in invalid_densities else 180
                row = {
                    "experiment": f"combination_{density}_occupied_{occ}_seed{seed}",
                    "traffic": density,
                    "occupancy": occ,
                    "seed": seed,
                    "nr_sim_t_reached": nr_sim_t_reached,
                    "C1_entered": 100,
                    "C1_parked": park_rate,
                    "C1_park_rate": park_rate,
                    "A2_vehicles_with_slot": "",
                    "A2_assigned_uniq": "",
                    "A1_p50_ms": 100.0,
                    "A3_p50_ms": 200.0,
                    "A3_p95_ms": 300.0,
                    "A5_jain": 0.70,
                    "A5b_overlap_rate_pct": 9.0,
                    "A5b_overlap_rate_pct_d010": 1.0,
                    "nr_plr_pct": PLR[series],
                    "D1_cp_tx_bytes_per_vehicle_mean": TX_BYTES[series],
                }
                if series in WINNERS_CLAIMS:
                    winners, claims = WINNERS_CLAIMS[series]
                    row["A2_vehicles_with_slot"] = winners
                    row["A2_assigned_uniq"] = claims
                if is_broker:
                    e1_base = 40 if occ == 30 else 30
                    row["E1_rsu_coverage_pct"] = e1_base + (2 if seed == 2 else 0)
                rows.append(row)
    return rows


def write_series(series: str, out_dir: Path, invalid_densities: set | None = None) -> None:
    is_broker = series in ("post_simtime_br", "post_br")
    columns = BROKER_COLUMNS if is_broker else COLUMNS
    rows = rows_for_series(series, invalid_densities=invalid_densities)
    series_dir = out_dir / series
    series_dir.mkdir(parents=True, exist_ok=True)
    with (series_dir / "summary_all_experiments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


CONFIG_TOML = {
    "campaign_simtime_ll": """\
[h3]
cluster_resolution = 10
spot_resolution = 14

[gossip]
gossip_interval_ms = 500
neighbor_k = 1

[assignment.backoff]
max_distance_m = 500.0

[crdt]
slot_ttl_secs = 3600

[coordinator]
backend = "leaderless"
""",
    "campaign_simtime_ll_k2": """\
[h3]
cluster_resolution = 10
spot_resolution = 14

[gossip]
gossip_interval_ms = 500
neighbor_k = 2

[assignment.backoff]
max_distance_m = 500.0

[crdt]
slot_ttl_secs = 3600

[coordinator]
backend = "leaderless"
""",
    "campaign_simtime_br": """\
[h3]
cluster_resolution = 10
spot_resolution = 14

[gossip]
gossip_interval_ms = 500
neighbor_k = 1

[assignment.backoff]
max_distance_m = 500.0

[crdt]
slot_ttl_secs = 3600

[coordinator]
backend = "centralized"

[broker]
state_log_interval_secs = 45
claim_ttl_sim_s = 120.0
lease_sim_s = 8.0
uplink_timeout_sim_s = 0.5
""",
}


def write_configs(out_dir: Path) -> None:
    for tree, content in CONFIG_TOML.items():
        run_dir = out_dir / tree / "scenario1" / "seed1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_used.toml").write_text(content)


def write_mismatch_configs(out_dir: Path) -> None:
    tree_dir = out_dir / "campaign_simtime_ll"
    (tree_dir / "scenario1" / "seed1").mkdir(parents=True, exist_ok=True)
    (tree_dir / "scenario1" / "seed1" / "config_used.toml").write_text(CONFIG_TOML["campaign_simtime_ll"])
    (tree_dir / "scenario2" / "seed1").mkdir(parents=True, exist_ok=True)
    mismatched = CONFIG_TOML["campaign_simtime_ll"].replace(
        "gossip_interval_ms = 500", "gossip_interval_ms = 600"
    )
    (tree_dir / "scenario2" / "seed1" / "config_used.toml").write_text(mismatched)


def write_effective_mismatch_configs(out_dir: Path) -> None:
    """One run leaves broker.rtt_ms absent (default 0), another sets it
    explicitly to a different value -> effective-value mismatch."""
    tree_dir = out_dir / "campaign_simtime_br"
    (tree_dir / "scenario1" / "seed1").mkdir(parents=True, exist_ok=True)
    (tree_dir / "scenario1" / "seed1" / "config_used.toml").write_text(CONFIG_TOML["campaign_simtime_br"])
    (tree_dir / "scenario2" / "seed1").mkdir(parents=True, exist_ok=True)
    explicit_rtt = CONFIG_TOML["campaign_simtime_br"].replace(
        "[broker]\n", "[broker]\nrtt_ms = 10\n"
    )
    (tree_dir / "scenario2" / "seed1" / "config_used.toml").write_text(explicit_rtt)


def write_broker_suffix_metrics(out_dir: Path, suffix: str) -> None:
    write_series("post_simtime_br", out_dir)
    src = out_dir / "post_simtime_br"
    dst = out_dir / f"post_simtime_br{suffix}"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary_all_experiments.csv").write_text(
        (src / "summary_all_experiments.csv").read_text()
    )
    import shutil

    shutil.rmtree(src)


def write_broker_suffix_full_metrics(out_dir: Path, suffix: str) -> None:
    """Broker series suffixed, but a leaderless series (post_simtime_ll) and
    the pre-fix broker series (post_br) present UNSUFFIXED, to prove the
    suffix is never applied to them even though it's set."""
    write_series("post_simtime_br", out_dir)
    src = out_dir / "post_simtime_br"
    dst = out_dir / f"post_simtime_br{suffix}"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary_all_experiments.csv").write_text(
        (src / "summary_all_experiments.csv").read_text()
    )
    import shutil

    shutil.rmtree(src)

    write_series("post_simtime_ll", out_dir)
    write_series("post_br", out_dir)


def write_broker_suffix_configs(out_dir: Path, suffix: str) -> None:
    """campaign_simtime_br<suffix> with rtt_ms=99 explicit; the UNSUFFIXED
    campaign_simtime_br tree has a deliberately different rtt_ms=5, so a test
    reading 99 proves the suffixed tree was used (not a silent fallback).
    campaign_simtime_ll has no suffixed counterpart at all, proving the
    suffix is never applied to a leaderless tree."""
    suffixed_br = CONFIG_TOML["campaign_simtime_br"].replace(
        "[broker]\n", "[broker]\nrtt_ms = 99\n"
    )
    run_dir = out_dir / f"campaign_simtime_br{suffix}" / "scenario1" / "seed1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.toml").write_text(suffixed_br)

    unsuffixed_br = CONFIG_TOML["campaign_simtime_br"].replace(
        "[broker]\n", "[broker]\nrtt_ms = 5\n"
    )
    run_dir = out_dir / "campaign_simtime_br" / "scenario1" / "seed1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.toml").write_text(unsuffixed_br)

    run_dir = out_dir / "campaign_simtime_ll" / "scenario1" / "seed1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.toml").write_text(CONFIG_TOML["campaign_simtime_ll"])


def write_caos_invalid_metrics(out_dir: Path) -> None:
    """Realistic production scenario: BOTH GeoGrid arms (k=1 and k=2) run
    leaderless, so caos is invalid for both, not just k=1 -> gap_to_broker
    and winners_parked rows for caos should be entirely dropped."""
    write_series("post_simtime_ll", out_dir, invalid_densities={"caos"})
    write_series("post_simtime_ll_k2", out_dir, invalid_densities={"caos"})
    write_series("post_simtime_br", out_dir, invalid_densities=set())


def main() -> None:
    metrics_dir = FIXTURES / "metrics"
    for series in BASE:
        write_series(series, metrics_dir)

    metrics_missing_dir = FIXTURES / "metrics_missing_series"
    for series in ("post_simtime_br", "post_simtime_ll", "post_ll", "post_br"):
        write_series(series, metrics_missing_dir)

    configs_dir = FIXTURES / "configs"
    write_configs(configs_dir)

    configs_mismatch_dir = FIXTURES / "configs_mismatch"
    write_mismatch_configs(configs_mismatch_dir)

    configs_effective_mismatch_dir = FIXTURES / "configs_effective_mismatch"
    write_effective_mismatch_configs(configs_effective_mismatch_dir)

    metrics_broker_suffix_dir = FIXTURES / "metrics_broker_suffix"
    write_broker_suffix_metrics(metrics_broker_suffix_dir, "_v2")

    metrics_broker_suffix_full_dir = FIXTURES / "metrics_broker_suffix_full"
    write_broker_suffix_full_metrics(metrics_broker_suffix_full_dir, "_v2")

    configs_broker_suffix_dir = FIXTURES / "configs_broker_suffix"
    write_broker_suffix_configs(configs_broker_suffix_dir, "_v2")

    metrics_caos_invalid_dir = FIXTURES / "metrics_caos_invalid"
    write_caos_invalid_metrics(metrics_caos_invalid_dir)


if __name__ == "__main__":
    main()
