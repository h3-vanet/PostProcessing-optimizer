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
    "A2_assigned_uniq",
    "A2_attempts",
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

WINNERS_ATTEMPTS = {
    "post_simtime_ll": (50, 75),  # (A2_assigned_uniq, A2_attempts) -> ratio 1.5
    "post_simtime_ll_k2": (55, 66),  # ratio 1.2
}

PLR = {"post_simtime_ll": 5.0, "post_simtime_ll_k2": 4.0, "post_simtime_br": 3.0, "post_ll": 6.0, "post_br": 3.5}
TX_BYTES = {"post_simtime_ll": 1000, "post_simtime_ll_k2": 1200, "post_simtime_br": 800, "post_ll": 1100, "post_br": 850}

# leaderless caos rows are invalid (25/25 in the real campaign); model it here
# for post_simtime_ll only, to exercise validity filtering + run_validity table.
INVALID_CELLS = {"post_simtime_ll": {"caos"}}

E1_BASE = {"post_simtime_br": 40}


def rows_for_series(series: str) -> list[dict]:
    rows = []
    base = BASE[series]
    is_broker = series in ("post_simtime_br", "post_br")
    for density in DENSITIES:
        for occ in OCCUPANCIES:
            occ_base = base[density] if occ == 30 else base[density] - 10
            for seed in SEEDS:
                park_rate = occ_base + (2 if seed == 2 else 0)
                nr_sim_t_reached = 100 if density in INVALID_CELLS.get(series, set()) else 180
                row = {
                    "experiment": f"combination_{density}_occupied_{occ}_seed{seed}",
                    "traffic": density,
                    "occupancy": occ,
                    "seed": seed,
                    "nr_sim_t_reached": nr_sim_t_reached,
                    "C1_entered": 100,
                    "C1_parked": park_rate,
                    "C1_park_rate": park_rate,
                    "A2_assigned_uniq": "",
                    "A2_attempts": "",
                    "nr_plr_pct": PLR[series],
                    "D1_cp_tx_bytes_per_vehicle_mean": TX_BYTES[series],
                }
                if series in WINNERS_ATTEMPTS:
                    winners, attempts = WINNERS_ATTEMPTS[series]
                    row["A2_assigned_uniq"] = winners
                    row["A2_attempts"] = attempts
                if is_broker:
                    e1_base = 40 if occ == 30 else 30
                    row["E1_rsu_coverage_pct"] = e1_base + (2 if seed == 2 else 0)
                rows.append(row)
    return rows


def write_series(series: str, out_dir: Path) -> None:
    is_broker = series in ("post_simtime_br", "post_br")
    columns = BROKER_COLUMNS if is_broker else COLUMNS
    rows = rows_for_series(series)
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


if __name__ == "__main__":
    main()
