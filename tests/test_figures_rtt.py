import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paper_figures as pf
import rtt_compare_macros as rc
from paper_tables import NumberRegistry

COLS = ["traffic", "occupancy", "seed", "nr_sim_t_reached", "C1_park_rate"]


def write_summary(metrics_dir: Path, series: str, rows) -> None:
    d = metrics_dir / series
    d.mkdir(parents=True, exist_ok=True)
    with (d / "summary_all_experiments.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _rows(base: int):
    out = []
    for traffic in ("minimo", "normale", "trafficato"):
        for occ in (30, 95):
            for seed in (1, 2):
                out.append({
                    "traffic": traffic,
                    "occupancy": occ,
                    "seed": seed,
                    "nr_sim_t_reached": 180,
                    "C1_park_rate": base + (2 if seed == 2 else 0),
                })
    return out


def test_rtt_compare_table_and_macros(tmp_path):
    metrics = tmp_path / "metrics"
    write_summary(metrics, "post_simtime_br", _rows(90))       # RTT 0
    write_summary(metrics, "post_simtime_br_rtt50", _rows(94))  # RTT 50
    rows = rc.compare(metrics)
    numbers = NumberRegistry()
    latex = rc.build(rows, numbers)
    assert "\\label{tab:rtt_compare}" in latex
    assert "Broker RTT 0 (\\%)" in latex
    # RTT50 - RTT0 = 94 - 90 = 4 (occ30/95 both shift equally)
    assert numbers.as_dict()["rttDiffSparse"] == "+4.0"
    assert numbers.as_dict()["rttZeroSparse"] == "91.0"
    assert numbers.as_dict()["rttFiftySparse"] == "95.0"


def test_park_rate_by_density(tmp_path):
    metrics = tmp_path / "metrics"
    write_summary(metrics, "post_simtime_ll", _rows(70))
    write_summary(metrics, "post_simtime_ll_k2", _rows(80))
    write_summary(metrics, "post_simtime_br_rtt50", _rows(95))
    data = pf.park_rate_by_density(metrics)
    assert set(data) == {"GeoGrid k=1", "GeoGrid k=2", "Broker"}
    assert data["GeoGrid k=2"]["sparse"][0] == 81.0
    assert data["Broker"]["nominal"][0] == 96.0
