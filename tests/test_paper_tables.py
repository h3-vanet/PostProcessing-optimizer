import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paper_tables as pt

FIXTURES = Path(__file__).parent / "fixtures"
METRICS_DIR = FIXTURES / "metrics"
CONFIGS_DIR = FIXTURES / "configs"
METRICS_MISSING_DIR = FIXTURES / "metrics_missing_series"
CONFIGS_MISMATCH_DIR = FIXTURES / "configs_mismatch"


def read(path: Path) -> str:
    return path.read_text()


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("out")
    report = pt.run([METRICS_DIR], CONFIGS_DIR, out_dir)
    return out_dir, report


# --------------------------------------------------------------------------- #
# Validity filter
# --------------------------------------------------------------------------- #


def test_validity_filter_marks_leaderless_caos_invalid():
    df = pt.discover_series([METRICS_DIR], pt.Report())["post_simtime_ll"]
    caos = df[df["density"] == "caos"]
    assert not caos["valid"].any()
    other = df[df["density"] != "caos"]
    assert other["valid"].all()


def test_run_validity_table(full_run):
    out_dir, _ = full_run
    text = read(out_dir / "tables" / "11_run_validity.tex")
    assert "post\\_simtime\\_ll & 16 & 12 & caos=4" in text
    assert "post\\_simtime\\_br & 16 & 16 & --" in text


# --------------------------------------------------------------------------- #
# Hand-computed numeric checks
# --------------------------------------------------------------------------- #


def test_park_rate_density_sparse_k1():
    # post_simtime_ll, minimo (sparse): occ30 seed1/2 = 90/92, occ95 seed1/2 = 80/82
    values = [90, 92, 80, 82]
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    assert mean == 86.0
    assert round(sd, 4) == round(statistics.stdev([90, 92, 80, 82]), 4)

    df = pt.discover_series([METRICS_DIR], pt.Report())["post_simtime_ll"]
    got = df[(df["valid"]) & (df["density"] == "sparse")]["C1_park_rate"].tolist()
    assert sorted(got) == sorted(values)


def test_gap_to_broker_sparse(full_run):
    out_dir, _ = full_run
    # broker sparse mean = (97+99+87+89)/4 = 93.0 [occ30: 97,99; occ95: 87,89]
    broker_vals = [97, 99, 87, 89]
    k1_vals = [90, 92, 80, 82]
    k2_vals = [94, 96, 84, 86]
    broker_mean = statistics.mean(broker_vals)
    k1_mean = statistics.mean(k1_vals)
    k2_mean = statistics.mean(k2_vals)
    gap_k1 = broker_mean - k1_mean
    gap_k2 = broker_mean - k2_mean
    share_closed = (gap_k1 - gap_k2) / gap_k1 * 100

    text = read(out_dir / "tables" / "03_gap_to_broker.tex")
    assert f"sparse & {gap_k1:.1f} & {gap_k2:.1f} & {share_closed:.1f}" in text


def test_gap_to_broker_caos_k1_missing(full_run):
    # leaderless caos rows are all invalid -> k1 column must read "--"
    out_dir, _ = full_run
    text = read(out_dir / "tables" / "03_gap_to_broker.tex")
    assert "caos & -- &" in text


def test_occupancy_drop_broker_sparse(full_run):
    out_dir, _ = full_run
    # broker sparse: occ30 mean=(97+99)/2=98, occ95 mean=(87+89)/2=88, drop=10
    text = read(out_dir / "tables" / "04_occupancy_drop.tex")
    assert "Broker & sparse & 10.0" in text


def test_worst_cells_k1(full_run):
    out_dir, _ = full_run
    # k1 valid cells: sparse(86,76 avg? compute below), nominal, congested @ occ 30/95;
    # caos excluded (invalid). Minimum mean cell among those is congested@95:
    # occ95 seed1/2 = 65,67 -> mean 66
    text = read(out_dir / "tables" / "05_worst_cells.tex")
    assert "GeoGrid k=1 & congested & 95 & 66.0" in text


def test_worst_cells_k2_is_caos(full_run):
    out_dir, _ = full_run
    # k2 caos IS valid; base=70, occ95 -> occ_base=60, seed1/2=60/62, mean=61
    text = read(out_dir / "tables" / "05_worst_cells.tex")
    assert "GeoGrid k=2 & caos & 95 & 61.0" in text


def test_winners_parked_ratio(full_run):
    out_dir, _ = full_run
    # k1: winners=50, attempts=75 -> ratio 1.50 constant across all cells
    # k2: winners=55, attempts=66 -> ratio 1.20 constant across all cells
    text = read(out_dir / "tables" / "06_winners_parked.tex")
    assert "GeoGrid k=1 & sparse & 50.0 &" in text
    assert text.count("1.50") >= 3  # sparse, nominal, congested (caos invalid for k1)
    assert "GeoGrid k=2 & sparse & 55.0 &" in text
    assert "1.20" in text


def test_channel_table(full_run):
    out_dir, _ = full_run
    text = read(out_dir / "tables" / "07_channel.tex")
    assert "GeoGrid k=1 & sparse & 5.00 & 1000" in text
    assert "GeoGrid k=2 & sparse & 4.00 & 1200" in text
    assert "Broker & sparse & 3.00 & 800" in text


def test_timer_fix_robustness(full_run):
    out_dir, _ = full_run
    # broker sparse: pre-fix mean = (92+94+82+84)/4 = 88.0, sim-time mean = 93.0
    # -> diff = 5.0
    pre = [92, 94, 82, 84]
    post = [97, 99, 87, 89]
    diff = statistics.mean(post) - statistics.mean(pre)
    text = read(out_dir / "tables" / "10_timer_fix_robustness.tex")
    assert f"Broker & sparse & {statistics.mean(pre):.1f} & {statistics.mean(post):.1f} & {diff:.1f}" in text


# --------------------------------------------------------------------------- #
# Parameters table / config handling
# --------------------------------------------------------------------------- #


def test_parameters_table_values(full_run):
    out_dir, _ = full_run
    text = read(out_dir / "tables" / "01_parameters.tex")
    assert "GeoGrid k=1 & neighbor\\_k & 1" in text
    assert "GeoGrid k=2 & neighbor\\_k & 2" in text
    assert "Broker & backend & centralized" in text
    assert "Broker & claim\\_ttl\\_sim\\_s & 120.0" in text
    # non-broker arms must not carry broker-only keys
    assert "GeoGrid k=1 & claim\\_ttl\\_sim\\_s" not in text


def test_config_mismatch_raises(tmp_path):
    with pytest.raises(pt.ConfigMismatchError, match="gossip_interval_ms"):
        pt.load_configs(CONFIGS_MISMATCH_DIR, pt.Report())


# --------------------------------------------------------------------------- #
# Missing-series skip behaviour
# --------------------------------------------------------------------------- #


def test_missing_series_warns_and_skips(tmp_path):
    report = pt.Report()
    series = pt.discover_series([METRICS_MISSING_DIR], report)
    assert "post_simtime_ll_k2" not in series
    assert any("post_simtime_ll_k2" in w for w in report.warnings)


def test_missing_series_run_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    report = pt.run([METRICS_MISSING_DIR], CONFIGS_DIR, out_dir)
    # winners_parked needs GeoGrid arms; k2 missing but k1 present -> table still
    # built, with the k2 arm simply absent (no series to report "--" rows for).
    assert (out_dir / "tables" / "06_winners_parked.tex").exists()
    text = (out_dir / "tables" / "06_winners_parked.tex").read_text()
    assert "GeoGrid k=1 & sparse & 50.0" in text
    assert "GeoGrid k=2" not in text
    assert any("post_simtime_ll_k2" in w for w in report.warnings)


# --------------------------------------------------------------------------- #
# numbers.tex / macro hygiene
# --------------------------------------------------------------------------- #


def test_numbers_registry_rejects_non_alpha_macro():
    reg = pt.NumberRegistry()
    with pytest.raises(ValueError):
        reg.add("gap_sparse", "1.0")


def test_numbers_registry_rejects_conflicting_duplicate():
    reg = pt.NumberRegistry()
    reg.add("foo", "1.0")
    with pytest.raises(ValueError):
        reg.add("foo", "2.0")


def test_numbers_tex_written(full_run):
    out_dir, _ = full_run
    text = read(out_dir / "numbers.tex")
    assert "\\newcommand{\\" in text
    for line in text.strip().splitlines():
        assert line.startswith("\\newcommand{\\")


def test_table_column_limit_enforced():
    with pytest.raises(AssertionError):
        pt.render_latex(["a"] * 7, [], "caption", "tab:too_wide")


def test_check_txt_written(full_run):
    out_dir, _ = full_run
    text = read(out_dir / "check.txt")
    assert "Runs per series" in text
    assert "Config consistency" in text
    assert "Skipped tables" in text
