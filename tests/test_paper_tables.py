import json
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
CONFIGS_EFFECTIVE_MISMATCH_DIR = FIXTURES / "configs_effective_mismatch"
METRICS_BROKER_SUFFIX_DIR = FIXTURES / "metrics_broker_suffix"
METRICS_BROKER_SUFFIX_FULL_DIR = FIXTURES / "metrics_broker_suffix_full"
CONFIGS_BROKER_SUFFIX_DIR = FIXTURES / "configs_broker_suffix"
METRICS_CAOS_INVALID_DIR = FIXTURES / "metrics_caos_invalid"


def read(path: Path) -> str:
    return path.read_text()


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("out")
    report, numbers = pt.run([METRICS_DIR], CONFIGS_DIR, out_dir)
    return out_dir, report, numbers


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
    out_dir, _, _ = full_run
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
    out_dir, _, _ = full_run
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
    out_dir, _, _ = full_run
    text = read(out_dir / "tables" / "03_gap_to_broker.tex")
    assert "caos & -- &" in text


def test_occupancy_drop_broker_sparse(full_run):
    out_dir, _, _ = full_run
    # broker sparse: occ30 mean=(97+99)/2=98, occ95 mean=(87+89)/2=88, drop=10
    text = read(out_dir / "tables" / "04_occupancy_drop.tex")
    assert "Broker & sparse & 10.0" in text


def test_worst_cells_k1(full_run):
    out_dir, _, _ = full_run
    # k1 valid cells: sparse(86,76 avg? compute below), nominal, congested @ occ 30/95;
    # caos excluded (invalid). Minimum mean cell among those is congested@95:
    # occ95 seed1/2 = 65,67 -> mean 66
    text = read(out_dir / "tables" / "05_worst_cells.tex")
    assert "GeoGrid k=1 & congested & 95 & 66.0" in text


def test_worst_cells_k2_is_caos(full_run):
    out_dir, _, _ = full_run
    # k2 caos IS valid; base=70, occ95 -> occ_base=60, seed1/2=60/62, mean=61
    text = read(out_dir / "tables" / "05_worst_cells.tex")
    assert "GeoGrid k=2 & caos & 95 & 61.0" in text


def test_winners_parked_ratio(full_run):
    out_dir, _, _ = full_run
    # k1: winners(A2_vehicles_with_slot)=40, claims(A2_assigned_uniq)=50
    #     -> ratio of means = 50/40 = 1.25, constant across all cells
    # k2: winners=50, claims=55 -> ratio = 55/50 = 1.10
    text = read(out_dir / "tables" / "06_winners_parked.tex")
    assert "GeoGrid k=1 & sparse & 40.0 & 50.0 &" in text
    assert text.count("1.25") >= 3  # sparse, nominal, congested (caos invalid for k1)
    assert "GeoGrid k=2 & sparse & 50.0 & 55.0 &" in text
    assert "1.10" in text


def test_channel_table(full_run):
    out_dir, _, _ = full_run
    text = read(out_dir / "tables" / "07_channel.tex")
    assert "GeoGrid k=1 & sparse & 5.0 & 1000" in text
    assert "GeoGrid k=2 & sparse & 4.0 & 1200" in text
    assert "Broker & sparse & 3.0 & 800" in text


def test_timer_fix_robustness(full_run):
    out_dir, _, _ = full_run
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
    out_dir, _, _ = full_run
    text = read(out_dir / "tables" / "01_parameters.tex")
    # main GeoGrid tree is k=2; values read from config_used.toml
    assert "$k$ & 2 & gossip k-ring & yes & 1--2" in text
    assert "$r_c$ & 10 & H3 cluster resolution & no & --" in text
    assert "$r_s$ & 14 & H3 spot resolution & no & --" in text
    assert "$\\ell_c$ & 75.9 & average cluster-cell edge length (m) & no & --" in text
    assert "$\\ell_s$ & 1.55 & average spot-cell edge length (m) & no & --" in text
    assert "$T_{\\text{TTL}}$ & 3600 & CRDT slot TTL (s) & no & --" in text
    assert "$D_{\\max}$ & 500.0 &" in text
    # tuned keys are absent from the fixture TOML -> effective defaults, marked,
    # and their rationale is only "tuned"
    assert "$\\alpha$ & 0.7 (default) & tuned & no & --" in text
    assert "$\\beta$ & 0.3 (default) & tuned & no & --" in text
    assert "$T_{\\text{base}}$ & 200.0 (default) & tuned & no & --" in text
    assert "$n_{\\text{ae}}$ & 10 (default) & tuned & no & --" in text
    # broker RTT effective default (fixture broker TOML has no rtt_ms)
    assert "$\\text{RTT}$ & 0 (default) & broker uplink round-trip (ms) & yes & 0, 50" in text
    # env / scenario-level design rows
    assert "MCS & 14 &" in text
    assert "$N_{\\text{RSU}}$ & 4 &" in text
    # keys the core ignores must never appear in the .tex
    assert "claim\\_ttl\\_sim\\_s" not in text
    assert "lease\\_sim\\_s" not in text
    assert "uplink\\_timeout\\_sim\\_s" not in text


def test_build_parameters_reads_present_tuned_values():
    # when a tuned key IS present in config_used.toml it must be reported as
    # the effective value (no "(default)" marker)
    configs = {
        "campaign_simtime_ll_k2": {
            "assignment.backoff.alfa": 0.3098,
            "assignment.backoff.beta": 0.6902,
            "assignment.backoff.t_base_ms": 307.7,
            "gossip.anti_entropy_every_n_rounds": 50,
        }
    }
    result = pt.build_parameters(configs, pt.NumberRegistry(), pt.Report())
    assert "$\\alpha$ & 0.3098 & tuned & no & --" in result.latex
    assert "$\\beta$ & 0.6902 & tuned & no & --" in result.latex
    assert "$T_{\\text{base}}$ & 307.7 & tuned & no & --" in result.latex
    assert "$n_{\\text{ae}}$ & 50 & tuned & no & --" in result.latex


def test_ignored_by_core_keys_in_check_txt(full_run):
    out_dir, _, _ = full_run
    text = read(out_dir / "check.txt")
    assert "ignored by core" in text
    assert "claim_ttl_sim_s" in text
    assert "lease_sim_s" in text
    assert "uplink_timeout_sim_s" in text


def test_config_mismatch_raises(tmp_path):
    with pytest.raises(pt.ConfigMismatchError, match="gossip_interval_ms"):
        pt.load_configs(CONFIGS_MISMATCH_DIR, pt.Report())


def test_effective_value_mismatch_raises(tmp_path):
    # one run leaves broker.rtt_ms absent (effective default 0), the other
    # sets it explicitly to 10 -> effective values differ -> must raise even
    # though neither run's raw TOML keys directly collide.
    with pytest.raises(pt.ConfigMismatchError, match="rtt_ms"):
        pt.load_configs(CONFIGS_EFFECTIVE_MISMATCH_DIR, pt.Report())


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
    report, _numbers = pt.run([METRICS_MISSING_DIR], CONFIGS_DIR, out_dir)
    # winners_parked needs GeoGrid arms; k2 missing but k1 present -> table still
    # built, with the k2 arm simply absent (no series to report "--" rows for).
    assert (out_dir / "tables" / "06_winners_parked.tex").exists()
    text = (out_dir / "tables" / "06_winners_parked.tex").read_text()
    assert "GeoGrid k=1 & sparse & 40.0" in text
    assert "GeoGrid k=2" not in text
    assert any("post_simtime_ll_k2" in w for w in report.warnings)


# --------------------------------------------------------------------------- #
# --broker-suffix
# --------------------------------------------------------------------------- #


def test_broker_suffix_discovery():
    report = pt.Report()
    series = pt.discover_series([METRICS_BROKER_SUFFIX_DIR], report, broker_suffix="_v2")
    assert "post_simtime_br" in series
    # without the suffix, the same directory must NOT be found
    report2 = pt.Report()
    series2 = pt.discover_series([METRICS_BROKER_SUFFIX_DIR], report2, broker_suffix="")
    assert "post_simtime_br" not in series2


# --- (1) suffix applies to broker CONFIG trees too, and captions/parameters
# read the effective value from the SUFFIXED tree, never the unsuffixed one ---


def test_load_configs_applies_broker_suffix():
    report = pt.Report()
    configs = pt.load_configs(CONFIGS_BROKER_SUFFIX_DIR, report, broker_suffix="_v2")
    assert "campaign_simtime_br" in configs
    # the suffixed tree (rtt_ms=99) must be the one loaded under the
    # unsuffixed key, NOT the unsuffixed tree on disk (rtt_ms=5)
    assert configs["campaign_simtime_br"]["broker.rtt_ms"] == 99


def test_broker_rtt_note_reads_suffixed_tree():
    report = pt.Report()
    configs = pt.load_configs(CONFIGS_BROKER_SUFFIX_DIR, report, broker_suffix="_v2")
    note = pt.broker_rtt_note(configs)
    assert "99" in note
    assert "5" not in note.split("=")[-1]  # not the unsuffixed tree's value


def test_parameters_table_reads_suffixed_broker_tree(tmp_path):
    out_dir = tmp_path / "out"
    report, _numbers = pt.run(
        [METRICS_BROKER_SUFFIX_FULL_DIR],
        CONFIGS_BROKER_SUFFIX_DIR,
        out_dir,
        broker_suffix="_v2",
    )
    text = (out_dir / "tables" / "01_parameters.tex").read_text()
    # RTT row must read the SUFFIXED broker tree (rtt_ms=99), not the
    # unsuffixed tree on disk (rtt_ms=5)
    assert "$\\text{RTT}$ & 99 & broker uplink round-trip (ms) & yes & 0, 50" in text
    assert "$\\text{RTT}$ & 5 " not in text


def test_check_txt_reports_suffixed_config_tree_source(tmp_path):
    out_dir = tmp_path / "out"
    report, _numbers = pt.run(
        [METRICS_BROKER_SUFFIX_FULL_DIR],
        CONFIGS_BROKER_SUFFIX_DIR,
        out_dir,
        broker_suffix="_v2",
    )
    text = (out_dir / "check.txt").read_text()
    assert "campaign_simtime_br_v2" in text


# --- (2) suffix never applied to pre-fix (post_br) or leaderless series/trees ---


def test_broker_suffix_never_applied_to_prefix_series():
    report = pt.Report()
    series = pt.discover_series([METRICS_BROKER_SUFFIX_FULL_DIR], report, broker_suffix="_v2")
    # post_br is unsuffixed on disk; it must be found even though a broker
    # suffix is set, because pre-fix series are excluded from BROKER_SERIES
    assert "post_br" in series
    assert not any("post_br" in w and "not found" in w for w in report.warnings)


def test_broker_suffix_never_applied_to_leaderless_series():
    report = pt.Report()
    series = pt.discover_series([METRICS_BROKER_SUFFIX_FULL_DIR], report, broker_suffix="_v2")
    assert "post_simtime_ll" in series


def test_broker_suffix_never_applied_to_leaderless_config_tree():
    # campaign_simtime_ll has no suffixed "_v2" counterpart on disk at all;
    # it must still be found because leaderless trees are never suffixed.
    report = pt.Report()
    configs = pt.load_configs(CONFIGS_BROKER_SUFFIX_DIR, report, broker_suffix="_v2")
    assert "campaign_simtime_ll" in configs


def test_broker_series_and_config_trees_exclude_prefix_and_leaderless():
    assert "post_br" not in pt.BROKER_SERIES
    assert "post_simtime_ll" not in pt.BROKER_SERIES
    assert "post_simtime_ll_k2" not in pt.BROKER_SERIES
    assert "mcs5_simtime_ll_k2" not in pt.BROKER_SERIES
    assert "campaign_simtime_ll" not in pt.BROKER_CONFIG_TREES
    assert "campaign_simtime_ll_k2" not in pt.BROKER_CONFIG_TREES
    assert "mcs5_simtime_ll_k2" not in pt.BROKER_CONFIG_TREES


# --- (3) fail loudly on a suffixed broker series with no matching suffixed
# config tree, instead of silently falling back to the unsuffixed tree ---


def test_missing_suffixed_config_tree_raises():
    report = pt.Report()
    series = pt.discover_series([METRICS_BROKER_SUFFIX_FULL_DIR], report, broker_suffix="_v2")
    # CONFIGS_DIR only has the unsuffixed campaign_simtime_br, never
    # campaign_simtime_br_v2 -> must raise, not fall back to it.
    configs = pt.load_configs(CONFIGS_DIR, pt.Report(), broker_suffix="_v2")
    assert "campaign_simtime_br" not in configs  # confirms no fallback happened
    with pytest.raises(pt.MissingConfigTreeError, match="campaign_simtime_br_v2"):
        pt.verify_broker_config_pairing(series, configs, "_v2")


def test_missing_suffixed_config_tree_raises_via_run(tmp_path):
    out_dir = tmp_path / "out"
    with pytest.raises(pt.MissingConfigTreeError):
        pt.run([METRICS_BROKER_SUFFIX_FULL_DIR], CONFIGS_DIR, out_dir, broker_suffix="_v2")


def test_verify_broker_config_pairing_noop_without_suffix():
    # with no --broker-suffix, missing config trees are just a warning
    # (existing behaviour), never a hard failure.
    report = pt.Report()
    series = pt.discover_series([METRICS_BROKER_SUFFIX_FULL_DIR], report, broker_suffix="")
    configs = pt.load_configs(CONFIGS_DIR, pt.Report(), broker_suffix="")
    pt.verify_broker_config_pairing(series, configs, "")  # must not raise


# --------------------------------------------------------------------------- #
# --expect
# --------------------------------------------------------------------------- #


def test_check_expectations_pass(full_run):
    _out_dir, _report, numbers = full_run
    macro = "parkRateSparseGeoGridKOne"
    actual = float(numbers.as_dict()[macro])
    mismatches = pt.check_expectations(numbers, {macro: actual})
    assert mismatches == []


def test_check_expectations_within_tolerance(full_run):
    _out_dir, _report, numbers = full_run
    macro = "parkRateSparseGeoGridKOne"
    actual = float(numbers.as_dict()[macro])
    mismatches = pt.check_expectations(numbers, {macro: actual + 0.04})
    assert mismatches == []


def test_check_expectations_fail(full_run):
    _out_dir, _report, numbers = full_run
    macro = "parkRateSparseGeoGridKOne"
    actual = float(numbers.as_dict()[macro])
    mismatches = pt.check_expectations(numbers, {macro: actual + 5.0})
    assert len(mismatches) == 1
    assert macro in mismatches[0]


def test_check_expectations_missing_macro(full_run):
    _out_dir, _report, numbers = full_run
    mismatches = pt.check_expectations(numbers, {"noSuchMacro": 1.0})
    assert len(mismatches) == 1
    assert "not produced" in mismatches[0]


def test_main_expect_exits_nonzero(tmp_path):
    out_dir = tmp_path / "out"
    expect_file = tmp_path / "expect.json"
    expect_file.write_text(json.dumps({"noSuchMacro": 1.0}))
    rc = pt.main(
        [
            "--metrics", str(METRICS_DIR),
            "--configs", str(CONFIGS_DIR),
            "--out", str(out_dir),
            "--expect", str(expect_file),
        ]
    )
    assert rc == 1


def test_main_expect_exits_zero_on_match(tmp_path):
    out_dir = tmp_path / "out"
    report, numbers = pt.run([METRICS_DIR], CONFIGS_DIR, out_dir)
    macro = "parkRateSparseGeoGridKOne"
    actual = float(numbers.as_dict()[macro])
    expect_file = tmp_path / "expect.json"
    expect_file.write_text(json.dumps({macro: actual}))
    rc = pt.main(
        [
            "--metrics", str(METRICS_DIR),
            "--configs", str(CONFIGS_DIR),
            "--out", str(out_dir),
            "--expect", str(expect_file),
        ]
    )
    assert rc == 0


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
    out_dir, _, _ = full_run
    text = read(out_dir / "numbers.tex")
    assert "\\newcommand{\\" in text
    for line in text.strip().splitlines():
        assert line.startswith("\\newcommand{\\")


def test_density_capitalized_in_macro_names(full_run):
    # Macro names capitalize density (parkRateSparse...) while table text
    # stays lowercase (density column reads "sparse").
    _out_dir, _report, numbers = full_run
    macros = numbers.as_dict()
    assert "parkRateSparseGeoGridKTwo" in macros
    assert "robustSparseBroker" in macros
    assert not any(m.startswith("parkRatesparse") for m in macros)
    assert not any(m.startswith("robustsparse") for m in macros)


def test_density_macro_capitalizes():
    assert pt.density_macro("sparse") == "Sparse"
    assert pt.density_macro("nominal") == "Nominal"
    assert pt.density_macro("congested") == "Congested"
    assert pt.density_macro("caos") == "Caos"


def test_table_column_limit_enforced():
    with pytest.raises(AssertionError):
        pt.render_latex(["a"] * 7, [], "caption", "tab:too_wide")


def test_check_txt_written(full_run):
    out_dir, _, _ = full_run
    text = read(out_dir / "check.txt")
    assert "Runs per series" in text
    assert "Config consistency" in text
    assert "Skipped tables" in text


# --------------------------------------------------------------------------- #
# (1) drop all-dash density rows, with a caption note
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def caos_invalid_run(tmp_path_factory):
    # both GeoGrid arms leaderless -> caos invalid for both -> the caos row
    # in gap_to_broker and winners_parked has nothing but dashes.
    out_dir = tmp_path_factory.mktemp("caos_out")
    report, numbers = pt.run([METRICS_CAOS_INVALID_DIR], CONFIGS_DIR, out_dir)
    return out_dir, report, numbers


def test_gap_to_broker_drops_all_dash_caos_row(caos_invalid_run):
    out_dir, _, _ = caos_invalid_run
    text = read(out_dir / "tables" / "03_gap_to_broker.tex")
    assert "caos &" not in text
    assert "sparse &" in text  # other densities remain
    assert "caos omitted: leaderless runs invalid." in text


def test_winners_parked_drops_all_dash_caos_rows(caos_invalid_run):
    out_dir, _, _ = caos_invalid_run
    text = read(out_dir / "tables" / "06_winners_parked.tex")
    assert "caos &" not in text
    assert "GeoGrid k=1 & sparse &" in text
    assert "caos omitted: leaderless runs invalid." in text


def test_filter_all_dash_rows_drops_only_fully_dashed():
    rows = [
        ["sparse", "1.0", "2.0"],
        ["caos", "--", "--"],
        ["nominal", "3.0", "--"],  # partially dashed -> kept
    ]
    kept = pt.filter_all_dash_rows(rows, value_start_idx=1)
    assert kept == [["sparse", "1.0", "2.0"], ["nominal", "3.0", "--"]]


def test_omitted_density_note_names_missing_density():
    rows = [["sparse", "1.0"], ["nominal", "2.0"], ["congested", "3.0"]]
    note = pt.omitted_density_note(rows, density_col_idx=0)
    assert "caos omitted: leaderless runs invalid." in note


def test_omitted_density_note_empty_when_all_present():
    rows = [["sparse", "1.0"], ["nominal", "2.0"], ["congested", "3.0"], ["caos", "4.0"]]
    assert pt.omitted_density_note(rows, density_col_idx=0) == ""


def test_occupancy_drop_and_channel_and_timer_fix_still_have_all_densities(full_run):
    # sanity check the filter doesn't over-trigger on the main fixture, where
    # only k1 (not k2) is invalid at caos, so caos survives via k2/broker.
    out_dir, _, _ = full_run
    for name in ("04_occupancy_drop", "07_channel", "10_timer_fix_robustness"):
        text = read(out_dir / "tables" / f"{name}.tex")
        assert "caos" in text


# --------------------------------------------------------------------------- #
# (2) PLR formatted to 1 decimal everywhere
# --------------------------------------------------------------------------- #


def test_plr_one_decimal_in_channel_table(full_run):
    out_dir, _, _ = full_run
    text = read(out_dir / "tables" / "07_channel.tex")
    assert "5.00" not in text
    assert "4.00" not in text
    assert "3.00" not in text
    assert "5.0 &" in text


# --------------------------------------------------------------------------- #
# (3) "omitted when not yet available" sentence only when something IS omitted
# --------------------------------------------------------------------------- #


def test_rsu_sensitivity_caption_mentions_omission_when_series_missing(full_run):
    out_dir, _, _ = full_run
    report_text = read(out_dir / "check.txt")
    assert "rsu_sensitivity" in report_text  # skipped in this fixture (no RSU series)


def test_mcs_caption_omits_sentence_when_all_rows_present(tmp_path):
    # build a fixture where every MCS combo's series is present, so the
    # "omitted when not yet available" sentence should NOT appear.
    import shutil

    metrics_dir = tmp_path / "metrics"
    for series in ("post_simtime_br", "post_simtime_ll_k2"):
        shutil.copytree(METRICS_DIR / series, metrics_dir / series)
    # duplicate the same data under the MCS5 series names so all 4 combos
    # in the mcs table have data (content doesn't matter for this check).
    shutil.copytree(METRICS_DIR / "post_simtime_br", metrics_dir / "mcs5_simtime_br")
    shutil.copytree(METRICS_DIR / "post_simtime_ll_k2", metrics_dir / "mcs5_simtime_ll_k2")

    out_dir = tmp_path / "out"
    report, _numbers = pt.run([metrics_dir], CONFIGS_DIR, out_dir)
    text = read(out_dir / "tables" / "09_mcs.tex")
    assert "omitted when that campaign is not yet available" not in text


def test_mcs_caption_keeps_sentence_when_a_row_is_missing(full_run):
    out_dir, _, _ = full_run
    text = read(out_dir / "tables" / "09_mcs.tex")
    assert "omitted when that campaign is not yet available" in text


def test_rsu_sensitivity_caption_keeps_omission_sentence(full_run):
    out_dir, _, _ = full_run
    # count=4 (post_simtime_br) is present, but 9/16/25 are not -> the table
    # IS built (row 4 has data) and must still mention the omission.
    text = read(out_dir / "tables" / "08_rsu_sensitivity.tex")
    assert "RSU count 9/16/25 rows are omitted when their campaign is not yet available." in text


# --------------------------------------------------------------------------- #
# (4) claims-per-winner macros
# --------------------------------------------------------------------------- #


def test_claims_per_winner_macros(full_run):
    _out_dir, _report, numbers = full_run
    macros = numbers.as_dict()
    # k1: claims=50, winners=40 -> ratio 1.25; k2: claims=55, winners=50 -> 1.10
    assert macros["claimsPerWinnerSparseGeoGridKOne"] == "1.25"
    assert macros["claimsPerWinnerNominalGeoGridKOne"] == "1.25"
    assert macros["claimsPerWinnerCongestedGeoGridKOne"] == "1.25"
    assert macros["claimsPerWinnerSparseGeoGridKTwo"] == "1.10"
    # caos invalid for k1 (main fixture) -> no macro emitted for it
    assert "claimsPerWinnerCaosGeoGridKOne" not in macros
    assert "claimsPerWinnerCaosGeoGridKTwo" in macros


def test_claims_per_winner_macro_matches_table_value(full_run):
    out_dir, _report, numbers = full_run
    text = read(out_dir / "tables" / "06_winners_parked.tex")
    macros = numbers.as_dict()
    assert "GeoGrid k=2 & sparse & 50.0 & 55.0 & 90.0 & 1.10" in text
    assert macros["claimsPerWinnerSparseGeoGridKTwo"] == "1.10"
