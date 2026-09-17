import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import caos_crash_macros as cc
import gossip_probe_macros as gp
from paper_tables import NumberRegistry

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_probe_exact_five_share_below_one():
    stats = gp.parse_probe(FIXTURES / "probe" / "probe_gossip_rounds.jsonl")
    assert stats["vehicles"] == 2
    assert stats["rounds"] == 6
    assert stats["intervals"] == 4
    # intervals are [5, 5, 10, 15]
    assert stats["exact_five"] == 2
    assert stats["exact_five_share"] == 50.0
    assert stats["exact_five_share"] < 100.0  # some intervals are not exactly 5
    assert stats["multiple_five"] == 4
    assert stats["multiple_five_share"] == 100.0


def test_gossip_macros_written(tmp_path):
    stats = gp.parse_probe(FIXTURES / "probe" / "probe_gossip_rounds.jsonl")
    numbers = NumberRegistry()
    gp.build_macros(stats, numbers)
    numbers.write(tmp_path / "gossip_numbers.tex")
    text = (tmp_path / "gossip_numbers.tex").read_text()
    assert "\\newcommand{\\gossipProbeIntervals}{4}" in text
    assert "\\newcommand{\\gossipProbeExactFiveCount}{2}" in text
    assert "\\newcommand{\\gossipProbeExactFiveShare}{50.00}" in text
    assert "\\newcommand{\\gossipProbeMultipleFiveShare}{100.00}" in text


def test_parse_crashes_per_campaign():
    per = cc.parse_crashes(FIXTURES / "caos" / "caos_crashes.txt")
    assert per["campaign_ll"] == {"total": 3, "segv": 1, "abort": 1, "neither": 1}
    assert per["campaign_simtime_ll"] == {"total": 1, "segv": 1, "abort": 0, "neither": 0}


def test_caos_macros_written(tmp_path):
    per = cc.parse_crashes(FIXTURES / "caos" / "caos_crashes.txt")
    numbers = NumberRegistry()
    cc.build_macros(per, numbers)
    numbers.write(tmp_path / "caos_numbers.tex")
    text = (tmp_path / "caos_numbers.tex").read_text()
    assert "\\newcommand{\\caosCampaignLlTotal}{3}" in text
    assert "\\newcommand{\\caosCampaignLlSegv}{1}" in text
    assert "\\newcommand{\\caosCampaignLlAbort}{1}" in text
    assert "\\newcommand{\\caosCampaignLlNeither}{1}" in text
    assert "\\newcommand{\\caosCampaignSimtimeLlTotal}{1}" in text
    assert "\\newcommand{\\caosAllTotal}{4}" in text
    assert "\\newcommand{\\caosAllSegv}{2}" in text
    assert "\\newcommand{\\caosAllAbort}{1}" in text
    assert "\\newcommand{\\caosAllNeither}{1}" in text
