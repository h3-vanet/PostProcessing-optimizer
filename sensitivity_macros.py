#!/usr/bin/env python3
"""Generate LaTeX macros for the tuned-parameter sensitivity campaign.

Three datasets, each an ``assemble_dataset.py`` output directory:

* ``--phase-a``       nominal density, 10 LHS configurations (Phase A);
* ``--xd-sparse``     sparse density,  baseline + sens_c01 + sens_c03;
* ``--xd-congested``  congested density, same three configurations.

Every configuration is evaluated on the same 25 cells (5 occupancy x 5 seeds),
so all comparisons are *paired*: differences are taken cell by cell and a 95%
t-interval is computed over the 25 cell differences. A config that does not
carry exactly the 25 valid cells is an error, not a warning.

Macro families (LaTeX names cannot contain digits, hence CfgOne/CfgThree):

    sens<Resp>Base<Dens>             baseline mean
    sens<Resp><Cfg><Dens>            configuration mean
    sens<Resp>Delta<Cfg><Dens>[Lo|Hi]  paired diff vs baseline, 95% CI
    sens<Resp>Spread<Dens>[Lo|Hi]    paired diff CfgOne - CfgThree, 95% CI

    <Resp> in {Park, Bytes};  <Dens> in {Sparse, Nominal, Congested}

Phase-A surrogate outputs are prefixed ``sensPdp`` / ``sensRLoco`` so that a
number read off the surrogate can never be mistaken for one measured directly
on configurations (the two are different quantities).

Signed values carry an explicit sign: use them in math mode ($\\sensX$).

Usage:
    python3 sensitivity_macros.py --phase-a DIR --xd-sparse DIR \\
        --xd-congested DIR --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from paper_tables import NumberRegistry

CELLS = 25
DENSITIES = (("Sparse", "xd_sparse"), ("Nominal", "phase_a"),
             ("Congested", "xd_congested"))
CONFIGS = {"sens_c01": "CfgOne", "sens_c03": "CfgThree"}
RESPONSES = {"C1_park_rate": ("Park", 2),
             "D1_cp_tx_bytes_per_vehicle_mean": ("Bytes", 0),
             "A5_jain": ("Jain", 3),
             "A5b_overlap_rate_pct_d010": ("Overlap", 2),
             "A1_p50_ms": ("Timer", 0),
             "nr_plr_pct": ("Plr", 1)}
SHORT = {"C1_park_rate": "Park", "A5_jain": "Jain",
         "A5b_overlap_rate_pct_d010": "Overlap", "A1_p50_ms": "Timer",
         "D1_cp_tx_bytes_per_vehicle_mean": "Bytes", "nr_plr_pct": "Plr"}
T975_FALLBACK = {24: 2.0639}


def t_crit(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:
        return T975_FALLBACK[df]


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower() == "true"


def base_id(cid: str) -> str:
    """sens_c01_sparse -> sens_c01, so all densities share config names."""
    return "_".join(cid.split("_")[:2])


def load_runs(d: Path) -> tuple[pd.DataFrame, int]:
    raw = pd.read_csv(d / "dataset_per_run.csv")
    df = raw[_truthy(raw["valid"])].copy()
    df["base_id"] = df["config_id"].map(base_id)
    df["baseline"] = _truthy(df["is_baseline"])
    return df, len(raw)


def baseline_id(df: pd.DataFrame) -> str:
    ids = df.loc[df["baseline"], "base_id"].unique()
    if len(ids) != 1:
        raise ValueError(f"expected exactly one baseline, got {list(ids)}")
    return ids[0]


def cells(df: pd.DataFrame, bid: str, col: str) -> pd.Series:
    s = df[df["base_id"] == bid].set_index(["occupancy", "seed"])[col]
    s = pd.to_numeric(s, errors="coerce")
    if len(s) != CELLS or s.index.duplicated().any() or s.isna().any():
        raise ValueError(f"{bid}/{col}: need {CELLS} valid unique cells, "
                         f"got {len(s)}")
    return s


def paired(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    d = (a - b).dropna()
    if len(d) != CELLS:
        raise ValueError(f"pairing broken: {len(d)} common cells")
    m, sd = float(d.mean()), float(d.std(ddof=1))
    h = t_crit(len(d) - 1) * sd / math.sqrt(len(d))
    return m, m - h, m + h


def fmt(v: float, dec: int, signed: bool = False) -> str:
    return f"{v:+.{dec}f}" if signed else f"{v:.{dec}f}"


def add_ci(numbers, name, triple, dec):
    m, lo, hi = triple
    numbers.add(name, fmt(m, dec, True))
    numbers.add(name + "Lo", fmt(lo, dec, True))
    numbers.add(name + "Hi", fmt(hi, dec, True))


def build_density_macros(runs: dict, numbers: NumberRegistry) -> tuple[dict, dict]:
    report = {}
    means: dict = {}
    for dname, key in DENSITIES:
        df = runs[key]
        bid = baseline_id(df)
        means[dname] = {}
        for col, (rname, dec) in RESPONSES.items():
            base = cells(df, bid, col)
            numbers.add(f"sens{rname}Base{dname}", fmt(base.mean(), dec))
            means[dname][("base", col)] = float(base.mean())
            per_cfg = {}
            for cid, cname in CONFIGS.items():
                s = cells(df, cid, col)
                per_cfg[cid] = s
                numbers.add(f"sens{rname}{cname}{dname}", fmt(s.mean(), dec))
                means[dname][(cid, col)] = float(s.mean())
                add_ci(numbers, f"sens{rname}Delta{cname}{dname}",
                       paired(s, base), dec)
            spread = paired(per_cfg["sens_c01"], per_cfg["sens_c03"])
            add_ci(numbers, f"sens{rname}Spread{dname}", spread, dec)
            report[(rname, dname)] = spread
    return report, means


def p_bound(p: float) -> str:
    """Render a p-value as the tightest conventional bound it satisfies."""
    for bound in (0.001, 0.01, 0.05):
        if p < bound:
            return f"{bound:g}"
    return f"{p:.3f}"


def read_paper_macro(path: Path, name: str):
    """Read one ``\\newcommand{\\name}{value}`` from a paper macro file."""
    import re
    if not path.exists():
        return None
    m = re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % re.escape(name),
                  path.read_text())
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def build_paper_extras(runs: dict, means: dict, numbers: NumberRegistry,
                       paper_macros: Path) -> None:
    """Paper-only macros: Phase-A park-rate range, beta-Jain correlation, the
    gossip range actually covered, the largest cross-density park delta, the
    worst overlap, and the baseline-vs-main-campaign replication gaps."""
    df = runs["phase_a"]
    g = df.groupby("base_id").agg(
        jain=("A5_jain", "mean"),
        park=("C1_park_rate", "mean"),
        bytes=("D1_cp_tx_bytes_per_vehicle_mean", "mean"),
        alfa=("alfa", "mean"),
        beta=("beta", "mean"),
        gossip=("gossip_interval_ms", "mean"),
    )
    if len(g) != int("10"):
        raise ValueError(f"expected 10 Phase-A configurations, got {len(g)}")

    try:
        from scipy.stats import spearmanr
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("scipy is required for the beta-Jain correlation") from exc

    rho, pval = spearmanr(g["beta"], g["jain"])
    rho_no1, pval_no1 = spearmanr(g.drop(index="sens_c01")["beta"],
                                  g.drop(index="sens_c01")["jain"])
    numbers.add("sensJainBetaRho", f"{float(rho):.2f}")
    numbers.add("sensJainBetaP", p_bound(float(pval)))
    numbers.add("sensJainBetaRhoNoCfgOne", f"{float(rho_no1):.2f}")
    numbers.add("sensJainBetaPNoCfgOne", p_bound(float(pval_no1)))

    numbers.add("sensParkPhaseALo", f"{g['park'].min():.2f}")
    numbers.add("sensParkPhaseAHi", f"{g['park'].max():.2f}")
    numbers.add("sensTGossipCoveredLo", f"{g['gossip'].min():g}")
    numbers.add("sensTGossipCoveredHi", f"{g['gossip'].max():g}")
    numbers.add("sensTGossipCfgOne", f"{g.loc['sens_c01', 'gossip']:g}")
    numbers.add("sensAlfaBaseline", f"{g.loc['sens_c00', 'alfa']:.2f}")
    numbers.add("sensBetaBaseline", f"{g.loc['sens_c00', 'beta']:.2f}")

    max_delta = 0.0
    for dname, _ in DENSITIES:
        for cid in CONFIGS:
            max_delta = max(max_delta, abs(
                means[dname][(cid, "C1_park_rate")]
                - means[dname][("base", "C1_park_rate")]))
    numbers.add("sensParkMaxAbsDelta", f"{max_delta:.2f}")

    overlap_max = max(
        means[dname][(k, "A5b_overlap_rate_pct_d010")]
        for dname, _ in DENSITIES for k in ("base", "sens_c01", "sens_c03"))
    numbers.add("sensOverlapMax", f"{overlap_max:.2f}")

    main_park = read_paper_macro(paper_macros, "parkRateNominalGeoGridKTwo")
    main_bytes = read_paper_macro(paper_macros, "txBytesNominalGeoGridKTwo")
    base_park = float(g.loc["sens_c00", "park"])
    base_bytes = float(g.loc["sens_c00", "bytes"])
    if main_park:
        numbers.add("sensReproParkGapNominal", f"{abs(base_park - main_park):.2f}")
    if main_bytes:
        numbers.add("sensReproBytesGapPctNominal",
                    f"{abs(base_bytes - main_bytes) / main_bytes * 100:.1f}")


def write_paper_table(means: dict, path: Path) -> None:
    """Compact paper table: four responses x (Base/C01/C03) at three densities."""
    rows = [
        ("Park rate (\\%)", "C1_park_rate", 2),
        ("Jain index", "A5_jain", 3),
        ("Overlap (\\%)", "A5b_overlap_rate_pct_d010", 2),
        ("CP tx bytes/vehicle", "D1_cp_tx_bytes_per_vehicle_mean", 0),
    ]
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Sensitivity to the operating point: tuned baseline and",
        "configurations C01 and C03 at the three evaluable densities (mean over",
        "\\sensCells{} cells).}",
        "\\label{tab:sens-paper}", "\\footnotesize",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}lccccccccc@{}}", "\\toprule",
        " & \\multicolumn{3}{c}{\\textbf{Sparse}} & "
        "\\multicolumn{3}{c}{\\textbf{Nominal}} & "
        "\\multicolumn{3}{c}{\\textbf{Congested}} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "\\textbf{Response} & Base & C01 & C03 & Base & C01 & C03 & Base & C01 & C03 \\\\",
        "\\midrule",
    ]
    for label, col, dec in rows:
        vals = [f"{means[dname][(k, col)]:.{dec}f}"
                for dname, _ in DENSITIES
                for k in ("base", "sens_c01", "sens_c03")]
        lines.append(f"{label} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_phase_a_macros(pa: Path, numbers: NumberRegistry) -> None:
    pdp = pd.read_csv(pa / "analysis" / "sensitivity_pdp.csv")

    def pdp_range(param: str, resp: str) -> float:
        s = pdp[(pdp["parameter"] == param) & (pdp["response"] == resp)]["pdp"]
        if s.empty:
            raise ValueError(f"no PDP for {param} on {resp}")
        return float(s.max() - s.min())

    numbers.add("sensPdpAlfaParkRange",
                fmt(pdp_range("alfa", "C1_park_rate"), 2))
    numbers.add("sensPdpGossipParkRange",
                fmt(pdp_range("gossip_interval_ms", "C1_park_rate"), 2))
    numbers.add("sensPdpGossipBytesRange",
                fmt(pdp_range("gossip_interval_ms",
                              "D1_cp_tx_bytes_per_vehicle_mean"), 0))

    summary = json.loads((pa / "analysis" / "sensitivity_summary.json").read_text())
    for resp, short in SHORT.items():
        r2 = summary["responses"][resp]["r2_loco"]
        numbers.add(f"sensRLoco{short}", fmt(r2, 2))
    dec = summary["decision"]
    numbers.add("sensResponsesWithSignal", str(dec["n_responses_with_signal"]))
    numbers.add("sensResponsesTotal", str(dec["n_responses"]))
    numbers.add("sensNullPerm", str(dec["null_perm"]))

    inter = pd.read_csv(pa / "analysis" / "sensitivity_interactions.csv")
    est = inter[_truthy(inter["estimable"])]
    numbers.add("sensMaxHTwo", fmt(float(est["H2"].max()), 3))

    for key, name in (("python", "Python"), ("numpy", "Numpy"),
                      ("pandas", "Pandas"), ("scikit-learn", "Sklearn"),
                      ("scipy", "Scipy")):
        if key in summary.get("versions", {}):
            numbers.add(f"sensVer{name}", summary["versions"][key])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-a", required=True, type=Path)
    parser.add_argument("--xd-sparse", required=True, type=Path)
    parser.add_argument("--xd-congested", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--paper-out", type=Path, default=None,
        help="paper generated/ dir: also writes sensitivity_numbers.tex and "
        "tables/sens_paper.tex (paper macros may reuse the thesis names)",
    )
    args = parser.parse_args(argv)

    runs, totals = {}, {}
    for key, d in (("phase_a", args.phase_a), ("xd_sparse", args.xd_sparse),
                   ("xd_congested", args.xd_congested)):
        runs[key], totals[key] = load_runs(d)

    numbers = NumberRegistry()
    numbers.add("sensCells", str(CELLS))
    numbers.add("sensConfigsPhaseA", str(runs["phase_a"]["config_id"].nunique()))
    numbers.add("sensRunsPhaseA", str(totals["phase_a"]))
    numbers.add("sensRunsCrossDensity",
                str(totals["xd_sparse"] + totals["xd_congested"]))

    report, means = build_density_macros(runs, numbers)
    build_phase_a_macros(args.phase_a, numbers)

    args.out.mkdir(parents=True, exist_ok=True)
    numbers.write(args.out / "sensitivity_numbers.tex")

    if args.paper_out is not None:
        paper_numbers = NumberRegistry()
        for k, v in numbers.as_dict().items():
            paper_numbers.add(k, v)
        build_paper_extras(runs, means, paper_numbers,
                           args.paper_out / "numbers.tex")
        args.paper_out.mkdir(parents=True, exist_ok=True)
        paper_numbers.write(args.paper_out / "sensitivity_numbers.tex")
        write_paper_table(means, args.paper_out / "tables" / "sens_paper.tex")
        print(f"paper: {args.paper_out / 'sensitivity_numbers.tex'} "
              f"({len(paper_numbers.as_dict())} macros) + tables/sens_paper.tex")

    for (rname, dname), (m, lo, hi) in report.items():
        print(f"{rname:8s} spread CfgOne-CfgThree {dname:9s}: "
              f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
