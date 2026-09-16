# PostProcessing-optimizer

## Regenerating paper tables

`paper_tables.py` is a standalone script that regenerates every LaTeX table
and prose-number macro used in the GeoGrid paper from post-processed
campaign metrics and run configs. It is independent of `post_process.py` /
`aggregate_seeds.py` and does not modify or import them — it only consumes
their output (`summary_all_experiments.csv`) and the run configs
(`config_used.toml`).

Requires Python 3.11+ and `pandas` (already in `requirements.txt`); no other
non-stdlib dependency (TOML parsing uses the stdlib `tomllib`).

### Usage

```bash
python3 paper_tables.py \
    --metrics /path/to/campaign_output_1 [/path/to/campaign_output_2 ...] \
    --configs /path/to/configs \
    --out out/
```

- `--metrics` — one or more directories, each expected to contain
  `<series>/summary_all_experiments.csv` for one or more of the known
  series (`post_simtime_ll`, `post_simtime_ll_k2`, `post_simtime_br`,
  `post_ll`, `post_br`, `rsu_simtime_{9,16,25}`,
  `mcs5_simtime_{br,ll_k2}`). Series not found in any `--metrics` dir have
  their tables skipped with a warning — this is expected for
  `rsu_simtime_{9,16,25}` and `mcs5_simtime_{br,ll_k2}` until those
  campaigns are run.
- `--configs` — a directory containing `configs/<tree>/<scenario>/<run>/
  config_used.toml` for each `campaign_simtime_*` tree (and later
  `rsu_simtime_*` / `mcs5_simtime_*`). Config trees may include runs still
  in progress; only runs present in the matching metrics CSV are counted.
- `--out` — output directory (gitignored via `/out/`). Written on every
  run:
  - `out/tables/<nn>_<name>.tex` — one booktabs table per question, each
    with a `\label{tab:<name>}` and a caption stating the per-cell `n` and
    which SD is reported (seed-only within a density/occupancy cell, or
    mixed occupancy+seed when aggregated across occupancy).
  - `out/numbers.tex` — `\newcommand` macros (letters-only names) for every
    number the paper prose quotes, so the text never hard-codes a figure.
  - `out/check.txt` — runs per series, valid runs (filter:
    `nr_sim_t_reached >= 179`), config consistency report, and which
    tables were skipped and why.

Real campaign data lives outside this repository; `fixtures/sample/` holds
a small representative sample with the real CSV/TOML headers, useful for a
smoke test:

```bash
python3 paper_tables.py --metrics fixtures/sample --configs fixtures/sample/configs --out /tmp/paper_out
```

### Tests

```bash
pip install -r tests/requirements-test.txt
python3 -m pytest tests/test_paper_tables.py -v
```

Synthetic fixtures under `tests/fixtures/` (regenerated via
`python3 tests/generate_fixtures.py` if the dataset needs to change) cover
all four densities, both occupancy levels used in the occupancy-drop table,
a config-mismatch case (to exercise the fail-loud parameter check), and a
metrics set missing one series (to exercise the skip-with-warning path).
