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
- `--broker-suffix` (default `""`) — appended to the directory name of every
  broker series and its matching config tree (`post_simtime_br` /
  `campaign_simtime_br`, `rsu_simtime_{9,16,25}`, `mcs5_simtime_br`) when
  looking them up under `--metrics` / `--configs`, e.g. `--broker-suffix _v2`
  looks for `post_simtime_br_v2/summary_all_experiments.csv` and
  `campaign_simtime_br_v2/.../config_used.toml`. Captions and the parameters
  table read the effective values (e.g. `broker.rtt_ms`) from the suffixed
  tree. Never applied to the pre-fix series (`post_br`) or to any leaderless
  series/tree (`post_simtime_ll*`, `mcs5_simtime_ll_k2`), even though
  `post_br` shares the Broker arm label. If a suffixed broker metrics series
  is found but its matching suffixed config tree is not, the script fails
  loudly (`MissingConfigTreeError`) instead of silently falling back to the
  unsuffixed tree.
- `--expect <file.json>` — optional JSON file of `{"macroName": expectedValue,
  ...}`. After generating the tables, compares each named macro from
  `numbers.tex` against its expected value (tolerance 0.05) and exits
  non-zero (printing every mismatch to stderr) if any differ — useful as a
  CI regression guard on the paper's quoted numbers.
- `--out` — output directory (gitignored via `/out/`). Written on every
  run:
  - `out/tables/<nn>_<name>.tex` — one booktabs table per question, each
    with a `\label{tab:<name>}` and a caption stating the per-cell `n` and
    which SD is reported (seed-only within a density/occupancy cell, or
    mixed occupancy+seed when aggregated across occupancy). Tables that
    include the Broker arm also state the effective `broker.rtt_ms` used.
    A density row whose values are all "--" (e.g. `caos` when every
    leaderless run at that density is invalid) is dropped entirely, and the
    caption notes which density was omitted and why. A caveat sentence like
    "rows are omitted when not yet available" (RSU sensitivity, MCS) is
    only added when a row actually was omitted for that reason.
  - `out/numbers.tex` — `\newcommand` macros (letters-only names) for every
    number the paper prose quotes, so the text never hard-codes a figure.
  - `out/check.txt` — runs per series, valid runs (filter:
    `nr_sim_t_reached >= 179`), which metrics file and which config tree
    (directory) were actually used for each series, config consistency
    report (including which TOML keys are ignored by the core and never
    surfaced in a table), and which tables were skipped and why.

#### Parameters table and config defaults

Some parameters the core actually uses (`gossip.sim_tick_secs`,
`broker.rtt_ms`, `broker.claim_ttl_secs`, `broker.state_log_interval_secs`)
may be absent from `config_used.toml`, in which case the core falls back to
a built-in default. The parameters table (`01_parameters.tex`) reports the
**effective** value for each of these — the TOML value if present, else the
default marked `(default)` — and the config-consistency check in
`load_configs` compares effective values across runs of the same tree (so a
run that relies on the default and a run that explicitly overrides it to
something different are correctly flagged as a mismatch, even though their
raw TOML doesn't collide on any single key). Other TOML keys the core
silently ignores (`claim_ttl_sim_s`, `lease_sim_s`, `uplink_timeout_sim_s`,
`uplink_max_retries`, `uplink_backoff_multiplier`) are never shown in a
table — they're reported in `check.txt` under "ignored by core" instead.

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
