# Bayesian Optimization — operating guide

Optimizes the coordination parameters by repeatedly running the 20-scenario
campaign with different `Config.toml` candidates and minimizing a weighted
objective. Runs on the **login node** inside **tmux**.

## What gets optimized

6 parameters (β is derived, not searched):

| parameter            | range        | notes                          |
|----------------------|--------------|--------------------------------|
| `alfa`               | 0.3 – 0.9    | `beta = 1 - alfa`              |
| `gossip_interval_ms` | 200 – 1000   | integer                        |
| `neighbor_k`         | 1 – 4        | integer                        |
| `t_base_ms`          | 100 – 500    | real                           |
| `cluster_resolution` | 8 – 10       | integer (H3)                   |
| `spot_resolution`    | 13 – 15      | integer, forced `> cluster`    |

**Objective (minimized), weighted 2× for occupancy ≥ 85:**
```
0.40·(1 − success·(1−double)) + 0.35·(1 − jain) + 0.25·(latency_p50/2000)
```

## One-time setup

```bash
# on the login node, in your project dir
python3 -m venv venv && source venv/bin/activate
pip install scikit-optimize pandas

# make sure the patched SLURM is in place (reads BO_CONFIG)
cp run_all_scenarios.slurm ~/vanet-parking/run_all_scenarios.slurm

# sanity: a single iteration's array must still run with the default config
sbatch --array=0-0 ~/vanet-parking/run_all_scenarios.slurm
```

Confirm the SLURM line `Using Config: .../Config.toml` appears in the log — that
proves the BO_CONFIG plumbing works (falls back to the default when unset).

## Run it (in tmux)

```bash
tmux new -s bo
source venv/bin/activate
python3 bo_optimize.py --n-calls 60 --n-initial 12 --max-parallel 10
# detach:  Ctrl-b  then  d
# reattach: tmux attach -t bo
```

## What it produces (under ~/vanet-parking/bo_runs/)

- `config_iter_NNN.toml` — the candidate config for each iteration
- `post_iter_NNN/` — post-processing output for that iteration
- `iter_NNN_<ts>.tar.gz` — **download these**: logs + post + config + objective
- `bo_results.csv` — one row per iteration (params + objective), the BO trace
- `best_params.json` — the winning configuration at the end

## Monitoring

```bash
tmux attach -t bo                       # watch live
tail -f ~/vanet-parking/bo_runs/bo_results.csv
squeue -u $USER                         # the array currently running
```

## Cost expectation

6 dimensions → ~60–90 evaluations to converge well. Each evaluation = 20 runs.
At ~30 min/run with 10 parallel, one evaluation ≈ 1 h, so a full BO is **days**.
The loop checkpoints every iteration to `bo_results.csv`, and each iteration is
archived, so an interruption loses at most the in-flight iteration.

## Before you start — ask HPRC

Email Carlisle one line: confirm it's OK to run a lightweight Python orchestrator
on the login node for a few days (it only does sbatch + squeue polling + reads a
CSV; all compute is in the jobs). Almost certainly fine, but worth confirming so
they don't kill the process.

## Phase 2 (later): per-traffic-profile optimization

Once the global optimum is found, repeat the BO four times restricting the
campaign to one traffic profile each (minimo / normale / trafficato / caos), then
compare. Since the vehicle can estimate local density from its k-ring neighbors,
a meaningful gap between global and per-profile optima motivates an adaptive,
regime-aware configuration — a stronger thesis contribution.
```
