#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_one_config.sh — submit ONE sensitivity configuration to `run_campaign_seeds.slurm`.
#
# Reads config_toml / log_subdir / zmq_offset_base / array from design.json, so
# the driver and the generator cannot drift apart. It only submits; it never
# touches the simulator sources, the campaign data, or the existing configs.
#
# Examples:
#   # full configuration (25 nominal cells), 4 configs concurrently
#   ./run_one_config.sh --design out/design.json --config-id sens_c00 \
#       --base ~/vanet-parking --sif-suffix gossip
#
#   # smoke test: a single nominal cell, to validate plumbing before Phase A
#   ./run_one_config.sh --design out/design.json --config-id sens_c00 \
#       --base ~/vanet-parking --sif-suffix gossip --smoke
#
#   # inspect the sbatch command without submitting
#   ./run_one_config.sh ... --dry-run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DESIGN=""
CONFIG_ID=""
BASE="${VANET_BASE:-$HOME/vanet-parking}"
SIF_SUFFIX="${SIF_SUFFIX:-}"
SUMO_CFG_DIR=""
LOG_BASE_DIR=""
ARRAY_OVERRIDE=""
THROTTLE="25"
TIME="03:00:00"
MEM="32G"
RSU_COUNT="0"
BEACON="0"
SIM_TIME="180"
JOB_NAME="sens"
SMOKE=0
DRY_RUN=0
EXTRA_EXPORT=""

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --design)         DESIGN="$2"; shift 2 ;;
        --config-id)      CONFIG_ID="$2"; shift 2 ;;
        --base)           BASE="$2"; shift 2 ;;
        --sif-suffix)     SIF_SUFFIX="$2"; shift 2 ;;
        --sumo-cfg-dir)   SUMO_CFG_DIR="$2"; shift 2 ;;
        --log-base-dir)   LOG_BASE_DIR="$2"; shift 2 ;;
        --array)          ARRAY_OVERRIDE="$2"; shift 2 ;;
        --throttle)       THROTTLE="$2"; shift 2 ;;
        --time)           TIME="$2"; shift 2 ;;
        --mem)            MEM="$2"; shift 2 ;;
        --rsu-count)      RSU_COUNT="$2"; shift 2 ;;
        --beacon)         BEACON="$2"; shift 2 ;;
        --sim-time)       SIM_TIME="$2"; shift 2 ;;
        --job-name)       JOB_NAME="$2"; shift 2 ;;
        --extra-export)   EXTRA_EXPORT="$2"; shift 2 ;;
        --smoke)          SMOKE=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        usage 0 ;;
        *) echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

[ -n "$DESIGN" ] || { echo "ERROR: --design is required" >&2; exit 1; }
[ -n "$CONFIG_ID" ] || { echo "ERROR: --config-id is required" >&2; exit 1; }
[ -f "$DESIGN" ] || { echo "ERROR: design file not found: $DESIGN" >&2; exit 1; }

BASE="${BASE/#\~/$HOME}"
SUMO_CFG_DIR="${SUMO_CFG_DIR:-$BASE/data/mappa/sumo_cfg_seeds}"
LOG_BASE_DIR="${LOG_BASE_DIR:-$BASE/logs_sensitivity}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$BASE/run_campaign_seeds.slurm}"
[ -f "$SLURM_SCRIPT" ] || { echo "ERROR: SLURM script not found: $SLURM_SCRIPT" >&2; exit 1; }

# ── Read the config record from design.json ─────────────────────────────────
readarray -t FIELDS < <(python3 - "$DESIGN" "$CONFIG_ID" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
cid = sys.argv[2]
rec = next((c for c in d["configs"] if c["config_id"] == cid), None)
if rec is None:
    sys.exit(f"config_id {cid!r} not in {sys.argv[1]}")
base = sys.argv[1].rsplit("/", 1)[0]
print(f"{base}/{rec['config_toml']}")
print(rec["log_subdir"])
print(rec["zmq_offset_base"])
print(f"{rec['array'][0]}-{rec['array'][1]}")
PY
)
CONFIG_TOML="${FIELDS[0]}"
LOG_SUBDIR="${FIELDS[1]}"
OFFSET="${FIELDS[2]}"
ARRAY_RANGE="${FIELDS[3]}"

[ -f "$CONFIG_TOML" ] || { echo "ERROR: config not found: $CONFIG_TOML" >&2; exit 1; }

if [ "$SMOKE" -eq 1 ]; then
    FIRST="${ARRAY_RANGE%%-*}"
    ARRAY="${FIRST}-${FIRST}%1"
    JOB_NAME="${JOB_NAME}-smoke"
else
    ARRAY="${ARRAY_OVERRIDE:-$ARRAY_RANGE}%$THROTTLE"
fi

EXPORT="ALL,SIF_SUFFIX=${SIF_SUFFIX},SUMO_CFG_DIR=${SUMO_CFG_DIR},LOG_BASE_DIR=${LOG_BASE_DIR}"
EXPORT+=",APPTAINERENV_SIM_TIME=${SIM_TIME},APPTAINERENV_RSU_COUNT=${RSU_COUNT},APPTAINERENV_BEACON=${BEACON}"
EXPORT+=",BO_CONFIG=${CONFIG_TOML},ZMQ_OFFSET_BASE=${OFFSET},LOG_SUBDIR=${LOG_SUBDIR}"
[ -n "$EXTRA_EXPORT" ] && EXPORT+=",${EXTRA_EXPORT}"

CMD=(sbatch --job-name="${JOB_NAME}-${CONFIG_ID}" --time="${TIME}" --mem="${MEM}"
     --array="${ARRAY}" --export="${EXPORT}" "${SLURM_SCRIPT}")

echo "config      : ${CONFIG_ID}  (${CONFIG_TOML})"
echo "log subdir  : ${LOG_SUBDIR}   offset base: ${OFFSET}"
echo "array       : ${ARRAY}"
echo "sumo cfg dir: ${SUMO_CFG_DIR}"
echo "log base dir: ${LOG_BASE_DIR}"

if [ "$DRY_RUN" -eq 1 ]; then
    printf 'DRY-RUN: '; printf '%q ' "${CMD[@]}"; printf '\n'
    exit 0
fi

"${CMD[@]}"
