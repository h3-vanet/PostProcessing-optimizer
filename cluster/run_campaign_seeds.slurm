#!/bin/bash
#SBATCH --job-name=vanet-sweep-broker
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/home/oliveriog/vanet-parking/slurm/slurm-broker-%A_%a.out
#SBATCH --error=/lustre/home/oliveriog/vanet-parking/slurm/slurm-broker-%A_%a.err
# ──────────────────────────────────────────────────────────────────────────
# SLURM job array — PACKED version, CENTRALIZED-BROKER variant.
# Uses core_multibroker.sif instead of core.sif. Everything else (van3twin,
# bridge, scenario/seed selection, port offsets) is identical to the
# leaderless/gossip script (run_all_scenarios_packed.slurm) — only the core
# image and the default log root differ, so gossip and broker runs can never
# land in the same log tree even if LOG_SUBDIR is forgotten at submit time.
#
# SLURM job array — PACKED version: multiple scenarios share a node.
# No --exclusive. Each task gets a unique ZMQ_PORT_OFFSET so co-located
# scenarios never collide on ZMQ ports (van3twin/bridge/core are offset-aware).
#
# Offset strategy: each task's offset = OFFSET_BASE + TASK_ID * PORT_STRIDE, so
# every array task gets a globally-unique port range and no two scenarios can
# collide no matter how Slurm spreads them across nodes. PORT_STRIDE is large
# enough (200) to never overlap the 6 ports each instance uses.
# All resulting ports must stay below 32768 (the ephemeral range).
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load apptainer 2>/dev/null || true
if ! command -v apptainer >/dev/null 2>&1; then
    for d in /opt/apptainer-*/bin; do [ -x "$d/apptainer" ] && export PATH="$d:${PATH}" && break; done
fi
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found on PATH"; exit 1; }

# ── Paths ────────────────────────────────────────────────────────────────
BASE="${HOME}/vanet-parking"
SIF_DIR="${SIF_DIR:-${BASE}/sif-offset}"            # offset-aware images
# SIF_SUFFIX set -> <img>_<suffix>.sif ; unset -> legacy campaign names
if [ -n "${SIF_SUFFIX:-}" ]; then
    SIF_VAN="${SIF_DIR}/van3twin_${SIF_SUFFIX}.sif"; SIF_BRI="${SIF_DIR}/bridge_${SIF_SUFFIX}.sif"; SIF_COR="${SIF_DIR}/core_${SIF_SUFFIX}.sif"
else
    SIF_VAN="${SIF_DIR}/van3twin_rsu_pc.sif"; SIF_BRI="${SIF_DIR}/bridge_pc.sif"; SIF_COR="${SIF_DIR}/core_rsu_pc.sif"
fi
# Optional per-image override: replace only the core image, keep van3twin/bridge
[ -n "${CORE_SIF_SUFFIX:-}" ] && SIF_COR="${SIF_DIR}/core_${CORE_SIF_SUFFIX}.sif"
for f in "${SIF_VAN}" "${SIF_BRI}" "${SIF_COR}"; do [ -f "$f" ] || { echo "ERROR: missing SIF $f"; exit 1; }; done
DATA_DIR="${BASE}/data"
# Overridable via sbatch --export (e.g. SUMO_CFG_DIR=.../sumo_cfg_seeds for
# multi-seed campaigns). MUST live under ${DATA_DIR}/mappa because that's the
# bind mount root — the container path is derived from the relative position.
SUMO_CFG_DIR="${SUMO_CFG_DIR:-${DATA_DIR}/mappa/sumo_cfg}"
CFG_REL="$(realpath --relative-to="${DATA_DIR}/mappa" "${SUMO_CFG_DIR}")"
case "${CFG_REL}" in
    ..*|/*) echo "ERROR: SUMO_CFG_DIR must be under ${DATA_DIR}/mappa (got ${SUMO_CFG_DIR})"; exit 1 ;;
esac

CONFIG_FILE="${BO_CONFIG:-${DATA_DIR}/Config.toml}"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"; exit 1
fi
echo "Using Config: ${CONFIG_FILE}"

# ── Port offset for this task ────────────────────────────────────────────
# GLOBALLY-UNIQUE offset per array task: task_id * stride. This guarantees no
# two scenarios ever share ports no matter how SLURM places them on nodes.
# (The previous modulo scheme collided when SLURM packed several same-modulo
# tasks onto one node → "Address already in use" → van3twin died at 20s.)
# A batch-slot base (ZMQ_OFFSET_BASE) is added so different configs in a batch
# occupy disjoint port ranges too.
PORT_STRIDE="${PORT_STRIDE:-50}"
OFFSET_BASE="${ZMQ_OFFSET_BASE:-0}"
TASK_ID=${SLURM_ARRAY_TASK_ID}
ZMQ_PORT_OFFSET=$(( OFFSET_BASE + TASK_ID * PORT_STRIDE ))
echo "Offset: base ${OFFSET_BASE} + task ${TASK_ID}*${PORT_STRIDE} = ${ZMQ_PORT_OFFSET}"

# ── Pick scenario ────────────────────────────────────────────────────────
mapfile -t SCENARIOS < <(find "${SUMO_CFG_DIR}" -name "*.sumo.cfg" | sort)
TOTAL=${#SCENARIOS[@]}
if [ "${TASK_ID}" -ge "${TOTAL}" ]; then
    echo "ERROR: task ID ${TASK_ID} >= total scenarios ${TOTAL}"; exit 1
fi
SCENARIO="${SCENARIOS[${TASK_ID}]}"
SCENARIO_NAME="$(basename "${SCENARIO}" .sumo.cfg)"
TRAFFIC_LEVEL=$(echo "${SCENARIO_NAME}" | sed 's/combination_\(.*\)_routes_parking_occupied_.*/\1/')
SEED=$(echo "${SCENARIO_NAME}" | sed 's/.*_seed\([0-9]*\)$/\1/')
MOB_TRACE="scenari_seed/usati/${TRAFFIC_LEVEL}_routes_seed${SEED}.rou.xml"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
# LOG_BASE_DIR lets the parallel BO route logs to a tree separate from the
# sequential BO's logs/ (which gets rm-rf'd between iterations).
# Default here is logs_broker/ (NOT logs/) — the leaderless script defaults to
# logs/, so a forgotten --export can never mix broker and gossip run trees.
LOG_ROOT="${LOG_BASE_DIR:-${BASE}/logs_broker}"
# LOG_SUBDIR (set by parallel BO, e.g. "b0") keeps each batch config's logs apart
LOG_SUBDIR="${LOG_SUBDIR:-}"
if [ -n "${LOG_SUBDIR}" ]; then
    LOG_DIR="${LOG_ROOT}/${LOG_SUBDIR}/${SCENARIO_NAME}/${RUN_ID}"
else
    LOG_DIR="${LOG_ROOT}/${SCENARIO_NAME}/${RUN_ID}"
fi
mkdir -p "${LOG_DIR}/van3twin" "${LOG_DIR}/bridge" "${LOG_DIR}/core"
cp "${CONFIG_FILE}" "${LOG_DIR}/config_used.toml"
echo "${CONFIG_FILE}" > "${LOG_DIR}/config_source.txt"
md5sum "${SIF_VAN}" "${SIF_BRI}" "${SIF_COR}" > "${LOG_DIR}/sif_used.md5"

echo "=== vanet-parking — packed sweep [CENTRALIZED BROKER variant] ==="
echo "Images:     ${SIF_VAN} ${SIF_BRI} ${SIF_COR}"
echo "Array Job:   ${SLURM_ARRAY_JOB_ID}, Task: ${TASK_ID}/${TOTAL}"
echo "Scenario:    ${SCENARIO_NAME}"
echo "Offset:      ${ZMQ_PORT_OFFSET}  (ports: pub=$((5555+ZMQ_PORT_OFFSET)) cmd=$((5558+ZMQ_PORT_OFFSET)) ...)"
echo "Node:        $(hostname)"
echo "Logs:        ${LOG_DIR}"
echo ""

export APPTAINER_CACHEDIR="${BASE}/cache"; mkdir -p "${APPTAINER_CACHEDIR}"

# instance names unique per task (offset alone isn't enough — same offset can
# recur on different nodes, but instance names are per-node, so include TASK_ID)
INST_VAN="van3twin-broker-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
INST_BRI="bridge-broker-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"
INST_COR="core-broker-${SLURM_ARRAY_JOB_ID}-${TASK_ID}"

cleanup() {
    echo ""; echo "=== Stopping instances (task ${TASK_ID}) ==="
    kill "${BRIDGE_PID:-}" "${CORE_PID:-}" 2>/dev/null || true
    sleep 1
    kill -9 "${BRIDGE_PID:-}" "${CORE_PID:-}" 2>/dev/null || true
    apptainer instance stop "${INST_COR}" 2>/dev/null || true
    apptainer instance stop "${INST_BRI}" 2>/dev/null || true
    apptainer instance stop "${INST_VAN}" 2>/dev/null || true
    echo "=== All stopped at $(date) ==="
}
trap cleanup EXIT INT TERM

# ── 1. van3twin ──────────────────────────────────────────────────────────
echo ">>> Starting van3twin (offset ${ZMQ_PORT_OFFSET})..."
export APPTAINERENV_ZMQ_PORT_OFFSET="${ZMQ_PORT_OFFSET}"
export APPTAINERENV_SUMO_FOLDER=/sim/mappa/
export APPTAINERENV_SUMO_CONFIG="/sim/mappa/${CFG_REL}/${SCENARIO_NAME}.sumo.cfg"
export APPTAINERENV_MOB_TRACE="${MOB_TRACE}"
export APPTAINERENV_VEHICLE_VISUALIZER=false
export APPTAINERENV_SUMO_GUI=false
export APPTAINERENV_SUMO_UPDATES=0.1
export APPTAINERENV_SUMO_WAIT=10
export APPTAINERENV_PENETRATION_RATE=1.0
export APPTAINERENV_SIM_TIME="${APPTAINERENV_SIM_TIME:-300}"
export APPTAINERENV_CSV_LOG=/sim/logs/results
export APPTAINERENV_BEACON="${APPTAINERENV_BEACON:-0}"
export APPTAINERENV_RSU_COUNT="${APPTAINERENV_RSU_COUNT:-0}"
export APPTAINERENV_RSU_COVERAGE_LOG="${APPTAINERENV_RSU_COVERAGE_LOG:-/sim/logs/rsu_coverage.csv}"
apptainer instance start \
    --bind "${DATA_DIR}/mappa:/sim/mappa:ro" \
    --bind "${LOG_DIR}/van3twin:/sim/logs" \
    --bind "${DATA_DIR}/entrypoint_pc.sh:/entrypoint.sh:ro" \
    "${SIF_VAN}" "${INST_VAN}"

apptainer exec "instance://${INST_VAN}" \
    bash -c "cd /build/ns-3-dev && /entrypoint.sh" \
    > "${LOG_DIR}/van3twin/stdout.log" 2>&1 &
VAN3TWIN_PID=$!
echo "    van3twin PID: ${VAN3TWIN_PID}"

VAN_PORT=$((5555 + ZMQ_PORT_OFFSET))
VAN_WAIT=300
echo "    Waiting for van3twin port ${VAN_PORT} (up to ${VAN_WAIT}s)..."
van_ready=0
for i in $(seq 1 "${VAN_WAIT}"); do
    if ! kill -0 "${VAN3TWIN_PID}" 2>/dev/null; then
        echo "    ERROR: van3twin exited before binding (after ${i}s)"
        tail -30 "${LOG_DIR}/van3twin/stdout.log"; exit 1
    fi
    if grep -q "bound on tcp://\*:${VAN_PORT}" "${LOG_DIR}/van3twin/stdout.log" 2>/dev/null; then
        echo "    van3twin ready (${i}s)"; van_ready=1; break
    fi
    sleep 1
done
[ "${van_ready}" -ne 1 ] && { echo "    ERROR: van3twin did not bind ${VAN_PORT} in ${VAN_WAIT}s"; tail -30 "${LOG_DIR}/van3twin/stdout.log"; exit 1; }

# ── 2. bridge ────────────────────────────────────────────────────────────
echo ">>> Starting bridge..."
export APPTAINERENV_ZMQ_PORT_OFFSET="${ZMQ_PORT_OFFSET}"
export APPTAINERENV_SUMO_HOST=localhost
export APPTAINERENV_ZMQ_VAN3TWIN_PORT=5555
export APPTAINERENV_ZMQ_PUSH_PORT=5557
export APPTAINERENV_ZMQ_CMD_PORT=5558
export APPTAINERENV_PYTHONUNBUFFERED=1

OCC_LEVEL=$(echo "${SCENARIO_NAME}" | sed 's/.*_parking_occupied_\([0-9]*\)_seed[0-9]*$/\1/')
export APPTAINERENV_PARKING_XML_PATH=/sim/mappa/parked_seed/parking_occupied_${OCC_LEVEL}_seed${SEED}.add.xml

export APPTAINERENV_BRIDGE_LOG_DIR=/sim/logs

apptainer instance start \
    --bind "${LOG_DIR}/bridge:/sim/logs" \
    --bind "${DATA_DIR}/mappa:/sim/mappa:ro" \
    "${SIF_BRI}" "${INST_BRI}"

apptainer exec "instance://${INST_BRI}" \
    python3 /app/python/bridge.py \
    > "${LOG_DIR}/bridge/stdout.log" 2>&1 &
BRIDGE_PID=$!
echo "    bridge PID: ${BRIDGE_PID}"
sleep 3

# ── 3. core ──────────────────────────────────────────────────────────────
echo ">>> Starting core..."
export APPTAINERENV_ZMQ_PORT_OFFSET="${ZMQ_PORT_OFFSET}"
apptainer instance start \
    --bind "${CONFIG_FILE}:/sim/Config.toml:ro" \
    --bind "${LOG_DIR}/core:/sim/logs" \
    "${SIF_COR}" "${INST_COR}"

apptainer exec "instance://${INST_COR}" \
    bash -c "cd /sim && exec /usr/local/bin/core" \
    > "${LOG_DIR}/core/stdout.log" 2>&1 &
CORE_PID=$!
echo "    core PID: ${CORE_PID}"

# ── 4. wait ──────────────────────────────────────────────────────────────
echo ""; echo "=== Running: ${SCENARIO_NAME} (offset ${ZMQ_PORT_OFFSET}) ==="
wait ${VAN3TWIN_PID} 2>/dev/null
VAN3TWIN_EXIT=$?
echo "van3twin exited with code ${VAN3TWIN_EXIT}"
sleep 3
echo "=== Scenario ${SCENARIO_NAME} complete — exit ${VAN3TWIN_EXIT} — $(date) ==="
