#!/bin/bash
#SBATCH --job-name=postprocess
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/home/oliveriog/vanet-parking/slurm/postprocess-%j.out
#SBATCH --error=/lustre/home/oliveriog/vanet-parking/slurm/postprocess-%j.err
# ──────────────────────────────────────────────────────────────────────────
# postprocess.slurm — run post_process.py + extract_nr_metrics.py +
# aggregate_seeds.py inside the ppbo container, against a given log tree.
#
# Usage:
#   sbatch --export=ALL,LOG_TREE=rerun_r12b4,POST_NAME=post_gossip postprocess.slurm
#   sbatch --export=ALL,LOG_TREE=rerun_r12b4_centralized,POST_NAME=post_broker postprocess.slurm
#
# LOG_TREE   : subdir of ${BASE}/logs_broker/ to read (required)
# POST_NAME  : subdir of ${BASE}/bo_rerun/ to write output into (required)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load apptainer 2>/dev/null || true
if ! command -v apptainer >/dev/null 2>&1; then
    for d in /opt/apptainer-*/bin; do [ -x "$d/apptainer" ] && export PATH="$d:${PATH}" && break; done
fi
command -v apptainer >/dev/null 2>&1 || { echo "ERROR: apptainer not found on PATH"; exit 1; }

BASE="${HOME}/vanet-parking"
PPBO="${BASE}/sif/ppbo.sif"
SCRIPTS="${BASE}/scripts"

: "${LOG_TREE:?ERROR: set LOG_TREE via --export (e.g. rerun_r12b4)}"
: "${POST_NAME:?ERROR: set POST_NAME via --export (e.g. post_gossip)}"
: "${ARM:?ERROR: set ARM via --export (leaderless|centralized)}"

LOG_BASE="${BASE}/logs_broker/${LOG_TREE}"
POST="${BASE}/bo_rerun/${POST_NAME}"
mkdir -p "${POST}"

echo "=== postprocess ==="
echo "LOG_BASE: ${LOG_BASE}"
echo "POST:     ${POST}"
echo ""

for f in extract_nr_metrics.py aggregate_seeds.py; do
    [ -f "${SCRIPTS}/${f}" ] || { echo "ERROR: missing ${SCRIPTS}/${f}"; exit 1; }
done
[ -f "${PPBO}" ] || { echo "ERROR: missing ${PPBO}"; exit 1; }
[ -d "${LOG_BASE}" ] || { echo "ERROR: missing log tree ${LOG_BASE}"; exit 1; }

N_EXP=$(find "${LOG_BASE}" -maxdepth 1 -mindepth 1 -type d | wc -l)
echo "Found ${N_EXP} experiment dirs under ${LOG_BASE}"
[ "${N_EXP}" -gt 0 ] || { echo "ERROR: no experiment dirs found"; exit 1; }

echo ""
echo ">>> 1/3 post_process.py (baked in ppbo /app, already seed-aware)"
apptainer exec \
    "${PPBO}" python3 /app/post_process.py \
    --log-base "${LOG_BASE}" --out "${POST}" --arm "${ARM}"

echo ""
echo ">>> 2/3 extract_nr_metrics.py"
apptainer exec \
    --bind "${SCRIPTS}:/scripts:ro" \
    "${PPBO}" python3 /scripts/extract_nr_metrics.py \
    --log-base "${LOG_BASE}" \
    --summary "${POST}/summary_all_experiments.csv"

echo ""
echo ">>> 3/3 aggregate_seeds.py"
apptainer exec \
    --bind "${SCRIPTS}:/scripts:ro" \
    "${PPBO}" python3 /scripts/aggregate_seeds.py \
    --summary "${POST}/summary_all_experiments.csv"

echo ""
echo "=== done — outputs in ${POST} ==="
ls -lh "${POST}"
