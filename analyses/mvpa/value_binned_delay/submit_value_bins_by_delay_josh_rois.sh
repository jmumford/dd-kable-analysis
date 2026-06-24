#!/bin/bash
#SBATCH --job-name=mvpa_valbins_josh
#SBATCH --output=/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis/logs/mvpa_valbins_josh_%A_%a.out
#SBATCH --error=/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis/logs/mvpa_valbins_josh_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --partition=russpold
#SBATCH --array=1-3

# Array indices 1, 2, 3 correspond to delay bins D1, D2, D3.
# Each job loads ALL subjects, computes binned patterns, and runs LOSO
# for its delay level. All ROIs are processed within the single job.

echo "======================================================================"
echo "SLURM Job ID:       ${SLURM_JOB_ID}"
echo "SLURM Array Task:   ${SLURM_ARRAY_TASK_ID}  (= delay bin)"
echo "Running on node:    $(hostname)"
echo "Starting at:        $(date)"
echo "======================================================================"

PROJECT_ROOT="/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis"
PY_SCRIPT="${PROJECT_ROOT}/analyses/mvpa/value_binned_delay/run_mvpa_value_bins_by_delay.py"

UV_SETUP="${PROJECT_ROOT}/setup_uv_sherlock.sh"
source "${UV_SETUP}"

DELAY_BIN="${SLURM_ARRAY_TASK_ID}"

export PYTHONUNBUFFERED=1

echo ""
echo "Processing delay bin: ${DELAY_BIN}"
echo ""

uv --directory "${PROJECT_ROOT}" run python "${PY_SCRIPT}" \
  --delay-bin "${DELAY_BIN}" \
  --atlas josh_orig \
  --verbose

EXIT_STATUS=$?

echo ""
echo "======================================================================"
if [ ${EXIT_STATUS} -eq 0 ]; then
  echo "SUCCESS: delay_bin=${DELAY_BIN} completed"
else
  echo "FAILED:  delay_bin=${DELAY_BIN} exit code ${EXIT_STATUS}"
fi
echo "Finished at: $(date)"
echo "======================================================================"

exit ${EXIT_STATUS}
