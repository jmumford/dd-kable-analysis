#!/bin/bash
#SBATCH --job-name=mvpa_pauli_sv_no_rt
#SBATCH --output=/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis/logs/mvpa_pauli_sv_no_rt_%A_%a.out
#SBATCH --error=/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis/logs/mvpa_pauli_sv_no_rt_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=russpold
#SBATCH --array=1-123%10

echo "======================================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "======================================================================"

PROJECT_ROOT="/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis"
PY_SCRIPT="${PROJECT_ROOT}/analyses/mvpa/run_mvpa_ridge.py"

SUB_LIST="/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/subject_lists/mvpa_subject_list.txt"

UV_SETUP="${PROJECT_ROOT}/setup_uv_sherlock.sh"
source "${UV_SETUP}"

SUB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SUB_LIST}")

if [ -z "${SUB_ID}" ]; then
  echo "ERROR: Could not read sub_id from ${SUB_LIST} at line ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo ""
echo "Processing subject: ${SUB_ID}"
echo "Python script: ${PY_SCRIPT}"
echo ""

uv --directory "${PROJECT_ROOT}" run python "${PY_SCRIPT}" \
  --sub-id "${SUB_ID}" \
  --atlas pauli_rois \
  --y-col SV_LL \
  --beta-series-dir beta_series_no_rt \
  --analysis-tag pauli_rois_SV_LL_no_rt \
  --verbose

EXIT_STATUS=$?

echo ""
echo "======================================================================"
if [ ${EXIT_STATUS} -eq 0 ]; then
  echo "SUCCESS: MVPA (SV_LL no-RT) completed for sub-${SUB_ID}"
else
  echo "FAILED: MVPA (SV_LL no-RT) failed for sub-${SUB_ID} (exit code: ${EXIT_STATUS})"
fi
echo "Finished at: $(date)"
echo "======================================================================"

exit ${EXIT_STATUS}
