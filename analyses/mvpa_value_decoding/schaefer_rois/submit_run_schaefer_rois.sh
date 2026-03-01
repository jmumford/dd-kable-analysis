#!/bin/bash
#SBATCH --job-name=mvpa_schaefer200
#SBATCH --output=/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis/logs/mvpa_schaefer200_%A_%a.out
#SBATCH --error=/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis/logs/mvpa_schaefer200_%A_%a.err
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
SCRIPT_DIR="${PROJECT_ROOT}/analyses/mvpa_value_decoding/schaefer_rois"
PY_SCRIPT="${SCRIPT_DIR}/run_mvpa.py"

SUB_LIST="/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/subject_lists/mvpa_subject_list.txt"

# Set up uv environment
UV_SETUP="${PROJECT_ROOT}/setup_uv_sherlock.sh"
source "${UV_SETUP}"

# 1-based array index -> line number in text file
SUB_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SUB_LIST}")

if [ -z "${SUB_ID}" ]; then
  echo "ERROR: Could not read sub_id from ${SUB_LIST} at line ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

echo ""
echo "Processing subject: ${SUB_ID}"
echo "Python script: ${PY_SCRIPT}"
echo ""

cd "${SCRIPT_DIR}"

# Run Schaefer-200 (add/remove --verbose as desired)
uv --directory "${PROJECT_ROOT}" run python "${PY_SCRIPT}" \
  --sub-id "${SUB_ID}" \
  --atlas schaefer200 \
  --y-col amount \
  --verbose

EXIT_STATUS=$?

echo ""
echo "======================================================================"
if [ ${EXIT_STATUS} -eq 0 ]; then
  echo "SUCCESS: MVPA completed for sub-${SUB_ID}"
else
  echo "FAILED: MVPA failed for sub-${SUB_ID} (exit code: ${EXIT_STATUS})"
fi
echo "Finished at: $(date)"
echo "======================================================================"

exit ${EXIT_STATUS}