#!/bin/bash
#SBATCH --job-name=traditional_model
#SBATCH --output=/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis/logs/traditional_model_%A_%a.out
#SBATCH --error=/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis/logs/traditional_model_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --partition=russpold
#SBATCH --array=1-454%10

# Traditional Model - SLURM Job Array Submission Script
# Processes all subjects and runs listed in the good_subs_csv file
# Jobs run in parallel with maximum of 10 jobs running simultaneously

echo "======================================================================"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "======================================================================"

# Define paths
PROJECT_ROOT="/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/scripts/dd-kable-analysis"
SCRIPT_DIR="${PROJECT_ROOT}/analyses/traditional_model_analysis"
GOOD_SUBS_CSV="/oak/stanford/groups/russpold/users/buckholtz/DD_Kable/subject_lists/initial_qa_pass_and_mask_pass_subjects_runs.csv"

# Set up uv environment
UV_SETUP="$PROJECT_ROOT/setup_uv_sherlock.sh"
source $UV_SETUP

# Read the specific line from CSV (SLURM_ARRAY_TASK_ID + 1 to skip header)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${GOOD_SUBS_CSV})

# Parse CSV columns: sub_id, session, run
SUB_ID=$(echo $LINE | cut -d',' -f1)
SESSION=$(echo $LINE | cut -d',' -f2)
RUN=$(echo $LINE | cut -d',' -f3)

if [ -z "${SUB_ID}" ] || [ -z "${RUN}" ]; then
    echo "ERROR: Could not parse sub_id and run from CSV"
    echo "Line ${SLURM_ARRAY_TASK_ID}: ${LINE}"
    exit 1
fi

echo ""
echo "Processing:"
echo "  Subject: ${SUB_ID}"
echo "  Session: ${SESSION}"
echo "  Run: ${RUN}"
echo ""

# Run the traditional model analysis using uv
cd ${SCRIPT_DIR}
uv --directory "$PROJECT_ROOT" run "${SCRIPT_DIR}/run_traditional_model.py" "$SUB_ID" "$RUN"

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "======================================================================"
if [ ${EXIT_STATUS} -eq 0 ]; then
    echo "SUCCESS: Analysis completed for sub-${SUB_ID}, run ${RUN}"
else
    echo "FAILED: Analysis failed for sub-${SUB_ID}, run ${RUN} (exit code: ${EXIT_STATUS})"
fi
echo "Finished at: $(date)"
echo "======================================================================"

exit ${EXIT_STATUS}
