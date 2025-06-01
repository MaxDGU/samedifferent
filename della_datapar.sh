#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# This script orchestrates an "ideal experiment" with data parity
# between single-task and MAML training for a specific seed and architecture.

# --- Environment Setup for Della ---
export PYTHONPATH=$PYTHONPATH:/scratch/gpfs/mg7411/samedifferent # Add project root to Python path

# --- Configuration ---
SEED_VAL=42                                                           # Seed for all runs
ARCH="conv6"                                                          # Architecture to test: conv2, conv4, or conv6
# PYTHON_EXEC="/opt/anaconda3/envs/tensorflow/bin/python"               # Python executable on Della (Old)
PYTHON_EXEC="python"                                                  # Rely on conda env's Python

# --- Paths ---
# Assuming this script is in the project root on Della: /scratch/gpfs/mg7411/samedifferent
CODE_ROOT_DIR="." 
# DATA_DIR_H5="/scratch/gpfs/mg7411/data/pb/pb"                        # Old Della path
DATA_DIR_H5="/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb"      # Corrected Path to HDF5 files on Della

# Base output directory for this specific experiment run
# Outputs will be relative to the CODE_ROOT_DIR on Della
BASE_EXP_DIR_NAME="ideal_datapar_exp_seed${SEED_VAL}_arch${ARCH}_della" # Added _della to distinguish
TOP_LEVEL_OUTPUT_DIR="${CODE_ROOT_DIR}/results/${BASE_EXP_DIR_NAME}"

SINGLETASK_RESULTS_ROOT="${TOP_LEVEL_OUTPUT_DIR}/single_task_runs"    # Base for single-task outputs
MAML_RESULTS_ROOT="${TOP_LEVEL_OUTPUT_DIR}/maml_runs"                 # Base for MAML script outputs
PCA_PLOTS_DIR="${TOP_LEVEL_OUTPUT_DIR}/pca_plots"                     # For PCA plots

# Tasks to iterate over for single-task training
PB_TASKS=('regular' 'lines' 'open' 'wider_line' 'scrambled' 'random_color' 'arrows' 'irregular' 'filled' 'original')

# --- Training Parameters ---
SINGLETASK_TRAIN_SCRIPT_PATH="${CODE_ROOT_DIR}/single_task/train_single_task_pb.py" 
SINGLETASK_NUM_EPOCHS=10 
SINGLETASK_BATCH_SIZE=32
SINGLETASK_LR=0.001
SINGLETASK_PATIENCE=10 
SINGLETASK_VAL_FREQ=5  

MAML_TRAIN_SCRIPT_PATH="${CODE_ROOT_DIR}/all_tasks/experiment_all_tasks_fomaml.py" 
MAML_NUM_EPOCHS=50 
MAML_META_BATCH_SIZE=16          
MAML_ADAPTATION_STEPS=5        # Remains 5 for training
MAML_NUM_ADAPTATION_SAMPLES=32 
MAML_LR=0.001                  # Corresponds to inner_lr = 1e-3

# --- Analysis Parameters ---
ANALYSIS_SCRIPT_PATH="${CODE_ROOT_DIR}/weight_space_analysis/analyze_all_weights.py" 

# Derived paths for results based on seed and arch
RESULTS_BASE_DIR_IDEAL_EXP="${TOP_LEVEL_OUTPUT_DIR}" # Adjusted as TOP_LEVEL_OUTPUT_DIR already includes seed/arch
# SINGLETASK_RESULTS_ROOT is already defined above
# MAML_RESULTS_ROOT is already defined above
MAML_EXPERIMENT_DIR_SPECIFIC="${MAML_RESULTS_ROOT}/exp_all_tasks_fomaml_${ARCH}_seed${SEED_VAL}_ideal_datapar" 
# PCA_PLOT_OUTPUT_DIR is already defined above

# Data directory for HDF5 files
HDF5_DATA_DIR="${DATA_DIR_H5}" # Uses var from top


# --- Ensure Python executable is found ---
if ! command -v ${PYTHON_EXEC} &> /dev/null
then
    echo "Error: Python executable '${PYTHON_EXEC}' not found. Please set PYTHON_EXEC correctly in the script."
    exit 1
fi
echo "Using Python: $(command -v ${PYTHON_EXEC})"

# --- Create output directories ---
mkdir -p "${SINGLETASK_RESULTS_ROOT}"
mkdir -p "${MAML_RESULTS_ROOT}"
mkdir -p "${PCA_PLOTS_DIR}"

echo "=========================================================="
echo "=== Starting Ideal Experiment with Data Parity (Della) ==="
echo "=========================================================="
echo "Seed for all runs: ${SEED_VAL}"
echo "Architecture: ${ARCH}"
echo "Single-Task Output Base: ${SINGLETASK_RESULTS_ROOT}"
echo "MAML Output Base: ${MAML_RESULTS_ROOT}"
echo "PCA Plot Output: ${PCA_PLOTS_DIR}"
echo "HDF5 Data Dir: ${HDF5_DATA_DIR}"
echo "Python Executable: ${PYTHON_EXEC}"
echo "PYTHONPATH: $PYTHONPATH" # Print PYTHONPATH for verification
echo "----------------------------------------------------------"

# --- Run Single-Task Trainings ---
echo -e "\n>>> Running Single-Task Trainings (${SINGLETASK_NUM_EPOCHS} epoch(s) each for ${ARCH}, seed ${SEED_VAL})..."
for task_name in "${PB_TASKS[@]}"; do
    echo "  Starting single-task training for: Task=${task_name}"
    ${PYTHON_EXEC} ${SINGLETASK_TRAIN_SCRIPT_PATH} \
        --task "${task_name}" \
        --architecture "${ARCH}" \
        --seed "${SEED_VAL}" \
        --epochs "${SINGLETASK_NUM_EPOCHS}" \
        --batch_size "${SINGLETASK_BATCH_SIZE}" \
        --lr "${SINGLETASK_LR}" \
        --patience "${SINGLETASK_PATIENCE}" \
        --val_freq "${SINGLETASK_VAL_FREQ}" \
        --data_dir "${HDF5_DATA_DIR}" \
        --output_dir "${SINGLETASK_RESULTS_ROOT}"
    echo "  Finished single-task training for: Task=${task_name}"
done
echo ">>> Finished all Single-Task Trainings."
echo "----------------------------------------------------------"

# --- Run MAML Training (All Tasks) ---
echo "\n>>> Running MAML Training (1 Super Epoch for ${ARCH}, seed ${SEED_VAL})..."
echo "    Using MAML Epochs: ${MAML_NUM_EPOCHS}, Meta-Batches/Epoch: ${MAML_NUM_ADAPTATION_SAMPLES}"

# --- Capture MAML script output to find the exact experiment path ---
MAML_RUN_LOG_TEMP=$(mktemp)
echo "Capturing MAML run output to ${MAML_RUN_LOG_TEMP}"

( ${PYTHON_EXEC} ${MAML_TRAIN_SCRIPT_PATH} \
    --architecture "${ARCH}" \
    --seed "${SEED_VAL}" \
    --epochs "${MAML_NUM_EPOCHS}" \
    --num_meta_batches_per_epoch "${MAML_NUM_ADAPTATION_SAMPLES}" \
    --meta_batch_size "${MAML_META_BATCH_SIZE}" \
    --inner_lr "${MAML_LR}" \
    --outer_lr 0.0001 \
    --adaptation_steps "${MAML_ADAPTATION_STEPS}" \
    --adaptation_steps_test 15 \
    --first_order \
    --output_base_dir "${MAML_RESULTS_ROOT}" \
    --data_dir "${HDF5_DATA_DIR}" \
    --track_bn_stats # Explicitly add flag
) 2>&1 | tee "${MAML_RUN_LOG_TEMP}"

MAML_ACTUAL_EXP_DIR=$(grep "Results will be saved to:" "${MAML_RUN_LOG_TEMP}" | sed -n 's/Results will be saved to: //p')

if [ -z "${MAML_ACTUAL_EXP_DIR}" ]; then
    echo "ERROR: Could not determine MAML experiment directory from script output stored in ${MAML_RUN_LOG_TEMP}."
    echo "Full log from MAML script:"
    cat "${MAML_RUN_LOG_TEMP}"
    rm -f "${MAML_RUN_LOG_TEMP}"
    exit 1
else
    echo "Successfully captured MAML experiment directory: ${MAML_ACTUAL_EXP_DIR}"
    # Optional: Clean up the log file if successfully parsed and not needed for other debugging.
    # If you want to keep it for other reasons, comment out the rm line.
    rm -f "${MAML_RUN_LOG_TEMP}"
fi
# --- End MAML script output capture ---

# Original find command - replaced by direct capture above
# MAML_ACTUAL_EXP_DIR=$(find "${MAML_RESULTS_ROOT}" -type d -name "exp_all_tasks_fomaml_${ARCH}_seed${SEED_VAL}_*" -print0 | xargs -0 ls -td | head -n 1)

# Check if MAML_ACTUAL_EXP_DIR is still empty after the new method (should have exited if grep failed)
if [ -z "${MAML_ACTUAL_EXP_DIR}" ]; then
    echo "ERROR: MAML_ACTUAL_EXP_DIR is empty even after attempting to capture from script output."
    exit 1
fi
echo "  MAML Experiment Output Directory (to be used by PCA): ${MAML_ACTUAL_EXP_DIR}"
echo "--- MAML Training Finished ---"
echo "----------------------------------------------------------"

# --- 3. PCA Analysis ---
echo "\n>>> Running PCA Analysis for ${ARCH}, seed ${SEED_VAL}..."
${PYTHON_EXEC} ${ANALYSIS_SCRIPT_PATH} \
    --results_dir "${SINGLETASK_RESULTS_ROOT}" \
    --output_plot_dir "${PCA_PLOTS_DIR}" \
    --specific_arch_to_analyze "${ARCH}" \
    --specific_seed_to_analyze "${SEED_VAL}" \
    --specific_maml_experiment_path "${MAML_ACTUAL_EXP_DIR}" 

echo "--- PCA Analysis Finished ---"
echo "=========================================================="
echo "=== Ideal Experiment Script Complete (Della) ==="
echo "=========================================================="
echo "Final Outputs:"
echo "  Single-task models are in subdirectories of: ${SINGLETASK_RESULTS_ROOT}/${ARCH}/seed_${SEED_VAL}/"
echo "  MAML model is in: ${MAML_ACTUAL_EXP_DIR}"
echo "  PCA plots should be in: ${PCA_PLOTS_DIR}"
echo "==========================================================" 