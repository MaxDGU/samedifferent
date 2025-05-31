#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# This script orchestrates an "ideal experiment" with data parity
# between single-task and MAML training for a specific seed and architecture.

# --- Configuration ---
SEED_VAL=42                                                           # Seed for all runs
ARCH="conv6"                                                          # Architecture to test: conv2, conv4, or conv6
PYTHON_EXEC="python"                                                  # Use "python" or specify path to your python executable
# PYTHON_EXEC="/Users/maxgupta/Desktop/Princeton/CoCoSci_Lab/samedifferent/same_different_paper/metasamedifferent/venv/bin/python" # Example: Update with your actual Python path if not in PATH or if using a venv

# --- Paths ---
# Assuming this script is in the project root. Adjust if not.
CODE_ROOT_DIR="." 
DATA_DIR_H5="${CODE_ROOT_DIR}/data/meta_h5/pb"                        # Path to HDF5 files

# Base output directory for this specific experiment run
BASE_EXP_DIR_NAME="ideal_datapar_exp_seed${SEED_VAL}_arch${ARCH}"
TOP_LEVEL_OUTPUT_DIR="${CODE_ROOT_DIR}/results/${BASE_EXP_DIR_NAME}"

SINGLETASK_RESULTS_ROOT="${TOP_LEVEL_OUTPUT_DIR}/single_task_runs"    # Base for single-task outputs
MAML_RESULTS_ROOT="${TOP_LEVEL_OUTPUT_DIR}/maml_runs"                 # Base for MAML script outputs
PCA_PLOTS_DIR="${TOP_LEVEL_OUTPUT_DIR}/pca_plots"                     # For PCA plots

# Tasks to iterate over for single-task training
PB_TASKS=('regular' 'lines' 'open' 'wider_line' 'scrambled' 'random_color' 'arrows' 'irregular' 'filled' 'original')

# --- Training Parameters ---
# SINGLETASK_TRAIN_SCRIPT_PATH="${CODE_ROOT_DIR}/single_task/train_single_task_pb.py"
SINGLETASK_TRAIN_SCRIPT_PATH="${CODE_ROOT_DIR}/single_task/train_single_task_pb.py" # Adjusted to be relative to CODE_ROOT_DIR
SINGLETASK_NUM_EPOCHS=10 # MODIFIED from 1 to 20
SINGLETASK_BATCH_SIZE=32
SINGLETASK_LR=0.001
SINGLETASK_PATIENCE=10 # Corresponds to args.patience in train_single_task_pb.py (default is 10)
SINGLETASK_VAL_FREQ=5  # Corresponds to args.val_freq in train_single_task_pb.py (default is 5)
# Note: train_single_task_pb.py itself has a default --epochs of 100.
# We are overriding it here for the ideal experiment.

# MAML_TRAIN_SCRIPT_PATH="${CODE_ROOT_DIR}/all_tasks/experiment_all_tasks_fomaml.py"
MAML_TRAIN_SCRIPT_PATH="${CODE_ROOT_DIR}/all_tasks/experiment_all_tasks_fomaml.py" # Adjusted
MAML_NUM_EPOCHS=50 # Number of meta-epochs
# ... other MAML parameters from previous discussions, ensure they are appropriate for data parity
MAML_META_BATCH_SIZE=16          # Was 4, then 16. Set to target desired num_tasks_per_epoch
MAML_ADAPTATION_STEPS=1        # Inner loop updates
MAML_NUM_ADAPTATION_SAMPLES=32 # k_shot + q_query for inner loop (e.g. 16+16 or 4+28)
                                 # This is (num_samples_per_task_in_meta_batch // 2 for support, // 2 for query if balanced)
                                 # Or more precisely, it's k-shot + k-query (shots for adaptation, query for meta-loss)
                                 # For PBDataset, each episode is 2*support_size. So if support_size is 4, an episode is 8.
                                 # If we want 16 samples for support and 16 for query for MAML adaptation, this means 4 episodes of support_size=4.
MAML_TASK_SAMPLER_ARGS="--pb_sample_num_episodes 4 --pb_support_size 4" # Makes 4*4*2=32 samples per task in meta-batch if using PBDataset directly
                                                                     # For SameDifferentDataset, it's more direct:
                                                                     # num_samples_per_class_train = (MAML_NUM_ADAPTATION_SAMPLES // 2) // num_classes (if binary, //2)
                                                                     # num_samples_per_class_test = (MAML_NUM_ADAPTATION_SAMPLES // 2) // num_classes
MAML_LR=0.001

# --- Analysis Parameters ---
# ANALYSIS_SCRIPT_PATH="${CODE_ROOT_DIR}/weight_space_analysis/analyze_pb_weights.py"
ANALYSIS_SCRIPT_PATH="${CODE_ROOT_DIR}/weight_space_analysis/analyze_all_weights.py" # Using the newer script

# Derived paths for results based on seed and arch
RESULTS_BASE_DIR_IDEAL_EXP="${CODE_ROOT_DIR}/results/ideal_datapar_exp_seed${SEED_VAL}_arch${ARCH}"
SINGLETASK_RESULTS_ROOT="${RESULTS_BASE_DIR_IDEAL_EXP}/single_task_runs"
MAML_RESULTS_ROOT="${RESULTS_BASE_DIR_IDEAL_EXP}/maml_runs"
MAML_EXPERIMENT_DIR_SPECIFIC="${MAML_RESULTS_ROOT}/exp_all_tasks_fomaml_${ARCH}_seed${SEED_VAL}_ideal_datapar" # Specific name
PCA_PLOT_OUTPUT_DIR="${RESULTS_BASE_DIR_IDEAL_EXP}/pca_plots"

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
echo "=== Starting Ideal Experiment with Data Parity ==="
echo "=========================================================="
echo "Seed for all runs: ${SEED_VAL}"
echo "Architecture: ${ARCH}"
echo "Single-Task Output Base: ${SINGLETASK_RESULTS_ROOT}"
echo "MAML Output Base: ${MAML_RESULTS_ROOT}"
echo "PCA Plot Output: ${PCA_PLOTS_DIR}"
echo "HDF5 Data Dir: ${DATA_DIR_H5}"
echo "Python Executable: ${PYTHON_EXEC}"
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
# experiment_all_tasks_fomaml.py saves outputs to: <output_base_dir>/exp_all_tasks_fomaml_<arch>_seed<seed>_Svar_Q3_<timestamp>/
${PYTHON_EXEC} ${MAML_TRAIN_SCRIPT_PATH} \
    --architecture "${ARCH}" \
    --seed "${SEED_VAL}" \
    --epochs "${MAML_NUM_EPOCHS}" \
    --num_meta_batches_per_epoch "${MAML_NUM_ADAPTATION_SAMPLES}" \
    --meta_batch_size "${MAML_META_BATCH_SIZE}" \
    --inner_lr "${MAML_LR}" \
    --outer_lr 0.0001 \
    --first_order \
    --output_base_dir "${MAML_RESULTS_ROOT}" \
    --data_dir "${DATA_DIR_H5}" \
    --task_sampler_args "${MAML_TASK_SAMPLER_ARGS}"

# Find the exact MAML experiment directory created
# (Assumes only one experiment run for this specific arch/seed in MAML_RESULTS_ROOT by this script invocation)
MAML_ACTUAL_EXP_DIR=$(find "${MAML_RESULTS_ROOT}" -type d -name "exp_all_tasks_fomaml_${ARCH}_seed${SEED_VAL}_*" -print -quit)

if [ -z "${MAML_ACTUAL_EXP_DIR}" ]; then
    echo "ERROR: Could not find MAML experiment directory in ${MAML_RESULTS_ROOT}"
    echo "       Expected pattern: exp_all_tasks_fomaml_${ARCH}_seed${SEED_VAL}_*"
    exit 1
fi
echo "  MAML Experiment Output Directory: ${MAML_ACTUAL_EXP_DIR}"
echo "--- MAML Training Finished ---"
echo "----------------------------------------------------------"

# --- 3. PCA Analysis ---
echo "\n>>> Running PCA Analysis for ${ARCH}, seed ${SEED_VAL}..."
# analyze_all_weights.py will need to be updated to use these new arguments
# to correctly load one initial model, 10 single-task final, and 1 MAML final.
${PYTHON_EXEC} ${ANALYSIS_SCRIPT_PATH} \
    --results_dir "${SINGLETASK_RESULTS_ROOT}" \
    --output_plot_dir "${PCA_PLOTS_DIR}" \
    --specific_arch_to_analyze "${ARCH}" \
    --specific_seed_to_analyze "${SEED_VAL}" \
    --specific_maml_experiment_path "${MAML_ACTUAL_EXP_DIR}" \
    # You might want to add common arguments like --maml_runs_base_dir if parts of the script still expect it, 
    # even if specific_maml_experiment_path overrides the MAML loading.
    # For this ideal run, --maml_runs_base_dir might not be strictly needed if the specific path is used.

echo "--- PCA Analysis Finished ---"
echo "=========================================================="
echo "=== Ideal Experiment Script Complete ==="
echo "=========================================================="
echo "Final Outputs:"
echo "  Single-task models are in subdirectories of: ${SINGLETASK_RESULTS_ROOT}/${ARCH}/seed_${SEED_VAL}/"
echo "  MAML model is in: ${MAML_ACTUAL_EXP_DIR}"
echo "  PCA plots should be in: ${PCA_PLOTS_DIR}"
echo "==========================================================" 