# Vanilla SGD Architecture Validation - Multi-Seed Analysis

## Overview
This experiment validates vanilla SGD performance across three CNN architectures (conv2, conv4, conv6) using **5 different random seeds** to obtain robust statistical results with mean ± standard deviation.

## Files Created

### Multi-Seed Array Job System
- `run_vanilla_sgd_validation_array.slurm` - SLURM array job (15 jobs: 5 seeds × 3 architectures)
- `run_vanilla_sgd_validation_array_test.slurm` - Quick test version (6 jobs: 2 seeds × 3 architectures)
- `scripts/aggregate_vanilla_sgd_results.py` - Aggregation script with statistics and plots
- `run_aggregate_results.slurm` - SLURM job for aggregation
- `run_full_vanilla_sgd_validation.sh` - Complete pipeline helper script

### Original Single-Seed Scripts
- `scripts/validate_vanilla_sgd_architectures_fixed.py` - Main validation script
- `run_vanilla_sgd_validation_fixed.slurm` - Single-seed SLURM script

## Experimental Design

### Seeds and Jobs
| Job Array Index | Seed | Architecture | Description |
|----------------|------|-------------|-------------|
| 0-4 | 42-46 | CONV2 | 5 seeds for conv2 |
| 5-9 | 42-46 | CONV4 | 5 seeds for conv4 |
| 10-14 | 42-46 | CONV6 | 5 seeds for conv6 |

**Total: 15 parallel jobs**

### Architecture Details
| Architecture | Parameters | Description |
|-------------|------------|-------------|
| Conv2CNN | 15,484,320 | 2-layer CNN: 6, 12 filters with 2x2 kernels |
| Conv4CNN | 10,090,206 | 4-layer CNN: 12, 24, 48, 96 filters |
| Conv6CNN | 8,299,908 | 6-layer CNN: 18, 36, 72, 144, 288, 576 filters |

## Running the Complete Experiment

### Option 1: Automated Pipeline (Recommended)
```bash
# Runs array job + aggregation with dependency
./run_full_vanilla_sgd_validation.sh
```

### Option 2: Manual Step-by-Step

**1. Submit Array Job:**
```bash
sbatch run_vanilla_sgd_validation_array.slurm
```

**2. Monitor Progress:**
```bash
squeue -u $(whoami)
```

**3. After All Jobs Complete, Aggregate Results:**
```bash
sbatch run_aggregate_results.slurm
```

### Option 3: Quick Test First
```bash
# Test with 2 seeds, 5 epochs
sbatch run_vanilla_sgd_validation_array_test.slurm

# Then aggregate test results
python scripts/aggregate_vanilla_sgd_results.py \
    --results_dir results/vanilla_sgd_validation_test_array \
    --seeds 42 43 \
    --architectures CONV2 CONV4 CONV6
```

### Option 4: With Job Dependencies (Automatic)
```bash
# Submit array job and get job ID
JOB_ID=$(sbatch --parsable run_vanilla_sgd_validation_array.slurm)

# Submit aggregation job that waits for array job to complete
sbatch --dependency=afterok:$JOB_ID run_aggregate_results.slurm
```

## Expected Results

### Statistical Output
Results will show **mean ± standard deviation** across 5 seeds:

```
Architecture  Parameters   Test Acc (%)    Val Acc (%)     Seeds
CONV2        15,484,320    50.2±1.3%       49.8±1.1%       5
CONV4        10,090,206    50.5±0.9%       50.1±1.4%       5  
CONV6         8,299,908    49.7±1.2%       50.3±0.8%       5
```

### Interpretation
- **~50% accuracy**: Confirms chance level performance
- **Low standard deviation**: Shows consistent poor performance
- **Validates bar charts**: Confirms vanilla SGD struggles on PB tasks

## Output Files

### Individual Seed Results
```
results/vanilla_sgd_validation_array/
├── seed_42/validation_results.json
├── seed_43/validation_results.json
├── seed_44/validation_results.json
├── seed_45/validation_results.json
└── seed_46/validation_results.json
```

### Aggregated Results
```
results/vanilla_sgd_validation_array/
├── aggregated_results.json                    # Complete statistics
├── vanilla_sgd_aggregated_results.png         # Main comparison plot
├── vanilla_sgd_aggregated_results.pdf         # Publication-ready plot
├── vanilla_sgd_seed_distribution.png          # Individual seed scatter
└── vanilla_sgd_seed_distribution.pdf          # Seed distribution plot
```

## Key Features

### Robust Statistics
- **Mean ± Standard Deviation** across 5 seeds
- **Individual seed scatter plots** showing distribution
- **Error bars** on main comparison plots
- **Chance level reference line** at 50%

### Professional Plots
- **Publication-ready figures** (PNG + PDF)
- **Mean comparison** with error bars
- **Seed distribution analysis**
- **Clear statistical annotations**

### Automated Pipeline
- **SLURM array jobs** for parallel execution
- **Automatic aggregation** with job dependencies
- **Comprehensive error checking**
- **Progress monitoring** and logging

## Validation Purpose

This multi-seed analysis provides:

1. **Robust Evidence**: Statistical significance with proper error estimates
2. **Reproducibility**: Multiple seeds ensure results aren't due to random chance
3. **Publication Quality**: Professional plots with error bars for papers
4. **Validation**: Confirms vanilla SGD consistently fails on PB tasks
5. **Meta-Learning Justification**: Strong baseline for comparing against meta-learning methods

The consistent ~50% accuracy across all architectures and seeds will provide definitive evidence that vanilla SGD struggles on PB tasks, validating that the meta-learning improvements shown in your bar charts represent genuine and significant advances.
