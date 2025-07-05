# Vanilla SGD Architecture Validation

## Overview
This validation experiment tests vanilla SGD performance across three CNN architectures (conv2, conv4, conv6) to reproduce and validate the bar chart results from previous experiments.

## Files Created

### Main Script
- `scripts/validate_vanilla_sgd_architectures.py` - Main validation script
- `run_vanilla_sgd_validation.slurm` - SLURM submission script  
- `scripts/test_vanilla_sgd_validation.py` - Local testing script

### Code Reuse
The validation script reuses existing baselines code:
- **Architecture definitions**: `baselines/models/{conv2,conv4,conv6}.py`
- **Data loading**: `data/vanilla_h5_dataset_creation.py`
- **Training utilities**: `baselines/models/utils.py`

## Architecture Details

| Architecture | Parameters | Description |
|-------------|------------|-------------|
| Conv2CNN | 15,484,320 | 2-layer CNN: 6, 12 filters with 2x2 kernels |
| Conv4CNN | 10,090,206 | 4-layer CNN: 12, 24, 48, 96 filters |
| Conv6CNN | 8,299,908 | 6-layer CNN: 18, 36, 72, 144, 288, 576 filters |

## Running the Experiment

### Local Testing
```bash
python scripts/test_vanilla_sgd_validation.py
```

### Cluster Execution
```bash
sbatch run_vanilla_sgd_validation.slurm
```

### Manual Execution
```bash
python scripts/validate_vanilla_sgd_architectures.py \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/vanilla_h5 \
    --save_dir /scratch/gpfs/mg7411/samedifferent/results/vanilla_sgd_validation \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --patience 10 \
    --seed 42 \
    --architectures conv2 conv4 conv6
```

## Expected Results

The validation should reproduce the vanilla SGD results from your bar charts:
- **Conv2**: Expected ~80-85% accuracy
- **Conv4**: Expected ~85-90% accuracy  
- **Conv6**: Expected ~90-95% accuracy

## Output Files

Results will be saved to:
- `results/vanilla_sgd_validation/seed_42/validation_results.json`
- `results/vanilla_sgd_validation/seed_42/vanilla_sgd_architecture_comparison.png`
- `results/vanilla_sgd_validation/seed_42/vanilla_sgd_architecture_comparison.pdf`

## Purpose

This validation serves as a sanity check to ensure:
1. Our vanilla SGD implementation is working correctly
2. The architecture definitions match expectations
3. Results are consistent with previous bar chart findings
4. The experimental setup is sound for further comparisons

The results should match the vanilla SGD (left side) bars in your previous charts, confirming that our implementation is correct and ready for more complex experiments.
