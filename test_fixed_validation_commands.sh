#!/bin/bash
# Quick test commands for the FIXED validation scripts

echo "=========================================================="
echo "FIXED VANILLA SGD VALIDATION - TEST COMMANDS"
echo "=========================================================="
echo "Environment: 'tensorflow' conda environment"
echo "Modules: anaconda3/2022.5 + cudatoolkit/12.6"
echo "=========================================================="
echo ""

echo "QUICK TEST (2 seeds, 5 epochs, 6 jobs):"
echo "sbatch run_vanilla_sgd_validation_array_test_fixed.slurm"
echo ""

echo "FULL EXPERIMENT (5 seeds, 100 epochs, 15 jobs):"
echo "sbatch run_vanilla_sgd_validation_array_fixed.slurm"
echo ""

echo "SINGLE JOB (all 3 architectures in one job):"
echo "sbatch run_vanilla_sgd_validation_single_fixed.slurm"
echo ""

echo "COMPLETE AUTOMATED PIPELINE:"
echo "./run_full_vanilla_sgd_validation_fixed.sh"
echo ""

echo "MANUAL WITH DEPENDENCIES:"
echo "JOB_ID=\$(sbatch --parsable run_vanilla_sgd_validation_array_fixed.slurm)"
echo "sbatch --dependency=afterok:\$JOB_ID run_aggregate_results_fixed.slurm"
echo ""

echo "MONITOR JOBS:"
echo "squeue -u \$(whoami)"
echo ""

echo "CHECK RESULTS:"
echo "ls results/vanilla_sgd_validation_array/"
echo "python scripts/aggregate_vanilla_sgd_results.py --results_dir results/vanilla_sgd_validation_array"
echo ""

echo "=========================================================="
echo "All scripts now use the CORRECT 'tensorflow' environment!"
echo "=========================================================="
