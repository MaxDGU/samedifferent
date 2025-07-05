#!/bin/bash
# Complete Vanilla SGD Validation with Multiple Seeds - FIXED VERSION

echo "============================================================"
echo "VANILLA SGD VALIDATION WITH MULTIPLE SEEDS - FIXED"
echo "============================================================"
echo "This script runs vanilla SGD validation across:"
echo "- 3 architectures: conv2, conv4, conv6"
echo "- 5 seeds: 42, 43, 44, 45, 46"
echo "- Total: 15 jobs (5 seeds × 3 architectures)"
echo ""
echo "ENVIRONMENT: Uses 'tensorflow' conda environment (FIXED)"
echo "MODULES: anaconda3/2022.5 + cudatoolkit/12.6 (SAME AS WORKING SCRIPTS)"
echo "============================================================"

# Check if we're on della
if [[ $HOSTNAME == *"della"* ]]; then
    echo "Running on della cluster - submitting SLURM jobs"
    
    # Submit array job
    echo "1. Submitting array job for validation..."
    JOB_ID=$(sbatch --parsable run_vanilla_sgd_validation_array_fixed.slurm)
    echo "   Array job ID: $JOB_ID"
    
    # Submit aggregation job as dependency
    echo "2. Submitting aggregation job (depends on array job completion)..."
    AGG_JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID run_aggregate_results_fixed.slurm)
    echo "   Aggregation job ID: $AGG_JOB_ID"
    
    echo ""
    echo "Jobs submitted successfully!"
    echo "Monitor progress with:"
    echo "  squeue -u $(whoami)"
    echo ""
    echo "When complete, results will be in:"
    echo "  results/vanilla_sgd_validation_array/"
    
else
    echo "Not on della cluster - showing manual commands"
    echo ""
    echo "On della, run these commands:"
    echo ""
    echo "1. Submit array job:"
    echo "   sbatch run_vanilla_sgd_validation_array_fixed.slurm"
    echo ""
    echo "2. After array job completes, submit aggregation:"
    echo "   sbatch run_aggregate_results_fixed.slurm"
    echo ""
    echo "3. Or submit with dependency (automatic):"
    echo "   JOB_ID=\$(sbatch --parsable run_vanilla_sgd_validation_array_fixed.slurm)"
    echo "   sbatch --dependency=afterok:\$JOB_ID run_aggregate_results_fixed.slurm"
    echo ""
    echo "4. Quick test first:"
    echo "   sbatch run_vanilla_sgd_validation_array_test_fixed.slurm"
    echo ""
    echo "5. Monitor jobs:"
    echo "   squeue -u \$(whoami)"
    echo ""
    echo "6. Check results:"
    echo "   ls results/vanilla_sgd_validation_array/"
fi

echo "============================================================"
