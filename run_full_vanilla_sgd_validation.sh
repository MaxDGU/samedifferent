#!/bin/bash
# Complete Vanilla SGD Validation with Multiple Seeds

echo "============================================================"
echo "VANILLA SGD VALIDATION WITH MULTIPLE SEEDS"
echo "============================================================"
echo "This script runs vanilla SGD validation across:"
echo "- 3 architectures: conv2, conv4, conv6"
echo "- 5 seeds: 42, 43, 44, 45, 46"
echo "- Total: 15 jobs (5 seeds × 3 architectures)"
echo "============================================================"

# Check if we're on della
if [[ $HOSTNAME == *"della"* ]]; then
    echo "Running on della cluster - submitting SLURM jobs"
    
    # Submit array job
    echo "1. Submitting array job for validation..."
    JOB_ID=$(sbatch --parsable run_vanilla_sgd_validation_array.slurm)
    echo "   Array job ID: $JOB_ID"
    
    # Submit aggregation job as dependency
    echo "2. Submitting aggregation job (depends on array job completion)..."
    AGG_JOB_ID=$(sbatch --parsable --dependency=afterok:$JOB_ID run_aggregate_results.slurm)
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
    echo "   sbatch run_vanilla_sgd_validation_array.slurm"
    echo ""
    echo "2. After array job completes, submit aggregation:"
    echo "   sbatch run_aggregate_results.slurm"
    echo ""
    echo "3. Or submit with dependency (automatic):"
    echo "   JOB_ID=\$(sbatch --parsable run_vanilla_sgd_validation_array.slurm)"
    echo "   sbatch --dependency=afterok:\$JOB_ID run_aggregate_results.slurm"
    echo ""
    echo "4. Monitor jobs:"
    echo "   squeue -u \$(whoami)"
    echo ""
    echo "5. Check results:"
    echo "   ls results/vanilla_sgd_validation_array/"
fi

echo "============================================================"
