#!/bin/bash
# Quick test of the fixed validation script

echo "Testing FIXED vanilla SGD validation script..."
echo "This test should be run on della cluster"
echo ""

# Test command for della
echo "Test command for della:"
echo "python scripts/validate_vanilla_sgd_architectures_fixed.py \\"
echo "    --data_dir data/meta_h5/pb \\"
echo "    --save_dir results/vanilla_sgd_validation_test \\"
echo "    --epochs 2 \\"
echo "    --batch_size 16 \\"
echo "    --lr 1e-4 \\"
echo "    --patience 5 \\"
echo "    --seed 42 \\"
echo "    --architectures conv6"
echo ""

# Full job submission
echo "Full job submission:"
echo "sbatch run_vanilla_sgd_validation_fixed.slurm"
