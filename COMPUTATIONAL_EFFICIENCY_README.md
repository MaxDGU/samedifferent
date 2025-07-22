# Computational Efficiency Comparison Experiment

This experiment tests the hypothesis that **First-Order MAML** is more training-time efficient but less test-time efficient than **Second-Order MAML**.

## 🎯 Hypothesis
- **FOMAML**: Faster training (no Hessian computation), slower test-time adaptation
- **Second-Order MAML**: Slower training (Hessian computation), faster test-time adaptation

## 📊 What This Measures

### Training Efficiency
- Time per training batch
- Peak memory usage  
- Forward vs backward pass time breakdown
- Optimization step time

### Test-Time Adaptation Efficiency  
- Accuracy vs number of adaptation steps
- Steps needed to reach target accuracies (60%, 65%, 70%)
- Time per adaptation step

## 🚀 How to Run on Della

### Step 1: Trial Run (On Della Only)
Test the script with minimal parameters **on Della** (local environment may have incomplete dependencies):

```bash
# SSH to Della and navigate to project
cd /scratch/gpfs/mg7411/samedifferent

# Activate the environment
conda activate tensorflow

# Run conservative trial (recommended - includes optimization fixes)
python scripts/computational_efficiency_trial_conservative.py

# Or run original trial
python scripts/computational_efficiency_trial.py
```

**Conservative Trial Features**:
- Conservative learning rates (inner_lr=0.001, outer_lr=0.0001)  
- Gradient clipping for stability
- Weight decay and learning rate scheduler
- Optimized for Second-Order MAML convergence

**Note**: The trial requires the full learn2learn installation which may not be available locally.

### Step 2: Full Experiment
If trial run succeeds, run the full experiment:

```bash
# Submit SLURM job
sbatch run_computational_efficiency.slurm
```

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u mg7411

# Watch output (replace JOBID with actual job ID)
tail -f results/computational_efficiency/slurm_output_JOBID.out
```

## 📁 Output Files

The experiment creates:
- `results/computational_efficiency/computational_efficiency_results.json` - Raw data
- `results/computational_efficiency/efficiency_summary.txt` - Human-readable summary
- `results/computational_efficiency/training_efficiency_comparison.png` - Training plots
- `results/computational_efficiency/adaptation_efficiency_comparison.png` - Adaptation plots

## ⚙️ Experiment Parameters

**Minimal Dataset** (for speed):
- Tasks: `['regular', 'lines', 'open']` (3 tasks only)
- Training batches measured: 300 (with 800 warmup)
- Test episodes: 150
- Max adaptation steps: 20

**Conservative Learning Rates** (for stability):
- Inner LR: 0.001 (was 0.1)
- Outer LR: 0.0001 (was 0.005)
- Gradient clipping: max_norm=1.0
- Weight decay: 1e-5

**Expected Runtime**: ~4-6 hours on GPU (increased for better learning)

## 📈 Expected Results

If hypothesis is correct:

**Training Efficiency**:
- FOMAML: ~2-3x faster training, ~50% less memory
- Second-Order: Slower due to Hessian computation

**Test-Time Efficiency**:
- FOMAML: Needs 8-12 steps to reach 70% accuracy
- Second-Order: Needs 3-5 steps to reach 70% accuracy

## 🔧 Troubleshooting

**If trial run fails**:
1. Check that you're in the correct environment: `conda activate tensorflow`
2. Verify data path exists: `ls /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb`
3. Check GPU availability: `nvidia-smi`

**If SLURM job fails**:
1. Check error log: `cat results/computational_efficiency/slurm_error_JOBID.err`
2. Verify modules loaded correctly in the output log
3. Check memory usage - may need to reduce batch sizes

**If results look wrong**:
- Both methods should learn (accuracy should increase with adaptation steps)
- Second-Order MAML should show slower training but faster adaptation
- Memory usage should be higher for Second-Order MAML

## 🎯 Success Criteria

The experiment succeeds if it shows:
1. **Training time trade-off**: FOMAML faster training, Second-Order slower
2. **Adaptation efficiency trade-off**: Second-Order reaches target accuracy in fewer steps
3. **Clear computational differences**: Memory and timing measurements show expected patterns

This will provide empirical evidence for the theoretical training vs test-time efficiency trade-off in meta-learning algorithms. 