# Three-Way Adaptation Efficiency Comparison

This experiment compares the adaptation speed of three different training methods when adapting to naturalistic data:

1. **First-Order MAML** (using pretrained meta-learning weights)
2. **Second-Order MAML** (using pretrained meta-learning weights)
3. **Vanilla SGD** (using pretrained vanilla weights)

## 🎯 Research Question

**Which method adapts fastest to new naturalistic tasks?**

- Meta-learning methods should adapt faster due to learned optimization
- Second-order MAML should outperform first-order due to more accurate gradients
- Vanilla SGD provides the baseline comparison

## 📊 What This Measures

### For Each Method:
- **Accuracy vs adaptation steps** (0-30 steps)
- **Steps needed to reach target accuracies** (55%, 60%, 65%, 70%)
- **Final adaptation accuracy** after 30 steps
- **Time per adaptation step**

### Models Used:
- **Meta-learning models**: Seeds 42-46 from `/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/`
- **Vanilla SGD models**: Seeds 47-51 from `/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular/`

### Test Data:
- **Naturalistic dataset**: `/scratch/gpfs/mg7411/samedifferent/data/naturalistic/test.h5`
- **Episodes per seed**: 100 test episodes
- **Total episodes**: ~1000 episodes across all seeds

## 🚀 How to Run on Della

### Step 1: Submit the Job
```bash
# SSH to Della and navigate to project
cd /scratch/gpfs/mg7411/samedifferent

# Submit SLURM job
sbatch run_three_way_adaptation_efficiency.slurm
```

### Step 2: Monitor Progress
```bash
# Check job status
squeue -u mg7411

# Watch output (replace JOBID with actual job ID)
tail -f results/three_way_adaptation_efficiency/slurm_output_JOBID.out
```

### Step 3: Check Results
```bash
# View results summary
tail -20 results/three_way_adaptation_efficiency/slurm_output_JOBID.out

# Check generated files
ls -la results/three_way_adaptation_efficiency/
```

## 📁 Output Files

The experiment creates:
- `three_way_adaptation_results.json` - Raw numerical results
- `three_way_adaptation_comparison.png` - Bar chart visualization
- `three_way_adaptation_comparison.pdf` - Publication-quality PDF
- `slurm_output_*.out` - Detailed execution log
- `slurm_error_*.err` - Error log (should be empty if successful)

## 📈 Expected Results

If meta-learning works as expected:

### Final Accuracy Ranking:
1. **Second-Order MAML** - Highest (best gradients)
2. **First-Order MAML** - Middle (meta-learned but approximate gradients)
3. **Vanilla SGD** - Lowest (no meta-learning)

### Adaptation Speed Ranking (Steps to 60%):
1. **Second-Order MAML** - Fewest steps
2. **First-Order MAML** - Medium steps
3. **Vanilla SGD** - Most steps (or may not reach 60%)

## ⚙️ Experiment Parameters

### Learning Rates:
- **Meta-learning inner LR**: 0.01 (for MAML adaptation)
- **Vanilla SGD LR**: 0.001 (for vanilla adaptation)

### Test Configuration:
- **Max adaptation steps**: 30
- **Episodes per seed**: 100
- **Target accuracies**: 55%, 60%, 65%, 70%
- **Expected runtime**: ~4 hours

### Model Seeds:
- **Meta-learning**: 42, 43, 44, 45, 46 (5 seeds)
- **Vanilla SGD**: 47, 48, 49, 50, 51 (5 seeds)

## 🔧 Troubleshooting

### If job fails to start:
1. Check that model files exist:
   ```bash
   ls /scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/seed_42/best_model.pt
   ls /scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular/seed_47/best_model.pt
   ```

2. Check that naturalistic data exists:
   ```bash
   ls /scratch/gpfs/mg7411/samedifferent/data/naturalistic/test.h5
   ```

### If models fail to load:
- Check the error log for specific loading issues
- Verify model architecture compatibility
- Ensure CUDA is available: `nvidia-smi`

### If results look unexpected:
- All methods should show improvement with adaptation steps
- Meta-learning methods should converge faster than vanilla
- Final accuracies should be reasonable (40-70% range)

## 🎯 Success Criteria

The experiment succeeds if it shows:
1. **Clear adaptation**: All methods improve accuracy with more steps
2. **Meta-learning advantage**: MAML methods adapt faster than vanilla
3. **Second-order superiority**: Second-order MAML outperforms first-order
4. **Statistical significance**: Error bars don't completely overlap

## 📊 Interpreting Results

### Bar Chart 1 - Final Accuracy:
- Shows final accuracy after 30 adaptation steps
- Higher is better
- Error bars show standard deviation across seeds

### Bar Chart 2 - Steps to 60%:
- Shows adaptation efficiency 
- Lower is better (fewer steps needed)
- ">" symbol means target was never reached

### Key Metrics to Report:
- **Accuracy improvement**: Final - Initial accuracy
- **Adaptation efficiency**: Steps needed for meaningful improvement
- **Method ranking**: Clear ordering of the three approaches

## 📚 Related Experiments

This experiment complements:
- **Computational efficiency comparison** (`run_computational_efficiency.slurm`)
- **Sample efficiency comparison** (various SLURM scripts)
- **PCA analysis of adaptation** (`scripts/adapt_and_visualize_pca.py`)

## 🏆 Expected Findings

This should demonstrate that:
1. **Meta-learning works**: Pretrained MAML adapts faster than vanilla SGD
2. **Second-order helps**: More accurate gradients improve adaptation speed
3. **Transfer learning**: Models trained on PB tasks can adapt to naturalistic tasks
4. **Quantitative validation**: Clear numerical evidence of meta-learning benefits 