# Sample Efficiency Comparison: FOMAML vs Second-Order MAML vs Vanilla SGD

## Overview

This experiment compares the sample efficiency of three training methods on the conv6 architecture using PB tasks:

1. **FOMAML** (First-Order MAML) - Meta-learning with first-order gradients
2. **Second-Order MAML** - Full meta-learning with second-order gradients  
3. **Vanilla SGD** - Standard supervised learning

**Goal**: Generate a single line plot with three lines showing validation accuracy vs number of data points seen.

## Architecture Details

**Conv6 Model**:
- 6 convolutional layers: 18→36→72→144→288→576 filters
- First layer: 18 filters, 6x6 kernel
- Subsequent layers: 2x2 kernels, doubling filters
- Max pooling after each conv layer
- Three fully connected layers with 1024 units each
- LayerNorm and dropout in FC layers
- Final binary classification layer

**Data**:
- 10 PB tasks: regular, lines, open, wider_line, scrambled, random_color, arrows, irregular, filled, original
- Variable support sizes (4, 6, 8, 10) for meta-learning
- H5 files located in `data/meta_h5/pb/`

## Files Overview

### Main Scripts
- `scripts/sample_efficiency_comparison.py` - Main comparison script
- `scripts/plot_sample_efficiency_results.py` - Plotting and analysis
- `scripts/sample_efficiency_comparison_simple.py` - Simplified version (vanilla SGD only)

### Testing & Verification
- `scripts/verify_sample_efficiency_setup.py` - Verify dependencies and data
- `scripts/test_sample_efficiency_quick.py` - Quick testing script
- `scripts/test_sample_efficiency_local.py` - Local testing (minimal parameters)

### Cluster Scripts
- `run_sample_efficiency_comparison.slurm` - Slurm job script for della

## Usage Instructions

### Local Testing (Optional)

1. **Verify Setup**:
   ```bash
   python scripts/verify_sample_efficiency_setup.py
   ```

2. **Quick Test**:
   ```bash
   # Test vanilla SGD only
   python scripts/test_sample_efficiency_quick.py
   
   # Test all three methods
   python scripts/test_sample_efficiency_quick.py --full
   ```

### Running on Della

1. **Copy files to della**:
   ```bash
   scp -r metasamedifferent/ mg7411@della.princeton.edu:/scratch/gpfs/mg7411/samedifferent/
   ```

2. **Submit job**:
   ```bash
   cd /scratch/gpfs/mg7411/samedifferent/metasamedifferent
   sbatch run_sample_efficiency_comparison.slurm
   ```

3. **Monitor progress**:
   ```bash
   squeue -u mg7411
   tail -f logs/sample_efficiency_*.out
   ```

### Manual Execution

For more control, run the script directly:

```bash
python scripts/sample_efficiency_comparison.py \
    --epochs 50 \
    --seed 42 \
    --meta_batch_size 16 \
    --vanilla_batch_size 64 \
    --inner_lr 0.001 \
    --outer_lr 0.0001 \
    --vanilla_lr 0.0001 \
    --adaptation_steps 5 \
    --val_frequency 25 \
    --save_dir results/sample_efficiency_comparison \
    --methods fomaml second_order vanilla \
    --data_dir data/meta_h5/pb
```

## Key Parameters

### Meta-Learning Parameters
- `--meta_batch_size`: Episodes per meta-batch (default: 16)
- `--inner_lr`: Inner loop learning rate (default: 0.001)
- `--outer_lr`: Outer loop learning rate (default: 0.0001)
- `--adaptation_steps`: Steps per adaptation (default: 5)

### Vanilla SGD Parameters
- `--vanilla_batch_size`: Batch size for vanilla SGD (default: 64)
- `--vanilla_lr`: Learning rate for vanilla SGD (default: 0.0001)

### General Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--val_frequency`: Validation frequency in batches (default: 25)
- `--seed`: Random seed for reproducibility (default: 42)

## Expected Outputs

### Files Generated
- `results/sample_efficiency_comparison/seed_42/`
  - `fomaml_results.json` - FOMAML training results
  - `second_order_maml_results.json` - Second-order MAML results
  - `vanilla_sgd_results.json` - Vanilla SGD results
  - `combined_results.json` - All results combined
  - `sample_efficiency_comparison.png` - Main comparison plot
  - `args.json` - Experiment configuration

### Plot Contents
- **X-axis**: Number of data points seen during training
- **Y-axis**: Validation accuracy (%)
- **Three lines**: One for each method (FOMAML, Second-Order MAML, Vanilla SGD)

## Troubleshooting

### Common Issues

1. **Learn2Learn Import Error**:
   ```bash
   pip install learn2learn
   ```

2. **Data Files Missing**:
   - Verify `data/meta_h5/pb/` contains all H5 files
   - Check with: `ls -la data/meta_h5/pb/ | wc -l` (should be ~240 files)

3. **CUDA Out of Memory**:
   - Reduce batch sizes: `--meta_batch_size 8 --vanilla_batch_size 32`
   - Increase validation frequency: `--val_frequency 50`

4. **Model Import Issues**:
   - Ensure project root is in Python path
   - Check imports with verification script

### Performance Tips

- **Faster Training**: Use smaller validation frequency for fewer checkpoints
- **Better Convergence**: Use learning rate scheduling (not implemented yet)
- **Memory Efficiency**: Reduce batch sizes if needed

## Results Interpretation

### Expected Behavior
- **FOMAML**: Should show faster initial learning due to meta-learning bias
- **Second-Order MAML**: May show best final performance but slower training
- **Vanilla SGD**: Should show steady improvement from random initialization

### Key Metrics
- **Sample Efficiency**: Data points needed to reach specific accuracy thresholds
- **Final Performance**: Maximum validation accuracy achieved
- **Learning Speed**: Rate of accuracy improvement per data point

## Next Steps

After successful completion:
1. Analyze results with plotting script
2. Compare to existing baselines
3. Generate publication-ready figures
4. Document findings and insights 