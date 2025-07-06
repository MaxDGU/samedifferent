#!/usr/bin/env python3
"""
Test script for multi-seed OOD statistical analysis.

This script creates simulated results to test the statistical analysis
functionality without running full experiments.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_simulated_results():
    """Create simulated results for testing."""
    print("🧪 Creating simulated results for testing...")
    
    # Simulation parameters
    seeds = [42, 43, 44, 45, 46]
    methods = ['fomaml', 'second_order', 'vanilla']
    holdout_task = 'scrambled'
    
    # Create realistic learning curves
    n_points = 30
    data_points = np.linspace(500000, 5000000, n_points)  # 0.5M to 5M data points
    
    # Method-specific performance characteristics
    method_params = {
        'fomaml': {'initial': 52, 'final': 74, 'noise': 2.0, 'learning_rate': 0.8},
        'second_order': {'initial': 50, 'final': 62, 'noise': 1.5, 'learning_rate': 0.6},
        'vanilla': {'initial': 50, 'final': 65, 'noise': 3.0, 'learning_rate': 0.4}
    }
    
    # Create results for each seed
    for seed in seeds:
        np.random.seed(seed)
        
        # Create seed directory
        seed_dir = f'results/sample_efficiency_ood_multiseed/holdout_{holdout_task}/seed_{seed}'
        os.makedirs(seed_dir, exist_ok=True)
        
        for method in methods:
            params = method_params[method]
            
            # Generate learning curve
            progress = np.linspace(0, 1, n_points)
            base_curve = params['initial'] + (params['final'] - params['initial']) * (progress ** params['learning_rate'])
            
            # Add realistic noise
            noise = np.random.normal(0, params['noise'], n_points)
            accuracies = base_curve + noise
            
            # Ensure monotonic improvement (mostly)
            for i in range(1, len(accuracies)):
                if accuracies[i] < accuracies[i-1] - 1.0:  # Allow some fluctuation
                    accuracies[i] = accuracies[i-1] + np.random.normal(0, 0.5)
            
            # Clip to reasonable range
            accuracies = np.clip(accuracies, 48, 80)
            
            # Create result structure
            result = {
                'method': method.upper() if method != 'second_order' else 'Second-Order MAML',
                'holdout_task': holdout_task,
                'train_tasks': ['regular', 'lines', 'open', 'wider_line', 'random_color', 
                               'arrows', 'irregular', 'filled', 'original'],
                'data_points_seen': data_points.tolist(),
                'val_accuracies': accuracies.tolist(),
                'total_data_points': int(data_points[-1]),
                'ood_evaluation': True,
                'config': {
                    'seed': seed,
                    'simulated': True,
                    'method_params': params
                }
            }
            
            # Save result file
            if method == 'fomaml':
                filename = f'fomaml_holdout_{holdout_task}_results.json'
            elif method == 'second_order':
                filename = f'second_order_maml_holdout_{holdout_task}_results.json'
            elif method == 'vanilla':
                filename = f'vanilla_sgd_holdout_{holdout_task}_results.json'
            
            filepath = os.path.join(seed_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        
        print(f"✅ Created simulated results for seed {seed}")
    
    print(f"✅ All simulated results created in results/sample_efficiency_ood_multiseed/")
    return True

def test_statistical_analysis():
    """Test the statistical analysis functionality."""
    print("\n📊 Testing statistical analysis...")
    
    # Import the multi-seed analysis
    sys.path.append('scripts')
    from sample_efficiency_holdout_multiseed import MultiSeedOODAnalysis
    
    # Create mock args
    class MockArgs:
        def __init__(self):
            self.holdout_task = 'scrambled'
            self.save_dir = 'results/sample_efficiency_ood_multiseed'
            self.epochs = 15
            self.meta_batch_size = 16
            self.vanilla_batch_size = 64
            self.inner_lr = 0.05
            self.outer_lr = 0.001
            self.vanilla_lr = 0.0001
            self.adaptation_steps = 3
            self.val_frequency = 3000
            self.data_dir = 'data/meta_h5/pb'
    
    # Test the analysis
    args = MockArgs()
    seeds = [42, 43, 44, 45, 46]
    methods = ['fomaml', 'second_order', 'vanilla']
    
    analyzer = MultiSeedOODAnalysis(args, seeds, methods)
    
    # Load results
    if not analyzer.load_results():
        print("❌ Failed to load simulated results")
        return False
    
    # Perform analysis
    analyzer.interpolate_curves()
    analyzer.perform_statistical_analysis()
    
    # Create output directory
    output_dir = 'results/sample_efficiency_ood_multiseed/holdout_scrambled/multiseed_analysis_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    analyzer.create_publication_plots(output_dir)
    
    # Save reports
    analyzer.save_statistical_report(output_dir)
    
    print(f"✅ Statistical analysis completed successfully!")
    print(f"✅ Results saved to: {output_dir}")
    
    # Print key findings
    if analyzer.statistical_results:
        print("\n📈 KEY FINDINGS:")
        print("-" * 40)
        
        final_stats = analyzer.statistical_results['overall_stats']
        for method in methods:
            if method in final_stats:
                final_perf = final_stats[method]['final_performance']
                print(f"{method:12}: {final_perf['mean']:5.1f}% ± {final_perf['sem']:4.1f}%")
        
        # Check for significant differences
        final_point_key = list(analyzer.statistical_results['key_points'].keys())[-1]
        final_comparisons = analyzer.statistical_results['key_points'][final_point_key]['comparisons']
        
        print("\n🔍 SIGNIFICANT DIFFERENCES:")
        print("-" * 40)
        for comp_name, comp_data in final_comparisons.items():
            if comp_data['significant']:
                print(f"{comp_name}: p={comp_data['p_value']:.3f}, d={comp_data['cohens_d']:.2f} ({comp_data['effect_size']})")
            else:
                print(f"{comp_name}: Not significant (p={comp_data['p_value']:.3f})")
    
    return True

def main():
    print("🧪 MULTI-SEED OOD STATISTICAL ANALYSIS TEST")
    print("=" * 50)
    
    # Step 1: Create simulated results
    if not create_simulated_results():
        print("❌ Failed to create simulated results")
        return
    
    # Step 2: Test statistical analysis
    if not test_statistical_analysis():
        print("❌ Failed to test statistical analysis")
        return
    
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("The multi-seed statistical analysis is working correctly.")
    print("You can now run the real experiments with:")
    print("  sbatch run_sample_efficiency_holdout_multiseed.slurm")
    print("=" * 50)

if __name__ == '__main__':
    main() 