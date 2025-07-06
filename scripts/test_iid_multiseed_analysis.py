#!/usr/bin/env python3
"""
Test script for IID multi-seed statistical analysis.

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

def create_simulated_iid_results():
    """Create simulated IID results for testing."""
    print("🧪 Creating simulated IID results for testing...")
    
    # Simulation parameters
    seeds = [42, 43, 44, 45, 46]
    methods = ['fomaml', 'second_order', 'vanilla']
    
    # Create realistic learning curves based on the provided plot
    n_points = 30
    data_points = np.linspace(200000, 5000000, n_points)  # 0.2M to 5M data points
    
    # Method-specific performance characteristics (based on your plot)
    method_params = {
        'fomaml': {'initial': 50, 'final': 75, 'noise': 2.5, 'learning_rate': 0.6},
        'second_order': {'initial': 50, 'final': 81, 'noise': 2.0, 'learning_rate': 0.7},
        'vanilla': {'initial': 50, 'final': 92, 'noise': 3.5, 'learning_rate': 1.2}  # Much better in IID
    }
    
    # Create results for each seed
    for seed in seeds:
        np.random.seed(seed)
        
        # Create seed directory
        seed_dir = f'results/sample_efficiency_iid_multiseed/seed_{seed}'
        os.makedirs(seed_dir, exist_ok=True)
        
        for method in methods:
            params = method_params[method]
            
            # Generate learning curve
            progress = np.linspace(0, 1, n_points)
            base_curve = params['initial'] + (params['final'] - params['initial']) * (progress ** params['learning_rate'])
            
            # Add realistic noise
            noise = np.random.normal(0, params['noise'], n_points)
            accuracies = base_curve + noise
            
            # For vanilla SGD, add some plateaus like in the real plot
            if method == 'vanilla':
                # Add some flat regions early on
                for i in range(5, 10):
                    accuracies[i] = accuracies[4] + np.random.normal(0, 1)
                
                # Add characteristic jumps
                for i in range(10, 15):
                    accuracies[i] = accuracies[9] + (i-9) * 5 + np.random.normal(0, 2)
            
            # Ensure monotonic improvement (mostly)
            for i in range(1, len(accuracies)):
                if accuracies[i] < accuracies[i-1] - 2.0:  # Allow some fluctuation
                    accuracies[i] = accuracies[i-1] + np.random.normal(0.5, 0.5)
            
            # Clip to reasonable range
            accuracies = np.clip(accuracies, 48, 95)
            
            # Create result structure (matching the IID script format)
            result = {
                'method': method.upper() if method != 'second_order' else 'Second-Order MAML',
                'data_points_seen': data_points.tolist(),
                'val_accuracies': accuracies.tolist(),
                'total_data_points': int(data_points[-1]),
                'config': {
                    'seed': seed,
                    'simulated': True,
                    'method_params': params
                }
            }
            
            # Save result file (matching the IID naming convention)
            if method == 'fomaml':
                filename = 'fomaml_results.json'
            elif method == 'second_order':
                filename = 'second_order_maml_results.json'
            elif method == 'vanilla':
                filename = 'vanilla_sgd_results.json'
            
            filepath = os.path.join(seed_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        
        print(f"✅ Created simulated IID results for seed {seed}")
    
    print(f"✅ All simulated IID results created in results/sample_efficiency_iid_multiseed/")
    return True

def test_iid_statistical_analysis():
    """Test the IID statistical analysis functionality."""
    print("\n📊 Testing IID statistical analysis...")
    
    # Import the multi-seed analysis
    sys.path.append('scripts')
    from sample_efficiency_iid_multiseed import MultiSeedIIDAnalysis
    
    # Create mock args
    class MockArgs:
        def __init__(self):
            self.save_dir = 'results/sample_efficiency_iid_multiseed'
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
    
    analyzer = MultiSeedIIDAnalysis(args, seeds, methods)
    
    # Load results
    if not analyzer.load_results():
        print("❌ Failed to load simulated IID results")
        return False
    
    # Perform analysis
    analyzer.interpolate_curves()
    analyzer.perform_statistical_analysis()
    
    # Create output directory
    output_dir = 'results/sample_efficiency_iid_multiseed/multiseed_analysis_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    analyzer.create_publication_plots(output_dir)
    
    # Save reports
    analyzer.save_statistical_report(output_dir)
    
    print(f"✅ IID statistical analysis completed successfully!")
    print(f"✅ Results saved to: {output_dir}")
    
    # Print key findings
    if analyzer.statistical_results:
        print("\n📈 KEY IID FINDINGS:")
        print("-" * 40)
        
        final_stats = analyzer.statistical_results['overall_stats']
        for method in methods:
            if method in final_stats:
                final_perf = final_stats[method]['final_performance']
                print(f"{method:12}: {final_perf['mean']:5.1f}% ± {final_perf['sem']:4.1f}%")
        
        # Check for significant differences
        final_point_key = list(analyzer.statistical_results['key_points'].keys())[-1]
        final_comparisons = analyzer.statistical_results['key_points'][final_point_key]['comparisons']
        
        print("\n🔍 SIGNIFICANT DIFFERENCES (IID):")
        print("-" * 40)
        for comp_name, comp_data in final_comparisons.items():
            if comp_data['significant']:
                print(f"{comp_name}: p={comp_data['p_value']:.3f}, d={comp_data['cohens_d']:.2f} ({comp_data['effect_size']})")
            else:
                print(f"{comp_name}: Not significant (p={comp_data['p_value']:.3f})")
    
    return True

def main():
    print("🧪 MULTI-SEED IID STATISTICAL ANALYSIS TEST")
    print("=" * 50)
    
    # Step 1: Create simulated results
    if not create_simulated_iid_results():
        print("❌ Failed to create simulated IID results")
        return
    
    # Step 2: Test statistical analysis
    if not test_iid_statistical_analysis():
        print("❌ Failed to test IID statistical analysis")
        return
    
    print("\n🎉 ALL IID TESTS PASSED!")
    print("=" * 50)
    print("The IID multi-seed statistical analysis is working correctly.")
    print("Expected pattern: Vanilla SGD should significantly outperform")
    print("meta-learning methods in the IID scenario.")
    print("")
    print("You can now run the real IID experiments with:")
    print("  sbatch run_sample_efficiency_iid_multiseed.slurm")
    print("=" * 50)

if __name__ == '__main__':
    main() 