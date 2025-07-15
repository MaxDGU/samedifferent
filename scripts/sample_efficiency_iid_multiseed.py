#!/usr/bin/env python3
"""
Multi-Seed IID Sample Efficiency Comparison with Statistical Analysis

This script runs the in-distribution sample efficiency experiment across multiple seeds 
and performs comprehensive statistical analysis with publication-quality visualizations.

It now integrates SLURM for robust execution and includes checkpointing.
"""

import os
import sys
import argparse
import json
import subprocess
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class MultiSeedIIDAnalysis:
    """Handle multi-seed IID experiment execution and statistical analysis."""
    
    def __init__(self, base_args, seeds, methods):
        self.base_args = base_args
        self.seeds = seeds
        self.methods = methods
        self.results_data = {}
        self.statistical_results = {}
        os.makedirs(self.base_args.save_dir, exist_ok=True)
        
    def generate_slurm_script(self):
        """Generate a SLURM array job script to run all seed experiments."""
        script_path = os.path.join(self.base_args.save_dir, "run_iid_multiseed.slurm")
        
        # Use the main comparison script that supports all three methods
        experiment_script = "scripts/sample_efficiency_comparison.py"

        slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name=iid_sample_efficiency
#SBATCH --output={self.base_args.save_dir}/slurm_out/slurm_%A_%a.out
#SBATCH --error={self.base_args.save_dir}/slurm_out/slurm_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-{len(self.seeds)-1}

# Load required modules
module load anaconda3
module load cuda/11.7.0

# Activate Conda environment
conda activate tensorflow

# Get seed for this job
SEEDS=({" ".join(map(str, self.seeds))})
SEED=${{SEEDS[$SLURM_ARRAY_TASK_ID]}}

echo "Running IID Sample Efficiency for seed $SEED"

python {experiment_script} \\
    --epochs {self.base_args.epochs} \\
    --seed $SEED \\
    --meta_batch_size {self.base_args.meta_batch_size} \\
    --vanilla_batch_size {self.base_args.vanilla_batch_size} \\
    --inner_lr {self.base_args.inner_lr} \\
    --outer_lr {self.base_args.outer_lr} \\
    --vanilla_lr {self.base_args.vanilla_lr} \\
    --adaptation_steps {self.base_args.adaptation_steps} \\
    --val_frequency {self.base_args.val_frequency} \\
    --save_dir {self.base_args.save_dir} \\
    --methods {" ".join(self.methods)} \\
    --data_dir {self.base_args.data_dir}

echo "Job for seed $SEED finished."
"""
        os.makedirs(os.path.join(self.base_args.save_dir, "slurm_out"), exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(slurm_script_content)
            
        print("\n✅ Generated SLURM script for experiment execution.")
        print(f"   Script saved to: {script_path}")
        print("\n👉 To run the experiments, submit the job to SLURM with:")
        print(f"   sbatch {script_path}")
        print("\nAfter the jobs complete, run this script again with the --analyze flag.")
        
    def load_results(self):
        """Load results from all seeds."""
        print("\n📊 Loading results from all seeds...")
        
        for seed in self.seeds:
            seed_dir = os.path.join(self.base_args.save_dir, f'seed_{seed}')
            
            if not os.path.exists(seed_dir):
                print(f"⚠️  Warning: Results directory not found for seed {seed}")
                continue
                
            seed_data = {}
            
            # Load each method's results
            for method in self.methods:
                if method == 'fomaml':
                    filename = 'fomaml_results.json'
                elif method == 'second_order':
                    filename = 'second_order_maml_results.json'
                elif method == 'vanilla':
                    filename = 'vanilla_sgd_results.json'
                else:
                    continue
                    
                filepath = os.path.join(seed_dir, filename)
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        method_data = json.load(f)
                        seed_data[method] = method_data
                else:
                    print(f"⚠️  Warning: Results file not found: {filepath}")
            
            if seed_data:
                self.results_data[seed] = seed_data
        
        print(f"✅ Loaded results for {len(self.results_data)} seeds")
        return len(self.results_data) > 0
    
    def interpolate_curves(self, target_points=50):
        """Interpolate all curves to common data points for statistical analysis."""
        print("\n🔄 Interpolating curves for statistical analysis...")
        
        # Find common data point range
        all_data_points = []
        for seed_data in self.results_data.values():
            for method_data in seed_data.values():
                all_data_points.extend(method_data['data_points_seen'])
        
        min_points = min(all_data_points)
        max_points = max(all_data_points)
        
        # Create common x-axis
        common_x = np.linspace(min_points, max_points, target_points)
        
        # Interpolate each curve
        interpolated_data = {}
        for seed in self.results_data:
            interpolated_data[seed] = {}
            for method in self.results_data[seed]:
                data_points = np.array(self.results_data[seed][method]['data_points_seen'])
                accuracies = np.array(self.results_data[seed][method]['val_accuracies'])
                
                # Interpolate to common x-axis
                interp_func = interp1d(data_points, accuracies, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
                interpolated_y = interp_func(common_x)
                
                interpolated_data[seed][method] = {
                    'data_points': common_x,
                    'accuracies': interpolated_y
                }
        
        self.interpolated_data = interpolated_data
        self.common_x = common_x
        print(f"✅ Interpolated curves to {target_points} common points")
    
    def perform_statistical_analysis(self):
        """Perform comprehensive statistical analysis."""
        print("\n📈 Performing statistical analysis...")
        
        # Key sample size points for analysis
        key_points_indices = [
            int(0.2 * len(self.common_x)),  # Early (20%)
            int(0.4 * len(self.common_x)),  # Mid-early (40%)
            int(0.6 * len(self.common_x)),  # Mid (60%)
            int(0.8 * len(self.common_x)),  # Late (80%)
            int(0.95 * len(self.common_x))  # Final (95%)
        ]
        
        key_points = self.common_x[key_points_indices]
        
        # Organize data for analysis
        analysis_data = {}
        for method in self.methods:
            method_curves = []
            for seed in self.results_data:
                if method in self.interpolated_data[seed]:
                    method_curves.append(self.interpolated_data[seed][method]['accuracies'])
            
            if method_curves:
                analysis_data[method] = np.array(method_curves)
        
        # Statistical tests at key points
        statistical_results = {}
        for i, point_idx in enumerate(key_points_indices):
            point_data = {}
            for method in analysis_data:
                point_data[method] = analysis_data[method][:, point_idx]
            
            # Pairwise comparisons
            comparisons = {}
            method_names = list(point_data.keys())
            
            for j in range(len(method_names)):
                for k in range(j+1, len(method_names)):
                    method1, method2 = method_names[j], method_names[k]
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(point_data[method1], point_data[method2])
                    
                    # Effect size (Cohen's d)
                    diff = point_data[method1] - point_data[method2]
                    pooled_std = np.sqrt((np.var(point_data[method1]) + np.var(point_data[method2])) / 2)
                    cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
                    
                    comparisons[f"{method1}_vs_{method2}"] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'mean_diff': float(np.mean(diff)),
                        'significant': bool(p_value < 0.05),
                        'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
                    }
            
            statistical_results[f"point_{i+1}"] = {
                'data_points': float(key_points[i]),
                'comparisons': comparisons,
                'means': {method: float(np.mean(point_data[method])) for method in point_data},
                'stds': {method: float(np.std(point_data[method])) for method in point_data},
                'sems': {method: float(stats.sem(point_data[method])) for method in point_data}
            }
        
        # Overall curve analysis
        overall_stats = {}
        for method in analysis_data:
            curves = analysis_data[method]
            overall_stats[method] = {
                'mean_curve': np.mean(curves, axis=0),
                'std_curve': np.std(curves, axis=0),
                'sem_curve': stats.sem(curves, axis=0),
                'final_performance': {
                    'mean': float(np.mean(curves[:, -1])),
                    'std': float(np.std(curves[:, -1])),
                    'sem': float(stats.sem(curves[:, -1]))
                }
            }
        
        self.statistical_results = {
            'key_points': statistical_results,
            'overall_stats': overall_stats,
            'key_sample_sizes': key_points,
            'n_seeds': len(self.results_data)
        }
        
        print(f"✅ Statistical analysis completed for {len(self.methods)} methods")
    
    def create_publication_plots(self, save_dir):
        """Create publication-quality plots with statistical analysis."""
        print("\n🎨 Creating publication-quality plots...")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        colors = {'fomaml': '#2E86AB', 'second_order': '#A23B72', 'vanilla': '#F18F01'}
        method_labels = {
            'fomaml': 'FOMAML',
            'second_order': 'Second-Order MAML', 
            'vanilla': 'Vanilla SGD'
        }
        
        # Main comparison plot with error bands
        fig, ax = plt.subplots(figsize=(14, 10))
        
        for method in self.methods:
            if method in self.statistical_results['overall_stats']:
                stats_data = self.statistical_results['overall_stats'][method]
                
                # Plot mean curve
                ax.plot(self.common_x / 1e6, stats_data['mean_curve'], 
                       color=colors.get(method, 'black'), linewidth=3, 
                       label=method_labels.get(method, method))
                
                # Plot error bands (±1 SEM)
                ax.fill_between(self.common_x / 1e6, 
                               stats_data['mean_curve'] - stats_data['sem_curve'],
                               stats_data['mean_curve'] + stats_data['sem_curve'],
                               color=colors.get(method, 'black'), alpha=0.2)
                
                # Plot individual seed trajectories (light)
                for seed in self.results_data:
                    if method in self.interpolated_data[seed]:
                        ax.plot(self.common_x / 1e6, 
                               self.interpolated_data[seed][method]['accuracies'],
                               color=colors.get(method, 'black'), alpha=0.3, linewidth=1)
        
        # Add significance markers
        for i, point_key in enumerate(self.statistical_results['key_points']):
            point_data = self.statistical_results['key_points'][point_key]
            x_pos = point_data['data_points'] / 1e6
            
            # Find significant comparisons
            significant_comparisons = []
            for comp_name, comp_data in point_data['comparisons'].items():
                if comp_data['significant']:
                    significant_comparisons.append(comp_name)
            
            if significant_comparisons:
                # Add significance marker
                max_y = max([point_data['means'][method] for method in point_data['means']])
                ax.annotate('*', xy=(x_pos, max_y + 2), fontsize=16, 
                           ha='center', va='bottom', color='red', weight='bold')
        
        # Formatting
        ax.set_xlabel('Number of Training Samples Seen (millions)', fontsize=14)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
        ax.set_title(f'IID Sample Efficiency Comparison: MAML vs Vanilla SGD\n'
                    f'Multi-Seed Analysis (n={self.statistical_results["n_seeds"]} seeds) with Statistical Significance', 
                    fontsize=16, weight='bold')
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(40, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'iid_multiseed_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'iid_multiseed_comparison.pdf'), bbox_inches='tight')
        
        # Statistical summary plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plots at key points
        key_point_data = []
        for i, point_key in enumerate(self.statistical_results['key_points']):
            point_data = self.statistical_results['key_points'][point_key]
            for method in self.methods:
                if method in point_data['means']:
                    key_point_data.append({
                        'Sample Size (M)': f"{point_data['data_points']/1e6:.1f}M",
                        'Method': method_labels.get(method, method),
                        'Accuracy': point_data['means'][method]
                    })
        
        if key_point_data:
            df = pd.DataFrame(key_point_data)
            sns.boxplot(data=df, x='Sample Size (M)', y='Accuracy', hue='Method', ax=axes[0])
            axes[0].set_title('Performance at Key Sample Sizes', fontsize=14, weight='bold')
            axes[0].set_ylabel('Accuracy (%)', fontsize=12)
            axes[0].set_xlabel('Sample Size', fontsize=12)
        
        # Final performance comparison
        final_data = []
        for method in self.methods:
            if method in self.statistical_results['overall_stats']:
                final_perf = self.statistical_results['overall_stats'][method]['final_performance']
                final_data.append({
                    'Method': method_labels.get(method, method),
                    'Mean': final_perf['mean'],
                    'SEM': final_perf['sem']
                })
        
        if final_data:
            df_final = pd.DataFrame(final_data)
            bars = axes[1].bar(df_final['Method'], df_final['Mean'], 
                              yerr=df_final['SEM'], capsize=5, 
                              color=[colors.get(method, 'black') for method in self.methods])
            axes[1].set_title('Final Performance Comparison', fontsize=14, weight='bold')
            axes[1].set_ylabel('Final Accuracy (%)', fontsize=12)
            axes[1].set_xlabel('Method', fontsize=12)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, df_final['Mean']):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{mean_val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'iid_statistical_summary.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'iid_statistical_summary.pdf'), bbox_inches='tight')
        
        print(f"✅ Publication plots saved to {save_dir}")
        return fig, axes
    
    def save_statistical_report(self, save_dir):
        """Save detailed statistical analysis report."""
        print("\n📝 Saving statistical analysis report...")
        
        # Create JSON-serializable copy of statistical results
        serializable_results = {
            'key_points': self.statistical_results['key_points'],
            'overall_stats': {},
            'key_sample_sizes': self.statistical_results['key_sample_sizes'].tolist(),
            'n_seeds': self.statistical_results['n_seeds']
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for method in self.statistical_results['overall_stats']:
            method_stats = self.statistical_results['overall_stats'][method]
            serializable_results['overall_stats'][method] = {
                'mean_curve': method_stats['mean_curve'].tolist(),
                'std_curve': method_stats['std_curve'].tolist(),
                'sem_curve': method_stats['sem_curve'].tolist(),
                'final_performance': method_stats['final_performance']
            }
        
        # Create comprehensive report
        report = {
            'experiment_info': {
                'experiment_type': 'iid_sample_efficiency',
                'seeds': self.seeds,
                'methods': self.methods,
                'n_seeds': len(self.results_data),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'statistical_results': serializable_results,
            'summary': {
                'significant_differences': {},
                'effect_sizes': {},
                'recommendations': []
            }
        }
        
        # Generate summary insights
        for point_key in self.statistical_results['key_points']:
            point_data = self.statistical_results['key_points'][point_key]
            sample_size = point_data['data_points']
            
            significant_diffs = []
            large_effects = []
            
            for comp_name, comp_data in point_data['comparisons'].items():
                if comp_data['significant']:
                    significant_diffs.append(f"{comp_name}: p={comp_data['p_value']:.3f}")
                
                if abs(comp_data['cohens_d']) > 0.8:
                    large_effects.append(f"{comp_name}: d={comp_data['cohens_d']:.2f}")
            
            report['summary']['significant_differences'][f'at_{sample_size/1e6:.1f}M'] = significant_diffs
            report['summary']['effect_sizes'][f'at_{sample_size/1e6:.1f}M'] = large_effects
        
        # Generate recommendations
        recommendations = []
        
        # Check if any method consistently outperforms others
        final_performances = {}
        for method in self.methods:
            if method in self.statistical_results['overall_stats']:
                final_performances[method] = self.statistical_results['overall_stats'][method]['final_performance']['mean']
        
        if final_performances:
            best_method = max(final_performances, key=final_performances.get)
            worst_method = min(final_performances, key=final_performances.get)
            
            recommendations.append(f"Best overall IID performance: {best_method} ({final_performances[best_method]:.1f}%)")
            recommendations.append(f"Worst overall IID performance: {worst_method} ({final_performances[worst_method]:.1f}%)")
            
            # Check if difference is significant
            if len(self.statistical_results['key_points']) > 0:
                final_point = list(self.statistical_results['key_points'].keys())[-1]
                final_comps = self.statistical_results['key_points'][final_point]['comparisons']
                
                for comp_name, comp_data in final_comps.items():
                    if comp_data['significant'] and abs(comp_data['cohens_d']) > 0.5:
                        recommendations.append(f"Significant difference found: {comp_name} (p={comp_data['p_value']:.3f}, d={comp_data['cohens_d']:.2f})")
        
        report['summary']['recommendations'] = recommendations
        
        # Save report
        with open(os.path.join(save_dir, 'statistical_analysis_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create human-readable summary
        with open(os.path.join(save_dir, 'statistical_summary.txt'), 'w') as f:
            f.write("IID SAMPLE EFFICIENCY STATISTICAL ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Experiment: In-distribution sample efficiency comparison\n")
            f.write(f"Seeds analyzed: {len(self.results_data)}\n")
            f.write(f"Methods compared: {', '.join(self.methods)}\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("FINAL PERFORMANCE COMPARISON:\n")
            f.write("-" * 40 + "\n")
            for method in self.methods:
                if method in self.statistical_results['overall_stats']:
                    final_perf = self.statistical_results['overall_stats'][method]['final_performance']
                    f.write(f"{method:15}: {final_perf['mean']:5.1f}% ± {final_perf['sem']:4.1f}%\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for rec in recommendations:
                f.write(f"• {rec}\n")
        
        print(f"✅ Statistical report saved to {save_dir}")

def main():
    """Main function to run the multi-seed analysis."""
    parser = argparse.ArgumentParser(
        description="Run and analyze multi-seed IID sample efficiency experiments with checkpointing and SLURM.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Experiment Arguments ---
    parser.add_argument('--epochs', type=int, default=1000, help="Total training epochs.")
    parser.add_argument('--meta_batch_size', type=int, default=16, help='Meta batch size')
    parser.add_argument('--vanilla_batch_size', type=int, default=64, help='Vanilla SGD batch size')
    parser.add_argument('--inner_lr', type=float, default=0.05, help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Outer loop learning rate')
    parser.add_argument('--vanilla_lr', type=float, default=1e-4, help='Vanilla SGD learning rate')
    parser.add_argument('--adaptation_steps', type=int, default=5, help='Number of adaptation steps')
    parser.add_argument('--val_frequency', type=int, default=500, help='Validation frequency (in batches)')
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb', help='Data directory')
    parser.add_argument('--save_dir', type=str, 
                        default="results/sample_efficiency_iid_multiseed", 
                        help="Directory to save results and checkpoints.")

    # --- Control Flow Arguments ---
    parser.add_argument('--analyze', action='store_true', 
                        help="Run analysis on existing results. Skips experiment execution.")

    args = parser.parse_args()

    # --- Script Header ---
    print("🎯 MULTI-SEED IID SAMPLE EFFICIENCY ANALYSIS (with Checkpointing & SLURM)")
    print("="*80)
    
    seeds = [42, 43, 44, 45, 46]
    methods = ['fomaml', 'second_order', 'vanilla']
    
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print(f"Output directory: {args.save_dir}")
    print("="*80)

    analyzer = MultiSeedIIDAnalysis(args, seeds, methods)

    if args.analyze:
        print("\n📊 PHASE 1: Loading and analyzing results...")
        if not analyzer.load_results():
            print("❌ No results found. Please run the experiments first by executing this script without the --analyze flag.")
            return

        print("\n📈 PHASE 2: Statistical analysis...")
        analyzer.interpolate_curves()
        analyzer.perform_statistical_analysis()

        print("\n🎨 PHASE 3: Creating visualizations...")
        analyzer.create_publication_plots(args.save_dir)

        print("\n📝 PHASE 4: Saving reports...")
        analyzer.save_statistical_report(args.save_dir)

    else:
        print("\n🚀 PHASE 1: Generating SLURM script for experiments...")
        analyzer.generate_slurm_script()

    print("\n🎉 ANALYSIS SCRIPT COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main() 