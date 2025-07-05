#!/usr/bin/env python3
"""
Plot OOD Sample Efficiency Results

This script loads and plots the out-of-distribution (OOD) sample efficiency 
comparison results from the holdout experiment, comparing how FOMAML, 
Second-Order MAML, and Vanilla SGD perform on novel tasks.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def load_ood_results(results_dir):
    """Load OOD holdout experiment results."""
    results = {}
    
    # Look for holdout result files
    result_patterns = {
        'FOMAML': 'fomaml_holdout_*_results.json',
        'Second-Order MAML': 'second_order_maml_holdout_*_results.json',
        'Vanilla SGD': 'vanilla_sgd_holdout_*_results.json'
    }
    
    for method, pattern in result_patterns.items():
        # Find files matching the pattern
        result_files = list(Path(results_dir).glob(pattern))
        
        if result_files:
            # Use the first matching file
            filepath = result_files[0]
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[method] = data
                    print(f"Loaded {method} results from {filepath}")
            except Exception as e:
                print(f"Error loading {method} results from {filepath}: {e}")
        else:
            print(f"No results found for {method} (pattern: {pattern})")
    
    return results

def plot_ood_sample_efficiency(results, save_path=None, show_plot=True):
    """Create the main OOD sample efficiency plot."""
    plt.figure(figsize=(14, 10))
    
    colors = {'FOMAML': '#1f77b4', 'Second-Order MAML': '#ff7f0e', 'Vanilla SGD': '#2ca02c'}
    linestyles = {'FOMAML': '-', 'Second-Order MAML': '--', 'Vanilla SGD': '-.'}
    markers = {'FOMAML': 'o', 'Second-Order MAML': 's', 'Vanilla SGD': '^'}
    
    holdout_task = None
    train_tasks = None
    
    # Plot each method's results
    for method, data in results.items():
        if 'data_points_seen' in data and 'val_accuracies' in data:
            x = data['data_points_seen']
            y = data['val_accuracies']
            
            # Store experiment info
            if holdout_task is None:
                holdout_task = data.get('holdout_task', 'unknown')
                train_tasks = data.get('train_tasks', [])
            
            plt.plot(x, y, 
                    color=colors.get(method, 'black'), 
                    linestyle=linestyles.get(method, '-'),
                    linewidth=3, 
                    label=method, 
                    marker=markers.get(method, 'o'), 
                    markersize=6,
                    markevery=max(1, len(x)//10))  # Show markers every 10% of points
    
    # Formatting
    plt.xlabel('Number of Training Data Points Seen', fontsize=14, fontweight='bold')
    plt.ylabel('Out-of-Distribution Validation Accuracy (%)', fontsize=14, fontweight='bold')
    
    title = f'Out-of-Distribution Sample Efficiency\nHoldout Task: "{holdout_task}" | Training Tasks: {len(train_tasks)} others'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.legend(fontsize=12, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Add text box with experiment details
    if holdout_task and train_tasks:
        experiment_info = f"Experiment Setup:\n• Holdout (OOD) Task: {holdout_task}\n• Training Tasks: {len(train_tasks)}\n• Tests novel task generalization"
        plt.text(0.02, 0.98, experiment_info, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return plt.gcf()

def plot_final_performance_comparison(results, save_path=None, show_plot=True):
    """Create a bar chart comparing final OOD performance."""
    methods = []
    final_accuracies = []
    max_accuracies = []
    data_points = []
    
    for method, data in results.items():
        if 'val_accuracies' in data and data['val_accuracies']:
            methods.append(method)
            final_accuracies.append(data['val_accuracies'][-1])
            max_accuracies.append(max(data['val_accuracies']))
            data_points.append(data.get('total_data_points', 0))
    
    if not methods:
        print("No valid results to plot")
        return None
    
    # Create subplot with two charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Chart 1: Final vs Max Accuracy
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, final_accuracies, width, label='Final OOD Accuracy', 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, max_accuracies, width, label='Maximum OOD Accuracy', 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.5)
    
    ax1.set_xlabel('Method', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Final vs Maximum OOD Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    # Chart 2: Data Efficiency (Accuracy per Data Point)
    efficiency = [acc / (dp / 1000) for acc, dp in zip(final_accuracies, data_points)]  # Accuracy per 1K data points
    
    bars3 = ax2.bar(methods, efficiency, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax2.set_xlabel('Method', fontweight='bold')
    ax2.set_ylabel('OOD Accuracy per 1K Data Points', fontweight='bold')
    ax2.set_title('Data Efficiency Comparison', fontweight='bold')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        comparison_path = save_path.replace('.png', '_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.savefig(comparison_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Comparison plot saved to {comparison_path}")
    
    if show_plot:
        plt.show()
    
    return fig

def generate_summary_report(results, save_path=None):
    """Generate a text summary of the OOD experiment results."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("OUT-OF-DISTRIBUTION SAMPLE EFFICIENCY EXPERIMENT SUMMARY")
    report_lines.append("=" * 80)
    
    if results:
        # Get experiment details from first result
        first_result = next(iter(results.values()))
        holdout_task = first_result.get('holdout_task', 'unknown')
        train_tasks = first_result.get('train_tasks', [])
        
        report_lines.append(f"Holdout (OOD) Task: {holdout_task}")
        report_lines.append(f"Training Tasks ({len(train_tasks)}): {', '.join(train_tasks)}")
        report_lines.append("")
        report_lines.append("RESULTS SUMMARY:")
        report_lines.append("-" * 50)
        
        # Sort methods by final performance
        method_performance = []
        for method, data in results.items():
            if 'val_accuracies' in data and data['val_accuracies']:
                final_acc = data['val_accuracies'][-1]
                max_acc = max(data['val_accuracies'])
                total_data = data.get('total_data_points', 0)
                improvement = final_acc - data['val_accuracies'][0] if len(data['val_accuracies']) > 1 else 0
                
                method_performance.append({
                    'method': method,
                    'final_acc': final_acc,
                    'max_acc': max_acc,
                    'total_data': total_data,
                    'improvement': improvement,
                    'efficiency': final_acc / (total_data / 1000)  # Acc per 1K data points
                })
        
        # Sort by final accuracy (descending)
        method_performance.sort(key=lambda x: x['final_acc'], reverse=True)
        
        for i, perf in enumerate(method_performance, 1):
            report_lines.append(f"{i}. {perf['method']}")
            report_lines.append(f"   Final OOD Accuracy: {perf['final_acc']:.2f}%")
            report_lines.append(f"   Maximum OOD Accuracy: {perf['max_acc']:.2f}%")
            report_lines.append(f"   Improvement: {perf['improvement']:+.2f}%")
            report_lines.append(f"   Data Points Used: {perf['total_data']:,}")
            report_lines.append(f"   Data Efficiency: {perf['efficiency']:.3f} acc/1K points")
            report_lines.append("")
        
        # Analysis
        report_lines.append("ANALYSIS:")
        report_lines.append("-" * 20)
        
        if len(method_performance) >= 2:
            best_method = method_performance[0]
            if 'MAML' in best_method['method'] and any('Vanilla' in m['method'] for m in method_performance):
                vanilla_perf = next(m for m in method_performance if 'Vanilla' in m['method'])
                advantage = best_method['final_acc'] - vanilla_perf['final_acc']
                
                if advantage > 5:
                    report_lines.append(f"✅ Meta-learning shows clear OOD advantage: {advantage:.1f}% better than Vanilla SGD")
                elif advantage > 0:
                    report_lines.append(f"⚠️  Meta-learning shows modest OOD advantage: {advantage:.1f}% better than Vanilla SGD")
                else:
                    report_lines.append(f"❌ Vanilla SGD outperforms meta-learning by {-advantage:.1f}% on OOD task")
                    report_lines.append("   This suggests the holdout task may not be sufficiently different from training tasks")
    
    else:
        report_lines.append("No valid results found!")
    
    report_lines.append("=" * 80)
    
    # Save report
    if save_path:
        report_path = save_path.replace('.png', '_summary.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"Summary report saved to {report_path}")
    
    # Print to console
    print('\n'.join(report_lines))
    
    return report_lines

def main():
    parser = argparse.ArgumentParser(description='Plot OOD Sample Efficiency Results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing the OOD experiment results')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the plot (optional)')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plots (useful for batch processing)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory {args.results_dir} does not exist")
        return
    
    print(f"Loading OOD results from: {args.results_dir}")
    results = load_ood_results(args.results_dir)
    
    if not results:
        print("No valid results found!")
        return
    
    print(f"Found results for {len(results)} methods: {list(results.keys())}")
    
    # Generate plots
    show_plots = not args.no_show
    
    # Main sample efficiency plot
    save_path = args.save_path or os.path.join(args.results_dir, 'ood_sample_efficiency_plot.png')
    plot_ood_sample_efficiency(results, save_path, show_plots)
    
    # Comparison charts
    plot_final_performance_comparison(results, save_path, show_plots)
    
    # Summary report
    generate_summary_report(results, save_path)
    
    print(f"\nPlotting completed! Check {args.results_dir} for output files.")

if __name__ == '__main__':
    main() 