#!/usr/bin/env python3
"""
Plot Sample Efficiency Results

This script loads saved results from the sample efficiency comparison
and creates visualization plots.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def load_results(results_dir):
    """Load results from JSON files."""
    results_list = []
    
    # Try to load combined results first
    combined_path = os.path.join(results_dir, 'combined_results.json')
    if os.path.exists(combined_path):
        with open(combined_path, 'r') as f:
            data = json.load(f)
        return data['results']
    
    # Otherwise, load individual result files
    result_files = {
        'fomaml_results.json': 'FOMAML',
        'second_order_maml_results.json': 'Second-Order MAML',
        'vanilla_sgd_results.json': 'Vanilla SGD'
    }
    
    for filename, method_name in result_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            results_list.append(data)
    
    return results_list

def plot_comparison(results_list, save_dir, title_suffix=""):
    """Create comparison plot of sample efficiency."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, results in enumerate(results_list):
        if 'data_points_seen' in results and 'val_accuracies' in results:
            plt.plot(results['data_points_seen'], results['val_accuracies'], 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)], 
                    linewidth=2, 
                    label=results['method'], 
                    marker=markers[i % len(markers)], 
                    markersize=4,
                    alpha=0.8)
    
    plt.xlabel('Number of Data Points Seen', fontsize=14)
    plt.ylabel('Validation Accuracy (%)', fontsize=14)
    plt.title(f'Sample Efficiency Comparison: Conv6 Architecture{title_suffix}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(save_dir, 'sample_efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'sample_efficiency_comparison.pdf'), bbox_inches='tight')
    
    print(f"Comparison plot saved to {os.path.join(save_dir, 'sample_efficiency_comparison.png')}")
    
    return plt

def plot_individual_methods(results_list, save_dir):
    """Create individual plots for each method."""
    for results in results_list:
        if 'data_points_seen' in results and 'val_accuracies' in results:
            plt.figure(figsize=(10, 6))
            plt.plot(results['data_points_seen'], results['val_accuracies'], 
                    linewidth=2, marker='o', markersize=4)
            plt.xlabel('Number of Data Points Seen', fontsize=12)
            plt.ylabel('Validation Accuracy (%)', fontsize=12)
            plt.title(f'{results["method"]} - Sample Efficiency', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save individual plot
            method_name = results['method'].lower().replace(' ', '_').replace('-', '_')
            plt.savefig(os.path.join(save_dir, f'{method_name}_sample_efficiency.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def print_summary(results_list):
    """Print summary statistics."""
    print("\n=== Sample Efficiency Summary ===")
    
    for results in results_list:
        if 'data_points_seen' in results and 'val_accuracies' in results:
            method = results['method']
            total_data = results.get('total_data_points', max(results['data_points_seen']))
            final_acc = results['val_accuracies'][-1] if results['val_accuracies'] else 0
            max_acc = max(results['val_accuracies']) if results['val_accuracies'] else 0
            
            # Find data points needed to reach certain accuracy thresholds
            thresholds = [60, 70, 80, 90]
            data_for_threshold = {}
            
            for threshold in thresholds:
                for i, acc in enumerate(results['val_accuracies']):
                    if acc >= threshold:
                        data_for_threshold[threshold] = results['data_points_seen'][i]
                        break
            
            print(f"\n{method}:")
            print(f"  Total data points seen: {total_data:,}")
            print(f"  Final validation accuracy: {final_acc:.2f}%")
            print(f"  Maximum validation accuracy: {max_acc:.2f}%")
            
            print("  Data points needed to reach:")
            for threshold in thresholds:
                if threshold in data_for_threshold:
                    print(f"    {threshold}% accuracy: {data_for_threshold[threshold]:,} data points")
                else:
                    print(f"    {threshold}% accuracy: Not reached")

def main():
    parser = argparse.ArgumentParser(description='Plot Sample Efficiency Results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing the results JSON files')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save plots (defaults to results_dir)')
    parser.add_argument('--title_suffix', type=str, default='',
                       help='Suffix to add to plot title')
    parser.add_argument('--individual_plots', action='store_true',
                       help='Create individual plots for each method')
    
    args = parser.parse_args()
    
    # Set save directory
    save_dir = args.save_dir if args.save_dir else args.results_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_dir}")
    results_list = load_results(args.results_dir)
    
    if not results_list:
        print("No results found! Please check the results directory.")
        return
    
    print(f"Found {len(results_list)} result sets:")
    for results in results_list:
        print(f"  - {results.get('method', 'Unknown method')}")
    
    # Create comparison plot
    plot_comparison(results_list, save_dir, args.title_suffix)
    
    # Create individual plots if requested
    if args.individual_plots:
        plot_individual_methods(results_list, save_dir)
    
    # Print summary
    print_summary(results_list)
    
    print(f"\nPlots saved to {save_dir}")

if __name__ == '__main__':
    main() 