#!/usr/bin/env python3
"""
Plot Sample Efficiency Results

This script loads and plots the sample efficiency comparison results
from FOMAML, Second-Order MAML, and Vanilla SGD training.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def load_results(results_dir):
    """Load all available results from the specified directory."""
    results = {}
    
    # Define the expected result files
    result_files = {
        'FOMAML': 'fomaml_results.json',
        'Second-Order MAML': 'second_order_maml_results.json',
        'Vanilla SGD': 'vanilla_sgd_results.json'
    }
    
    for method, filename in result_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    results[method] = json.load(f)
                print(f"Loaded {method} results from {filepath}")
            except Exception as e:
                print(f"Error loading {method} results: {e}")
        else:
            print(f"Warning: {method} results not found at {filepath}")
    
    return results

def plot_sample_efficiency(results, save_path=None):
    """Plot sample efficiency comparison."""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'FOMAML': 'blue',
        'Second-Order MAML': 'red',
        'Vanilla SGD': 'green'
    }
    
    markers = {
        'FOMAML': 'o',
        'Second-Order MAML': 's',
        'Vanilla SGD': '^'
    }
    
    for method, data in results.items():
        if 'data_points_seen' in data and 'val_accuracies' in data:
            x = data['data_points_seen']
            y = data['val_accuracies']
            
            plt.plot(x, y, 
                    color=colors.get(method, 'black'), 
                    marker=markers.get(method, 'o'),
                    linewidth=2, 
                    markersize=4,
                    label=method,
                    alpha=0.8)
    
    plt.xlabel('Number of Training Samples Seen', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Sample Efficiency Comparison: MAML vs Vanilla SGD', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    plt.xlim(left=0)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*50)
    print("SAMPLE EFFICIENCY COMPARISON SUMMARY")
    print("="*50)
    
    for method, data in results.items():
        if 'data_points_seen' in data and 'val_accuracies' in data:
            max_acc = max(data['val_accuracies'])
            total_data = data.get('total_data_points', data['data_points_seen'][-1])
            final_acc = data['val_accuracies'][-1]
            
            print(f"\n{method}:")
            print(f"  Total data points seen: {total_data:,}")
            print(f"  Final validation accuracy: {final_acc:.2f}%")
            print(f"  Maximum validation accuracy: {max_acc:.2f}%")
            
            # Find data points needed to reach certain accuracy thresholds
            for threshold in [60, 70, 80, 90]:
                for i, acc in enumerate(data['val_accuracies']):
                    if acc >= threshold:
                        data_points = data['data_points_seen'][i]
                        print(f"  Data points to reach {threshold}%: {data_points:,}")
                        break
                else:
                    print(f"  Data points to reach {threshold}%: Not achieved")

def main():
    parser = argparse.ArgumentParser(description='Plot sample efficiency results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing the results JSON files')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the plot (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found to plot.")
        return
    
    # Print summary
    print_summary(results)
    
    # Create plot
    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(args.results_dir, 'sample_efficiency_comparison.png')
    
    plot_sample_efficiency(results, save_path)

if __name__ == '__main__':
    main() 