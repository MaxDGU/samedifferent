#!/usr/bin/env python3
"""
Aggregate Vanilla SGD Validation Results Across Seeds

This script aggregates results from multiple seeds for each architecture
and produces mean ± std statistics with comprehensive plots.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from collections import defaultdict

def load_results(results_dir, seeds, architectures):
    """Load results from all seed directories."""
    all_results = defaultdict(list)
    
    print("Loading results from:")
    for seed in seeds:
        seed_dir = os.path.join(results_dir, f"seed_{seed}")
        results_file = os.path.join(seed_dir, "validation_results.json")
        
        if os.path.exists(results_file):
            print(f"  {results_file}")
            with open(results_file, 'r') as f:
                data = json.load(f)
                
            # Extract results for each architecture
            for result in data['results']:
                arch = result['architecture']
                if arch.lower() in [a.lower() for a in architectures]:
                    all_results[arch].append({
                        'seed': seed,
                        'test_acc': result['test_acc'],
                        'best_val_acc': result['best_val_acc'],
                        'parameters': result['parameters']
                    })
        else:
            print(f"  WARNING: Missing {results_file}")
    
    return all_results

def calculate_statistics(all_results):
    """Calculate mean and std for each architecture."""
    stats = {}
    
    for arch, results in all_results.items():
        if len(results) == 0:
            continue
            
        test_accs = [r['test_acc'] for r in results]
        val_accs = [r['best_val_acc'] for r in results]
        params = results[0]['parameters']  # Same for all seeds
        
        stats[arch] = {
            'test_acc_mean': np.mean(test_accs),
            'test_acc_std': np.std(test_accs),
            'val_acc_mean': np.mean(val_accs),
            'val_acc_std': np.std(val_accs),
            'parameters': params,
            'num_seeds': len(results),
            'test_accs': test_accs,
            'val_accs': val_accs
        }
    
    return stats

def plot_aggregated_results(stats, save_dir):
    """Create comprehensive plots of aggregated results."""
    
    # Prepare data
    architectures = list(stats.keys())
    test_means = [stats[arch]['test_acc_mean'] for arch in architectures]
    test_stds = [stats[arch]['test_acc_std'] for arch in architectures]
    val_means = [stats[arch]['val_acc_mean'] for arch in architectures]
    val_stds = [stats[arch]['val_acc_std'] for arch in architectures]
    
    x = np.arange(len(architectures))
    width = 0.35
    
    # Main comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars1 = ax.bar(x - width/2, test_means, width, yerr=test_stds, 
                   label='Test Accuracy', alpha=0.8, color='steelblue', capsize=5)
    bars2 = ax.bar(x + width/2, val_means, width, yerr=val_stds,
                   label='Best Val Accuracy', alpha=0.8, color='orange', capsize=5)
    
    ax.set_xlabel('Architecture', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Vanilla SGD Performance Across Architectures\n(Mean ± Std across 5 seeds)', 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bars, means, stds) in enumerate([(bars1, test_means, test_stds), 
                                            (bars2, val_means, val_stds)]):
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.annotate(f'{mean:.1f}±{std:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 5),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add chance level line
    ax.axhline(y=50.0, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='Chance Level (50%)')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vanilla_sgd_aggregated_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'vanilla_sgd_aggregated_results.pdf'), 
                bbox_inches='tight')
    
    # Individual seed results plot
    fig2, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Test accuracy scatter plot
    for i, arch in enumerate(architectures):
        test_accs = stats[arch]['test_accs']
        axes[0].scatter([i] * len(test_accs), test_accs, alpha=0.7, s=60, 
                       label=f'{arch} (n={len(test_accs)})')
        axes[0].errorbar(i, stats[arch]['test_acc_mean'], 
                        yerr=stats[arch]['test_acc_std'],
                        fmt='o', color='black', markersize=8, capsize=5)
    
    axes[0].set_xlabel('Architecture', fontweight='bold')
    axes[0].set_ylabel('Test Accuracy (%)', fontweight='bold')
    axes[0].set_title('Test Accuracy Distribution Across Seeds', fontweight='bold')
    axes[0].set_xticks(range(len(architectures)))
    axes[0].set_xticklabels(architectures)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=50.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Validation accuracy scatter plot
    for i, arch in enumerate(architectures):
        val_accs = stats[arch]['val_accs']
        axes[1].scatter([i] * len(val_accs), val_accs, alpha=0.7, s=60,
                       label=f'{arch} (n={len(val_accs)})')
        axes[1].errorbar(i, stats[arch]['val_acc_mean'], 
                        yerr=stats[arch]['val_acc_std'],
                        fmt='o', color='black', markersize=8, capsize=5)
    
    axes[1].set_xlabel('Architecture', fontweight='bold')
    axes[1].set_ylabel('Best Validation Accuracy (%)', fontweight='bold')
    axes[1].set_title('Validation Accuracy Distribution Across Seeds', fontweight='bold')
    axes[1].set_xticks(range(len(architectures)))
    axes[1].set_xticklabels(architectures)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=50.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vanilla_sgd_seed_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'vanilla_sgd_seed_distribution.pdf'), 
                bbox_inches='tight')
    
    print(f"Plots saved to:")
    print(f"  {os.path.join(save_dir, 'vanilla_sgd_aggregated_results.png')}")
    print(f"  {os.path.join(save_dir, 'vanilla_sgd_seed_distribution.png')}")
    
    return fig, fig2

def main():
    parser = argparse.ArgumentParser(description='Aggregate Vanilla SGD Validation Results')
    
    parser.add_argument('--results_dir', type=str, default='results/vanilla_sgd_validation_array',
                       help='Directory containing seed results')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46],
                       help='Seeds to aggregate')
    parser.add_argument('--architectures', nargs='+', default=['CONV2', 'CONV4', 'CONV6'],
                       help='Architectures to aggregate')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save aggregated results (default: results_dir)')
    
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = args.results_dir
    
    print("="*80)
    print("VANILLA SGD VALIDATION RESULTS AGGREGATION")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Seeds: {args.seeds}")
    print(f"Architectures: {args.architectures}")
    print(f"Save directory: {args.save_dir}")
    print("="*80)
    
    # Load results
    all_results = load_results(args.results_dir, args.seeds, args.architectures)
    
    if not all_results:
        print("ERROR: No results found!")
        return
    
    # Calculate statistics
    stats = calculate_statistics(all_results)
    
    # Create plots
    os.makedirs(args.save_dir, exist_ok=True)
    plot_aggregated_results(stats, args.save_dir)
    
    # Save aggregated results
    aggregated_results = {
        'experiment': 'vanilla_sgd_validation_aggregated',
        'description': 'Aggregated vanilla SGD validation results across multiple seeds',
        'seeds': args.seeds,
        'architectures': args.architectures,
        'statistics': stats,
        'raw_results': dict(all_results),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(args.save_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("AGGREGATED RESULTS SUMMARY")
    print("="*80)
    print(f"{'Architecture':<12} {'Parameters':<12} {'Test Acc (%)':<15} {'Val Acc (%)':<15} {'Seeds':<8}")
    print("-" * 80)
    
    for arch in sorted(stats.keys()):
        s = stats[arch]
        print(f"{arch:<12} {s['parameters']:<12,} "
              f"{s['test_acc_mean']:.1f}±{s['test_acc_std']:.1f}%{'':<6} "
              f"{s['val_acc_mean']:.1f}±{s['val_acc_std']:.1f}%{'':<6} "
              f"{s['num_seeds']:<8}")
    
    print("-" * 80)
    print(f"Results saved to: {args.save_dir}")
    print("="*80)
    
    # Check if results are at chance level
    print("\nINTERPRETation:")
    all_near_chance = True
    for arch, s in stats.items():
        if s['test_acc_mean'] > 55.0:  # More than 5% above chance
            all_near_chance = False
            print(f"⚠️  {arch}: {s['test_acc_mean']:.1f}% - Above chance level!")
        else:
            print(f"✓ {arch}: {s['test_acc_mean']:.1f}% - Near chance level as expected")
    
    if all_near_chance:
        print("\n✅ VALIDATION SUCCESSFUL: All architectures perform near chance level (~50%)")
        print("This confirms that vanilla SGD struggles on PB tasks, validating that")
        print("meta-learning improvements in your bar charts were genuine and significant!")
    else:
        print("\n❓ UNEXPECTED: Some architectures perform above chance level")
        print("This may indicate differences in experimental setup or data")

if __name__ == '__main__':
    main()
