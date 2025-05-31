#!/usr/bin/env python
"""
Compile results from all 10 seeds of conv2lr
"""

import os
import json
import pandas as pd
import numpy as np
import argparse

def compile_results(results_dir, seeds=None):
    """Compile results from all seeds into a single dataframe"""
    if seeds is None:
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    
    all_results = []
    
    for seed in seeds:
        seed_dir = os.path.join(results_dir, 'conv2', f'seed_{seed}')
        results_file = os.path.join(seed_dir, 'results.json')
        
        if not os.path.exists(results_file):
            print(f"Warning: Results file not found for seed {seed}")
            continue
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Extract test results for each task
            for task, metrics in data['test_results'].items():
                row = {
                    'seed': seed,
                    'task': task,
                    'test_accuracy': metrics['accuracy'],
                    'test_loss': metrics['loss'],
                    'best_val_accuracy': data['best_val_metrics']['accuracy'],
                    'best_val_loss': data['best_val_metrics']['loss'],
                    'best_epoch': data['best_val_metrics']['epoch']
                }
                all_results.append(row)
                
        except Exception as e:
            print(f"Error processing seed {seed}: {str(e)}")
    
    # Convert to dataframe
    if all_results:
        df = pd.DataFrame(all_results)
        return df
    else:
        print("No results found")
        return None

def main():
    parser = argparse.ArgumentParser(description='Compile conv2lr results')
    parser.add_argument('--results_dir', type=str, default='results/conv2lr_10seeds',
                        help='Directory containing results')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file for compiled results (default: results_dir/compiled_results.csv)')
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = os.path.join(args.results_dir, 'compiled_results.csv')
    
    # Compile results
    df = compile_results(args.results_dir)
    
    if df is not None:
        # Calculate mean and std across seeds for each task
        summary = df.groupby('task').agg({
            'test_accuracy': ['mean', 'std'],
            'test_loss': ['mean', 'std'],
            'best_val_accuracy': ['mean', 'std'],
            'best_epoch': ['mean', 'std']
        }).reset_index()
        
        # Save results
        df.to_csv(args.output_file, index=False)
        summary_file = os.path.splitext(args.output_file)[0] + '_summary.csv'
        summary.to_csv(summary_file)
        
        print(f"Results saved to {args.output_file}")
        print(f"Summary saved to {summary_file}")
        
        # Print summary
        print("\nSummary of test accuracy by task:")
        for _, row in summary.iterrows():
            task = row['task']
            mean_acc = row[('test_accuracy', 'mean')]
            std_acc = row[('test_accuracy', 'std')]
            print(f"{task}: {mean_acc:.4f} ± {std_acc:.4f}")
        
        # Print overall average
        overall_mean = df['test_accuracy'].mean()
        overall_std = df['test_accuracy'].std()
        print(f"\nOverall average test accuracy: {overall_mean:.4f} ± {overall_std:.4f}")

if __name__ == "__main__":
    main() 