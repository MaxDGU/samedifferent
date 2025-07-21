#!/usr/bin/env python3
"""
Parse Second-Order MAML Results from SLURM Output Files

This script parses SLURM output files to extract Second-Order MAML validation data points
and creates a sample efficiency plot showing data points seen vs validation accuracy.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

def parse_second_order_from_file(filepath):
    """Parse Second-Order MAML validation data from a single SLURM output file."""
    data_points = []
    val_accuracies = []
    
    print(f"Parsing file: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find all validation blocks for Second-Order MAML
        # Look for patterns like:
        # --- Validation at Batch 800 ---
        #   Data points seen: 4,795,442
        #   Current train accuracy: 88.23%
        #   Validation accuracy: 70.15%
        validation_pattern = r'--- Validation at Batch (\d+) ---\s+Data points seen: ([\d,]+)\s+Current train accuracy: [\d.]+%\s+Validation accuracy: ([\d.]+)%'
        
        matches = re.findall(validation_pattern, content)
        
        for batch, data_str, val_acc in matches:
            # Convert data points string to integer (remove commas)
            data_points_seen = int(data_str.replace(',', ''))
            val_accuracy = float(val_acc)
            
            data_points.append(data_points_seen)
            val_accuracies.append(val_accuracy)
        
        # Also look for epoch-level validation data if available
        # Pattern: Second-Order MAML Epoch 13: Train Loss: 0.2582, Train Acc: 87.83%, Val Loss: 1.0178, Val Acc: 69.59%
        epoch_pattern = r'Second-Order MAML Epoch (\d+): Train Loss: [\d.]+, Train Acc: [\d.]+%, Val Loss: [\d.]+, Val Acc: ([\d.]+)%'
        epoch_matches = re.findall(epoch_pattern, content)
        
        print(f"  Found {len(data_points)} batch validation points")
        print(f"  Found {len(epoch_matches)} epoch validation points")
        
        if data_points:
            print(f"  Data range: {min(data_points):,} to {max(data_points):,}")
            print(f"  Accuracy range: {min(val_accuracies):.2f}% to {max(val_accuracies):.2f}%")
        
        return data_points, val_accuracies
    
    except Exception as e:
        print(f"  Error parsing file: {e}")
        return [], []

def find_slurm_files(directory):
    """Find all SLURM output files in the directory."""
    slurm_files = []
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return slurm_files
    
    # Look for files matching pattern slurm_*_*.out
    for filename in os.listdir(directory):
        if filename.startswith('slurm_') and filename.endswith('.out'):
            slurm_files.append(os.path.join(directory, filename))
    
    slurm_files.sort()
    print(f"Found {len(slurm_files)} SLURM output files")
    
    return slurm_files

def plot_second_order_efficiency(all_data_points, all_accuracies, output_dir):
    """Create a sample efficiency plot for Second-Order MAML."""
    
    if not all_data_points:
        print("No data to plot!")
        return
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Initialize all_x for later use
    all_x = []
    for data_points in all_data_points:
        all_x.extend(data_points)
    
    # Plot individual seed curves
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_data_points)))
    
    for i, (data_points, accuracies) in enumerate(zip(all_data_points, all_accuracies)):
        if data_points and accuracies:
            plt.plot(data_points, accuracies, '-o', alpha=0.7, markersize=3,
                    color=colors[i], label=f'Seed {i}', linewidth=1.5)
    
    # Calculate and plot mean curve if we have multiple seeds
    if len(all_data_points) > 1:
        if all_x:
            min_x, max_x = min(all_x), max(all_x)
            
            # Interpolate all curves to common x-axis
            common_x = np.linspace(min_x, max_x, 50)
            interpolated_curves = []
            
            for data_points, accuracies in zip(all_data_points, all_accuracies):
                if len(data_points) > 1 and len(accuracies) > 1:
                    interp_y = np.interp(common_x, data_points, accuracies)
                    interpolated_curves.append(interp_y)
            
            if interpolated_curves:
                mean_curve = np.mean(interpolated_curves, axis=0)
                std_curve = np.std(interpolated_curves, axis=0)
                
                plt.plot(common_x, mean_curve, 'k-', linewidth=3, label='Mean', alpha=0.8)
                plt.fill_between(common_x, mean_curve - std_curve, mean_curve + std_curve,
                               alpha=0.2, color='black', label='±1 STD')
    
    # Add 70% target line
    plt.axhline(y=70, color='red', linestyle='--', linewidth=2, alpha=0.8, label='70% Target')
    
    # Formatting
    plt.xlabel('Data Points Seen', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Second-Order MAML Sample Efficiency: Data Points vs Validation Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable axis limits
    if all_x:
        plt.xlim(min(all_x) * 0.95, max(all_x) * 1.05)
    plt.ylim(45, 75)
    
    # Format x-axis to show millions
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'second_order_maml_sample_efficiency.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"✅ Plot saved to: {output_path}")

def save_extracted_data(all_data_points, all_accuracies, output_dir):
    """Save the extracted data to JSON for future use."""
    
    data = {
        'method': 'Second-Order MAML',
        'seeds': {},
        'summary': {
            'n_seeds': len(all_data_points),
            'total_data_points': sum(len(dp) for dp in all_data_points)
        }
    }
    
    for i, (data_points, accuracies) in enumerate(zip(all_data_points, all_accuracies)):
        data['seeds'][f'seed_{i}'] = {
            'data_points_seen': data_points,
            'val_accuracies': accuracies,
            'max_accuracy': max(accuracies) if accuracies else 0,
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'reached_70_percent': any(acc >= 70 for acc in accuracies) if accuracies else False
        }
        
        # Find when 70% was first reached
        if accuracies:
            for j, acc in enumerate(accuracies):
                if acc >= 70:
                    data['seeds'][f'seed_{i}']['first_70_percent'] = {
                        'data_points': data_points[j],
                        'accuracy': acc
                    }
                    break
    
    # Save to JSON
    json_path = os.path.join(output_dir, 'second_order_maml_extracted_data.json')
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Data saved to: {json_path}")
    
    # Print summary
    print("\n📊 SECOND-ORDER MAML SAMPLE EFFICIENCY SUMMARY:")
    print("="*50)
    
    seeds_reaching_70 = 0
    data_points_to_70 = []
    max_accuracies = []
    
    for i in range(len(all_data_points)):
        seed_data = data['seeds'][f'seed_{i}']
        max_acc = seed_data['max_accuracy']
        reached_70 = seed_data['reached_70_percent']
        max_accuracies.append(max_acc)
        
        print(f"Seed {i}: Max accuracy = {max_acc:.2f}%", end="")
        
        if reached_70:
            seeds_reaching_70 += 1
            first_70 = seed_data['first_70_percent']
            data_points_to_70.append(first_70['data_points'])
            print(f" | Reached 70% at {first_70['data_points']:,} data points")
        else:
            print(" | Did not reach 70%")
    
    print(f"\nSuccess rate: {seeds_reaching_70}/{len(all_data_points)} seeds reached 70%")
    
    if max_accuracies:
        mean_max_acc = np.mean(max_accuracies)
        std_max_acc = np.std(max_accuracies)
        print(f"Average max accuracy: {mean_max_acc:.2f}% ± {std_max_acc:.2f}%")
    
    if data_points_to_70:
        mean_to_70 = np.mean(data_points_to_70)
        std_to_70 = np.std(data_points_to_70)
        print(f"Average data points to 70%: {mean_to_70:,.0f} ± {std_to_70:,.0f}")
    else:
        print("No seeds reached 70% accuracy yet")

def main():
    parser = argparse.ArgumentParser(description='Parse Second-Order MAML results from SLURM output files')
    parser.add_argument('--input_dir', type=str, 
                       default='/scratch/gpfs/mg7411/samedifferent/results/sample_efficiency_second_order_only/slurm_out/',
                       help='Directory containing SLURM output files')
    parser.add_argument('--output_dir', type=str, default='results/second_order_maml_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 SECOND-ORDER MAML SAMPLE EFFICIENCY ANALYSIS")
    print("="*50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*50)
    
    # Find all SLURM files
    slurm_files = find_slurm_files(args.input_dir)
    
    if not slurm_files:
        print("❌ No SLURM output files found!")
        return
    
    # Parse each file
    all_data_points = []
    all_accuracies = []
    
    for filepath in slurm_files:
        data_points, val_accuracies = parse_second_order_from_file(filepath)
        if data_points and val_accuracies:
            all_data_points.append(data_points)
            all_accuracies.append(val_accuracies)
    
    if not all_data_points:
        print("❌ No Second-Order MAML data found in any files!")
        return
    
    print(f"\n✅ Successfully parsed {len(all_data_points)} seeds")
    
    # Create plot
    plot_second_order_efficiency(all_data_points, all_accuracies, args.output_dir)
    
    # Save extracted data
    save_extracted_data(all_data_points, all_accuracies, args.output_dir)
    
    print("\n🎉 Analysis completed!")

if __name__ == '__main__':
    main() 