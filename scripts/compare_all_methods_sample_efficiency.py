#!/usr/bin/env python3
"""
Compare All Methods Sample Efficiency

This script aggregates results from FOMAML, Second-Order MAML, and SGD parsers
and creates a comprehensive comparison plot showing sample efficiency across all methods.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import argparse
from pathlib import Path

def load_parsed_results(results_dir, method_name):
    """Load results from a parsed analysis directory."""
    # Look for JSON files with extracted data
    json_files = {
        'fomaml': 'fomaml_extracted_data.json',
        'second_order': 'second_order_maml_extracted_data.json', 
        'sgd': 'sgd_extracted_data.json'
    }
    
    json_file = json_files.get(method_name.lower())
    if not json_file:
        print(f"⚠️  Unknown method: {method_name}")
        return None
        
    json_path = os.path.join(results_dir, json_file)
    
    if not os.path.exists(json_path):
        print(f"⚠️  Results file not found: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {method_name} results from {json_path}")
        return data
    except Exception as e:
        print(f"❌ Error loading {method_name} results: {e}")
        return None

def extract_curves_from_data(data, method_name):
    """Extract individual seed curves from parsed data."""
    if not data or 'seeds' not in data:
        return [], []
    
    all_data_points = []
    all_accuracies = []
    
    for seed_key, seed_data in data['seeds'].items():
        if 'data_points_seen' in seed_data and 'val_accuracies' in seed_data:
            data_points = seed_data['data_points_seen']
            accuracies = seed_data['val_accuracies']
            
            if len(data_points) > 0 and len(accuracies) > 0:
                all_data_points.append(data_points)
                all_accuracies.append(accuracies)
    
    print(f"  Extracted {len(all_data_points)} seed curves for {method_name}")
    return all_data_points, all_accuracies

def interpolate_curves(all_data_points, all_accuracies, method_name, n_points=100):
    """Interpolate all curves to a common x-axis for statistical analysis."""
    if not all_data_points:
        return None, None
    
    # Find global data range across all seeds
    all_x = []
    for data_points in all_data_points:
        all_x.extend(data_points)
    
    if not all_x:
        return None, None
    
    min_x, max_x = min(all_x), max(all_x)
    common_x = np.linspace(min_x, max_x, n_points)
    
    # Interpolate each seed curve
    interpolated_curves = []
    for data_points, accuracies in zip(all_data_points, all_accuracies):
        if len(data_points) > 1 and len(accuracies) > 1:
            # Create interpolation function
            interp_func = interp1d(data_points, accuracies, 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
            interp_y = interp_func(common_x)
            
            # Only keep points within the original data range for this seed
            seed_min, seed_max = min(data_points), max(data_points)
            valid_mask = (common_x >= seed_min) & (common_x <= seed_max)
            
            if np.any(valid_mask):
                interpolated_curves.append(interp_y)
    
    if not interpolated_curves:
        return None, None
    
    # Convert to numpy array
    interpolated_curves = np.array(interpolated_curves)
    
    print(f"  Interpolated {len(interpolated_curves)} curves for {method_name}")
    return common_x, interpolated_curves

def calculate_statistics(common_x, interpolated_curves):
    """Calculate mean and standard deviation across seeds."""
    if interpolated_curves is None or len(interpolated_curves) == 0:
        return None, None, None
    
    # Calculate statistics at each point
    mean_curve = np.mean(interpolated_curves, axis=0)
    std_curve = np.std(interpolated_curves, axis=0)
    
    # Calculate confidence intervals (±1 std)
    upper_ci = mean_curve + std_curve
    lower_ci = mean_curve - std_curve
    
    return mean_curve, upper_ci, lower_ci

def create_comparison_plot(methods_data, output_dir):
    """Create comprehensive comparison plot."""
    
    plt.figure(figsize=(14, 10))
    
    # Color scheme for methods
    colors = {
        'fomaml': '#2E8B57',      # Sea Green
        'second_order': '#4169E1', # Royal Blue  
        'sgd': '#DC143C'          # Crimson
    }
    
    method_labels = {
        'fomaml': 'FOMAML',
        'second_order': 'Second-Order MAML',
        'sgd': 'Vanilla SGD'
    }
    
    # Track ranges for axis limits
    all_x_ranges = []
    all_y_ranges = []
    
    for method_name, method_data in methods_data.items():
        if method_data is None:
            continue
            
        common_x, interpolated_curves = method_data['interpolated']
        if common_x is None or interpolated_curves is None:
            continue
            
        mean_curve, upper_ci, lower_ci = method_data['statistics']
        if mean_curve is None:
            continue
        
        color = colors.get(method_name, '#666666')
        label = method_labels.get(method_name, method_name.upper())
        
        # Plot mean curve
        plt.plot(common_x, mean_curve, color=color, linewidth=3, 
                label=f'{label} (Mean)', alpha=0.9)
        
        # Plot confidence interval
        plt.fill_between(common_x, lower_ci, upper_ci, 
                        color=color, alpha=0.2, 
                        label=f'{label} (±1 STD)')
        
        # Track ranges
        all_x_ranges.extend([min(common_x), max(common_x)])
        all_y_ranges.extend([min(lower_ci), max(upper_ci)])
        
        # Print statistics
        final_mean = mean_curve[-1]
        final_std = (upper_ci[-1] - lower_ci[-1]) / 2
        max_mean = max(mean_curve)
        print(f"📊 {label}:")
        print(f"   Final accuracy: {final_mean:.2f}% ± {final_std:.2f}%")
        print(f"   Maximum accuracy: {max_mean:.2f}%")
        print(f"   Data range: {min(common_x):,.0f} to {max(common_x):,.0f}")
    
    # Add 70% target line
    plt.axhline(y=70, color='red', linestyle='--', linewidth=2, 
               alpha=0.8, label='70% Target')
    
    # Formatting
    plt.xlabel('Data Points Seen', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title('Sample Efficiency Comparison: FOMAML vs Second-Order MAML vs Vanilla SGD', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Set reasonable axis limits
    if all_x_ranges and all_y_ranges:
        x_margin = (max(all_x_ranges) - min(all_x_ranges)) * 0.05
        y_margin = (max(all_y_ranges) - min(all_y_ranges)) * 0.05
        
        plt.xlim(min(all_x_ranges) - x_margin, max(all_x_ranges) + x_margin)
        plt.ylim(max(45, min(all_y_ranges) - y_margin), 
                min(85, max(all_y_ranges) + y_margin))
    
    # Format x-axis to show millions
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plots
    output_path = os.path.join(output_dir, 'sample_efficiency_comparison_all_methods.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    print(f"\n✅ Comparison plot saved to: {output_path}")
    
    return output_path

def analyze_method_performance(methods_data):
    """Analyze and compare performance across methods."""
    
    print(f"\n{'='*80}")
    print("🎯 COMPREHENSIVE SAMPLE EFFICIENCY ANALYSIS")
    print(f"{'='*80}")
    
    performance_summary = {}
    
    for method_name, method_data in methods_data.items():
        if method_data is None or method_data['statistics'][0] is None:
            continue
            
        common_x, interpolated_curves = method_data['interpolated']
        mean_curve, upper_ci, lower_ci = method_data['statistics']
        
        # Find when method reaches 70% (if ever)
        reaches_70_indices = np.where(mean_curve >= 70)[0]
        
        performance_summary[method_name] = {
            'final_accuracy_mean': mean_curve[-1],
            'final_accuracy_std': (upper_ci[-1] - lower_ci[-1]) / 2,
            'max_accuracy_mean': max(mean_curve),
            'total_data_points': max(common_x),
            'reaches_70': len(reaches_70_indices) > 0,
            'data_to_70': common_x[reaches_70_indices[0]] if len(reaches_70_indices) > 0 else None,
            'n_seeds': len(interpolated_curves)
        }
    
    # Print detailed analysis
    method_labels = {
        'fomaml': 'FOMAML',
        'second_order': 'Second-Order MAML', 
        'sgd': 'Vanilla SGD'
    }
    
    print("\n📈 FINAL ACCURACY COMPARISON:")
    print("-" * 50)
    sorted_methods = sorted(performance_summary.items(), 
                          key=lambda x: x[1]['final_accuracy_mean'], reverse=True)
    
    for i, (method, perf) in enumerate(sorted_methods, 1):
        label = method_labels.get(method, method.upper())
        print(f"{i}. {label:20} | {perf['final_accuracy_mean']:6.2f}% ± {perf['final_accuracy_std']:4.2f}% | {perf['n_seeds']} seeds")
    
    print("\n🎯 TIME TO 70% ACCURACY:")
    print("-" * 50)
    methods_reaching_70 = [(method, perf) for method, perf in performance_summary.items() 
                          if perf['reaches_70']]
    
    if methods_reaching_70:
        sorted_70 = sorted(methods_reaching_70, key=lambda x: x[1]['data_to_70'])
        for i, (method, perf) in enumerate(sorted_70, 1):
            label = method_labels.get(method, method.upper())
            print(f"{i}. {label:20} | {perf['data_to_70']:8,.0f} data points")
    else:
        print("❌ No methods reached 70% accuracy")
    
    print("\n📊 MAXIMUM ACCURACY ACHIEVED:")
    print("-" * 50)
    sorted_max = sorted(performance_summary.items(), 
                       key=lambda x: x[1]['max_accuracy_mean'], reverse=True)
    
    for i, (method, perf) in enumerate(sorted_max, 1):
        label = method_labels.get(method, method.upper())
        print(f"{i}. {label:20} | {perf['max_accuracy_mean']:6.2f}%")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 50)
    
    if len(performance_summary) >= 2:
        best_final = max(performance_summary.items(), key=lambda x: x[1]['final_accuracy_mean'])
        best_label = method_labels.get(best_final[0], best_final[0].upper())
        
        print(f"🥇 Best final accuracy: {best_label} ({best_final[1]['final_accuracy_mean']:.2f}%)")
        
        if methods_reaching_70:
            fastest_70 = min(methods_reaching_70, key=lambda x: x[1]['data_to_70'])
            fastest_label = method_labels.get(fastest_70[0], fastest_70[0].upper())
            print(f"⚡ Fastest to 70%: {fastest_label} ({fastest_70[1]['data_to_70']:,.0f} data points)")
        
        # Check for surprising results
        sgd_performance = performance_summary.get('sgd')
        fomaml_performance = performance_summary.get('fomaml')
        
        if sgd_performance and fomaml_performance:
            sgd_final = sgd_performance['final_accuracy_mean']
            fomaml_final = fomaml_performance['final_accuracy_mean']
            
            if sgd_final > fomaml_final:
                advantage = sgd_final - fomaml_final
                print(f"🚨 SURPRISING: SGD outperforms FOMAML by {advantage:.2f}%!")
                print("   This challenges typical meta-learning assumptions")
    
    return performance_summary

def save_comprehensive_results(methods_data, performance_summary, output_dir):
    """Save comprehensive analysis results to JSON."""
    
    results = {
        'experiment_type': 'comprehensive_sample_efficiency_comparison',
        'methods_analyzed': list(methods_data.keys()),
        'performance_summary': performance_summary,
        'analysis_timestamp': str(np.datetime64('now')),
        'key_findings': {
            'best_final_method': max(performance_summary.items(), 
                                   key=lambda x: x[1]['final_accuracy_mean'])[0] if performance_summary else None,
            'methods_reaching_70': [method for method, perf in performance_summary.items() 
                                  if perf['reaches_70']],
            'total_methods_compared': len([m for m in methods_data.values() if m is not None])
        }
    }
    
    output_path = os.path.join(output_dir, 'comprehensive_sample_efficiency_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Comprehensive analysis saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare sample efficiency across all methods')
    
    # Input directories for each method
    parser.add_argument('--fomaml_dir', type=str, default='results/fomaml_analysis',
                       help='Directory containing FOMAML analysis results')
    parser.add_argument('--second_order_dir', type=str, default='results/second_order_maml_analysis', 
                       help='Directory containing Second-Order MAML analysis results')
    parser.add_argument('--sgd_dir', type=str, default='results/sgd_analysis',
                       help='Directory containing SGD analysis results')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results/comprehensive_comparison',
                       help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔬 COMPREHENSIVE SAMPLE EFFICIENCY COMPARISON")
    print("="*80)
    print(f"FOMAML results: {args.fomaml_dir}")
    print(f"Second-Order MAML results: {args.second_order_dir}")
    print(f"SGD results: {args.sgd_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Load results from each method
    methods_data = {}
    
    # Load FOMAML
    fomaml_data = load_parsed_results(args.fomaml_dir, 'fomaml')
    if fomaml_data:
        all_data_points, all_accuracies = extract_curves_from_data(fomaml_data, 'FOMAML')
        common_x, interpolated_curves = interpolate_curves(all_data_points, all_accuracies, 'FOMAML')
        statistics = calculate_statistics(common_x, interpolated_curves)
        methods_data['fomaml'] = {
            'raw_data': fomaml_data,
            'curves': (all_data_points, all_accuracies),
            'interpolated': (common_x, interpolated_curves),
            'statistics': statistics
        }
    
    # Load Second-Order MAML
    second_order_data = load_parsed_results(args.second_order_dir, 'second_order')
    if second_order_data:
        all_data_points, all_accuracies = extract_curves_from_data(second_order_data, 'Second-Order MAML')
        common_x, interpolated_curves = interpolate_curves(all_data_points, all_accuracies, 'Second-Order MAML')
        statistics = calculate_statistics(common_x, interpolated_curves)
        methods_data['second_order'] = {
            'raw_data': second_order_data,
            'curves': (all_data_points, all_accuracies),
            'interpolated': (common_x, interpolated_curves),
            'statistics': statistics
        }
    
    # Load SGD
    sgd_data = load_parsed_results(args.sgd_dir, 'sgd')
    if sgd_data:
        all_data_points, all_accuracies = extract_curves_from_data(sgd_data, 'SGD')
        common_x, interpolated_curves = interpolate_curves(all_data_points, all_accuracies, 'SGD')
        statistics = calculate_statistics(common_x, interpolated_curves)
        methods_data['sgd'] = {
            'raw_data': sgd_data,
            'curves': (all_data_points, all_accuracies),
            'interpolated': (common_x, interpolated_curves),
            'statistics': statistics
        }
    
    # Check if we have any valid data
    valid_methods = [name for name, data in methods_data.items() if data is not None]
    if not valid_methods:
        print("❌ No valid method data found! Please run the individual parsing scripts first.")
        return
    
    print(f"\n✅ Successfully loaded data for: {', '.join(valid_methods)}")
    
    # Create comparison plot
    create_comparison_plot(methods_data, args.output_dir)
    
    # Analyze performance
    performance_summary = analyze_method_performance(methods_data)
    
    # Save comprehensive results
    save_comprehensive_results(methods_data, performance_summary, args.output_dir)
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 