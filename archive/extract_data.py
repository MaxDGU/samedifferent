import json
import numpy as np
import pandas as pd
from collections import defaultdict

# Define the task list
tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
         'random_color', 'arrows', 'irregular', 'filled', 'original']

# VANILLA LEARNING EXTRACTION
print("\nVANILLA LEARNING RESULTS:")
print("-" * 60)
print(f"{'Task':<15} {'Conv2 Acc':<10} {'Conv2 Std':<10} {'Conv4 Acc':<10} {'Conv4 Std':<10} {'Conv6 Acc':<10} {'Conv6 Std':<10}")
print("-" * 60)

try:
    with open('all_stats.json', 'r') as f:
        vanilla_data = json.load(f)
    
    for task in tasks:
        conv2_acc = vanilla_data["per_architecture"]["conv2"]["per_task"][task]["mean_accuracy"] * 100
        conv2_std = vanilla_data["per_architecture"]["conv2"]["per_task"][task]["std_accuracy"] * 100
        conv4_acc = vanilla_data["per_architecture"]["conv4"]["per_task"][task]["mean_accuracy"] * 100
        conv4_std = vanilla_data["per_architecture"]["conv4"]["per_task"][task]["std_accuracy"] * 100
        conv6_acc = vanilla_data["per_architecture"]["conv6"]["per_task"][task]["mean_accuracy"] * 100
        conv6_std = vanilla_data["per_architecture"]["conv6"]["per_task"][task]["std_accuracy"] * 100
        
        print(f"{task:<15} {conv2_acc:<10.1f} {conv2_std:<10.1f} {conv4_acc:<10.1f} {conv4_std:<10.1f} {conv6_acc:<10.1f} {conv6_std:<10.1f}")
except Exception as e:
    print(f"Error extracting vanilla learning data: {e}")

# METALEARNING EXTRACTION
print("\nMETALEARNING RESULTS:")
print("-" * 60)
print(f"{'Task':<15} {'Conv2 Acc':<10} {'Conv2 Std':<10} {'Conv4 Acc':<10} {'Conv4 Std':<10} {'Conv6 Acc':<10} {'Conv6 Std':<10}")
print("-" * 60)

try:
    # Conv2 (hardcoded values from extract_and_plot.py)
    conv2_meta = {
        'regular': {'mean': 52.2, 'std': 5.0},
        'lines': {'mean': 78.6, 'std': 8.0},
        'open': {'mean': 48.8, 'std': 6.0},
        'wider_line': {'mean': 52.2, 'std': 5.0},
        'scrambled': {'mean': 82.0, 'std': 7.0},
        'random_color': {'mean': 49.8, 'std': 4.0},
        'arrows': {'mean': 48.6, 'std': 6.0},
        'irregular': {'mean': 45.4, 'std': 5.0},
        'filled': {'mean': 54.0, 'std': 6.0},
        'original': {'mean': 51.0, 'std': 5.0}
    }
    
    # Conv4 (hardcoded values from extract_and_plot.py)
    conv4_data = {
        'regular': [0.9000, 0.9600, 0.8600, 0.9400, 0.8600, 0.6000, 0.9200, 0.8600, 0.9000, 0.7800],
        'lines': [1.0000, 1.0000, 1.0000, 1.0000, 0.9800, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000],
        'open': [0.9400, 0.8800, 0.9200, 0.9000, 0.8400, 0.5600, 0.8600, 0.8400, 0.9200, 0.8600],
        'wider_line': [0.9000, 0.9000, 0.8600, 0.8600, 0.8400, 0.5400, 0.9000, 0.8200, 0.9000, 0.9000],
        'scrambled': [1.0000, 1.0000, 0.9800, 1.0000, 1.0000, 0.5000, 0.9800, 1.0000, 0.9800, 1.0000],
        'random_color': [0.8933, 0.9333, 0.8533, 0.8800, 0.8667, 0.6933, 0.8933, 0.7733, 0.9467, 0.8933],
        'arrows': [0.6600, 0.6800, 0.8200, 0.8200, 0.7600, 0.4600, 0.5800, 0.7200, 0.5800, 0.8200],
        'irregular': [0.9400, 0.9000, 0.8200, 0.8400, 0.8400, 0.4800, 0.8800, 0.8600, 0.9200, 0.8200],
        'filled': [0.8000, 0.8000, 0.8600, 0.9000, 1.0000, 0.4800, 0.9200, 0.8600, 0.9400, 0.8600],
        'original': [0.8400, 0.8600, 0.8800, 0.9000, 0.8800, 0.5200, 0.8800, 0.8400, 0.9200, 0.8800]
    }
    
    conv4_meta = {}
    for task in tasks:
        values = np.array(conv4_data[task])
        conv4_meta[task] = {
            'mean': values.mean() * 100,
            'std': values.std() * 100
        }
    
    # Conv6 (from pb_results_summary.csv)
    try:
        with open('pb_results_summary.csv', 'r') as f:
            csv_data = pd.read_csv(f)
        
        # Initialize with default values in case CSV data is missing
        conv6_meta = {task: {'mean': 0.0, 'std': 0.0} for task in tasks}
        
        for task in tasks:
            mean_key = f'conv6_{task}_mean'
            std_key = f'conv6_{task}_std'
            if mean_key in csv_data and std_key in csv_data:
                conv6_meta[task]['mean'] = float(csv_data[mean_key].iloc[0])
                conv6_meta[task]['std'] = float(csv_data[std_key].iloc[0])
    except Exception as e:
        print(f"Warning: Could not load conv6 meta results from CSV: {e}")
        # Use default values if CSV load fails
        conv6_meta = {task: {'mean': 0.0, 'std': 0.0} for task in tasks}
    
    # Print the metalearning results
    for task in tasks:
        conv2_acc = conv2_meta[task]['mean']
        conv2_std = conv2_meta[task]['std']
        conv4_acc = conv4_meta[task]['mean']
        conv4_std = conv4_meta[task]['std']
        conv6_acc = conv6_meta[task]['mean']
        conv6_std = conv6_meta[task]['std']
        
        print(f"{task:<15} {conv2_acc:<10.1f} {conv2_std:<10.1f} {conv4_acc:<10.1f} {conv4_std:<10.1f} {conv6_acc:<10.1f} {conv6_std:<10.1f}")
except Exception as e:
    print(f"Error extracting metalearning data: {e}") 