#!/usr/bin/env python3
"""
Diagnostic script to debug PCA weight visualization issues.
This script provides detailed output about all weight sets and their PCA coordinates.
"""

import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import sys
from collections import OrderedDict

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from meta_baseline.models.conv6lr import SameDifferentCNN

def load_model_weights(path, device):
    """Loads a model and returns its flattened weights."""
    model = SameDifferentCNN()
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                 state_dict = checkpoint['model']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        state_dict = new_state_dict
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        
        weights = [p.data.cpu().numpy().flatten() for p in model.parameters()]
        return np.concatenate(weights)
        
    except Exception as e:
        print(f"Error loading model at {path}: {e}")
        return None

def generate_initial_weights(seed, device):
    """Generate initial randomly initialized weights using the given seed."""
    try:
        print(f"  Generating initial weights for seed {seed}...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        model = SameDifferentCNN()
        model.to(device)
        
        weights = [p.data.cpu().numpy().flatten() for p in model.parameters()]
        flattened_weights = np.concatenate(weights)
        
        print(f"  Generated {len(flattened_weights)} weights, mean={flattened_weights.mean():.6f}, std={flattened_weights.std():.6f}")
        return flattened_weights
        
    except Exception as e:
        print(f"  Error generating initial weights for seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define paths and seeds
    meta_seeds = [42, 43, 44, 45, 46]
    vanilla_seeds = [47, 48, 49, 50, 51]
    single_tasks = ['original', 'filled', 'irregular', 'arrows', 'random_color', 
                   'scrambled', 'wider_line', 'open', 'lines', 'regular']
    
    meta_base_path = '/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/seed_{seed}/best_model.pt'
    vanilla_base_path = '/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular/seed_{seed}/best_model.pt'
    single_task_base_path = '/scratch/gpfs/mg7411/samedifferent/results/ideal_datapar_exp_seed42_archconv6_della/single_task_runs/{task}/conv6/seed_42/best_model.pth'

    all_weights = []
    labels = []
    annotations = []
    weight_stats = []

    # Load meta-trained models
    print("\n=== LOADING META-TRAINED MODELS ===")
    for seed in meta_seeds:
        path = meta_base_path.format(seed=seed)
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Meta-Trained')
            annotations.append(f'seed {seed}')
            weight_stats.append(f"Meta-{seed}: {len(weights)} weights, mean={weights.mean():.6f}, std={weights.std():.6f}")
            print(f"  ✓ Loaded meta model seed {seed}: {len(weights)} weights")
        else:
            print(f"  ✗ Failed to load meta model seed {seed}")

    # Load vanilla models
    print("\n=== LOADING VANILLA MODELS ===")
    for seed in vanilla_seeds:
        path = vanilla_base_path.format(seed=seed)
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Vanilla')
            annotations.append(f'seed {seed}')
            weight_stats.append(f"Vanilla-{seed}: {len(weights)} weights, mean={weights.mean():.6f}, std={weights.std():.6f}")
            print(f"  ✓ Loaded vanilla model seed {seed}: {len(weights)} weights")
        else:
            print(f"  ✗ Failed to load vanilla model seed {seed}")

    # Generate initial weights
    print("\n=== GENERATING INITIAL WEIGHTS ===")
    initial_count = 0
    for seed in vanilla_seeds:
        weights = generate_initial_weights(seed, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Initial')
            annotations.append(f'seed {seed}')
            weight_stats.append(f"Initial-{seed}: {len(weights)} weights, mean={weights.mean():.6f}, std={weights.std():.6f}")
            initial_count += 1
            print(f"  ✓ Generated initial weights for seed {seed}: {len(weights)} weights")
        else:
            print(f"  ✗ Failed to generate initial weights for seed {seed}")
    
    print(f"\nSuccessfully generated {initial_count}/5 initial weight sets")

    # Load single-task models
    print("\n=== LOADING SINGLE-TASK MODELS ===")
    for task in single_tasks:
        path = single_task_base_path.format(task=task)
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Single-Task')
            annotations.append(task)
            weight_stats.append(f"SingleTask-{task}: {len(weights)} weights, mean={weights.mean():.6f}, std={weights.std():.6f}")
            print(f"  ✓ Loaded single-task model {task}: {len(weights)} weights")
        else:
            print(f"  ✗ Failed to load single-task model {task}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total models loaded: {len(all_weights)}")
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    for label, count in label_counts.items():
        print(f"  {label}: {count} models")

    if len(all_weights) < 2:
        print("ERROR: Not enough models loaded for PCA")
        return

    # Perform PCA
    print(f"\n=== PERFORMING PCA ===")
    all_weights_array = np.array(all_weights)
    print(f"Weight matrix shape: {all_weights_array.shape}")
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_weights_array)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Detailed PCA results
    print(f"\n=== PCA COORDINATES ===")
    for i, (label, annotation, pc) in enumerate(zip(labels, annotations, principal_components)):
        print(f"{i:2d}. {label:12s} {annotation:12s}: PC1={pc[0]:8.3f}, PC2={pc[1]:8.3f}")

    # Check for clusters
    print(f"\n=== COORDINATE RANGES ===")
    for label_type in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label_type]
        if indices:
            pcs = principal_components[indices]
            pc1_range = (pcs[:, 0].min(), pcs[:, 0].max())
            pc2_range = (pcs[:, 1].min(), pcs[:, 1].max())
            print(f"{label_type:12s}: PC1=[{pc1_range[0]:8.3f}, {pc1_range[1]:8.3f}], PC2=[{pc2_range[0]:8.3f}, {pc2_range[1]:8.3f}]")

    # Check if Initial weights are clustered with other points
    if 'Initial' in labels:
        initial_indices = [i for i, l in enumerate(labels) if l == 'Initial']
        initial_pcs = principal_components[initial_indices]
        
        print(f"\n=== INITIAL WEIGHTS ANALYSIS ===")
        print(f"Initial weights PC coordinates:")
        for i, idx in enumerate(initial_indices):
            print(f"  Initial seed {annotations[idx]}: PC1={principal_components[idx][0]:8.3f}, PC2={principal_components[idx][1]:8.3f}")
        
        # Check distances to other points
        min_distances = []
        for init_pc in initial_pcs:
            distances = []
            for i, pc in enumerate(principal_components):
                if i not in initial_indices:  # Don't compare to other initial weights
                    dist = np.sqrt((init_pc[0] - pc[0])**2 + (init_pc[1] - pc[1])**2)
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))
        
        if min_distances:
            print(f"Minimum distances from initial weights to other points: {min_distances}")
            print(f"Average minimum distance: {np.mean(min_distances):.3f}")
        
        # Check if initial weights are visible (not too close to other clusters)
        pc1_span = principal_components[:, 0].max() - principal_components[:, 0].min()
        pc2_span = principal_components[:, 1].max() - principal_components[:, 1].min()
        print(f"Total PC1 span: {pc1_span:.3f}, Total PC2 span: {pc2_span:.3f}")
        
        if min_distances and np.mean(min_distances) < 0.01 * max(pc1_span, pc2_span):
            print("WARNING: Initial weights may be too close to other points to be visible!")
        else:
            print("Initial weights should be visible in the plot.")

if __name__ == '__main__':
    main() 