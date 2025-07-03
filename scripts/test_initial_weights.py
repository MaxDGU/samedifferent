#!/usr/bin/env python3
"""
Test script to verify initial weights generation works correctly.
"""

import os
import sys
import torch
import numpy as np

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from meta_baseline.models.conv6lr import SameDifferentCNN

def generate_initial_weights(seed, device):
    """Generate initial randomly initialized weights using the given seed."""
    try:
        print(f"Generating initial weights for seed {seed}...")
        # Set all random seeds for reproducible initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Create a new model with random initialization
        model = SameDifferentCNN()
        model.to(device)
        
        # Flatten all weights into a single vector
        weights = [p.data.cpu().numpy().flatten() for p in model.parameters()]
        flattened_weights = np.concatenate(weights)
        
        print(f"Successfully generated {len(flattened_weights)} weights")
        print(f"Weight stats: mean={flattened_weights.mean():.6f}, std={flattened_weights.std():.6f}")
        print(f"Min/Max: {flattened_weights.min():.6f}/{flattened_weights.max():.6f}")
        return flattened_weights
        
    except Exception as e:
        print(f"Error generating initial weights for seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    vanilla_seeds = [47, 48, 49, 50, 51]
    
    all_weights = []
    for seed in vanilla_seeds:
        weights = generate_initial_weights(seed, device)
        if weights is not None:
            all_weights.append(weights)
        print("-" * 50)
    
    if len(all_weights) > 1:
        print(f"\nGenerated {len(all_weights)} weight sets")
        
        # Check if weights are different between seeds
        for i in range(len(all_weights)):
            for j in range(i+1, len(all_weights)):
                diff = np.mean(np.abs(all_weights[i] - all_weights[j]))
                print(f"Mean abs difference between seed {vanilla_seeds[i]} and {vanilla_seeds[j]}: {diff:.6f}")
        
        # Test PCA on these weights
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(np.array(all_weights))
        print(f"\nPCA results:")
        print(f"PC1 range: {principal_components[:, 0].min():.3f} to {principal_components[:, 0].max():.3f}")
        print(f"PC2 range: {principal_components[:, 1].min():.3f} to {principal_components[:, 1].max():.3f}")
        print("Initial weights generation test PASSED")
    else:
        print("Initial weights generation test FAILED")

if __name__ == '__main__':
    main() 