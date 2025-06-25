import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import os

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the model classes to load the weights correctly
from meta_baseline.models.conv2lr import SameDifferentCNN as Conv2lrCNN
from meta_baseline.models.conv4lr import SameDifferentCNN as Conv4lrCNN
from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6lrCNN

# --- Configuration ---
SEEDS = [42, 123, 555, 789, 999, 111, 222, 333] # All 8 seeds
OUTPUT_DIR = Path('visualizations/domain_comparison_pca')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Define a mapping from architecture string to model class and paths
MODEL_CONFIG = {
    'conv2lr': {
        'class': Conv2lrCNN,
        'pb_path_template': 'results/pb_retrained_conv2lr/seed_{seed}/best_model.pt',
        'nat_path_template': 'logs_naturalistic_meta/conv2lr/seed_{seed}/best_model.pt',
    },
    'conv4lr': {
        'class': Conv4lrCNN,
        'pb_path_template': 'results/pb_retrained_conv4lr/seed_{seed}/best_model.pt',
        'nat_path_template': 'logs_naturalistic_meta/conv4lr/seed_{seed}/best_model.pt',
    },
    'conv6lr': {
        'class': Conv6lrCNN,
        'pb_path_template': 'results/pb_retrained_conv6lr/seed_{seed}/best_model.pt',
        'nat_path_template': 'logs_naturalistic_meta/conv6lr/seed_{seed}/best_model.pt',
    }
}

def load_and_flatten_weights(model_path, model_class):
    """Loads a model state_dict and returns the flattened weights."""
    if not model_path.exists():
        # print(f"Warning: Weight file not found at {model_path}")
        return None
    
    try:
        # Instantiate model to ensure state_dict keys match
        model = model_class()
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # The weights are often stored under 'model_state_dict'
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else: # Handle cases where the whole object is the state_dict
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        
        # Flatten all parameters into a single vector
        flat_weights = torch.cat([p.detach().flatten() for p in model.parameters()]).numpy()
        return flat_weights
        
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def main():
    """
    Main function to generate PCA plots for each architecture,
    comparing meta-training on PB vs. Naturalistic datasets.
    """
    print("--- Generating PCA plots for Meta-Training Domain Comparison ---")

    for arch, config in MODEL_CONFIG.items():
        print(f"\nProcessing architecture: {arch}...")
        
        all_weights = []
        labels = []
        
        # --- Load PB Weights ---
        for seed in SEEDS:
            path = Path(config['pb_path_template'].format(seed=seed))
            weights = load_and_flatten_weights(path, config['class'])
            if weights is not None:
                all_weights.append(weights)
                labels.append('PB')

        # --- Load Naturalistic Weights ---
        for seed in SEEDS:
            path = Path(config['nat_path_template'].format(seed=seed))
            weights = load_and_flatten_weights(path, config['class'])
            if weights is not None:
                all_weights.append(weights)
                labels.append('Naturalistic')
        
        if len(all_weights) < 2:
            print(f"Skipping {arch}: Not enough models found to perform PCA.")
            continue

        # --- Perform PCA ---
        weights_matrix = np.array(all_weights)
        pca = PCA(n_components=2)
        transformed_weights = pca.fit_transform(weights_matrix)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"PCA Explained Variance for {arch}: {explained_variance[0]:.2f} (PC1), {explained_variance[1]:.2f} (PC2)")

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = {'PB': '#1f77b4', 'Naturalistic': '#ff7f0e'}
        markers = {'PB': 'o', 'Naturalistic': 'X'}
        
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            ax.scatter(transformed_weights[indices, 0], 
                       transformed_weights[indices, 1],
                       c=colors[label],
                       marker=markers[label],
                       label=f'Meta-Trained on {label}',
                       s=100,
                       alpha=0.8,
                       edgecolors='k')

        ax.set_title(f'PCA of Final Model Weights for {arch.upper()}', fontsize=16)
        ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]*100:.1f}%)', fontsize=12)
        ax.legend(fontsize=12)
        
        # Save the plot
        output_path = OUTPUT_DIR / f'pca_domain_comparison_{arch}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved plot to {output_path}")

if __name__ == '__main__':
    main()
