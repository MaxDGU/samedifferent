import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Assuming the new vanilla models use the same architecture class as the meta-trained ones.
# We will use the meta_baseline models for loading.
from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6lrCNN

def load_model_weights(path, device):
    """Loads a model and returns its flattened weights."""
    model = Conv6lrCNN()
    try:
        checkpoint = torch.load(path, map_location=device)
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint and 'state_dict' in checkpoint['model']:
            state_dict = checkpoint['model']['state_dict']
        else:
            state_dict = checkpoint
            
        # The meta-trained models might have a 'module.' prefix from DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(device)
        
        # Flatten all weights into a single vector
        weights = [p.data.cpu().numpy().flatten() for p in model.parameters()]
        return np.concatenate(weights)
        
    except FileNotFoundError:
        print(f"Warning: Could not find model at {path}. Skipping.")
        return None
    except Exception as e:
        print(f"Error loading model at {path}: {e}. Skipping.")
        return None

def main():
    """Main function to perform PCA and plot the results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Define Paths ---
    # Using the original 5 seeds for comparison
    seeds = [47, 48, 49, 50, 51]
    seeds_meta = [3,4,5,6,7]
    
    # Paths for the newly trained vanilla models
    vanilla_base_path = '/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular/seed_{seed}/best_model.pt'
    
    # Paths for the meta-trained (FOMAML) models - THIS IS AN EDUCATED GUESS
    meta_base_path = '/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6/model_{seed}_pretesting.pt'

    vanilla_paths = [vanilla_base_path.format(seed=s) for s in seeds]
    meta_paths = [meta_base_path.format(seed=s) for s in seeds_meta]

    # --- Load Weights ---
    all_weights = []
    labels = []

    print("Loading vanilla models...")
    for path in vanilla_paths:
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Vanilla')

    print("Loading meta-trained models...")
    for path in meta_paths:
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Meta-Trained (FOMAML)')

    if len(all_weights) < 2:
        print("Error: Not enough models loaded to perform PCA. Check model paths.")
        return

    # --- Perform PCA ---
    print("Performing PCA...")
    all_weights = np.array(all_weights)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_weights)

    # --- Plot Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        indices = [j for j, l in enumerate(labels) if l == label]
        ax.scatter(principal_components[indices, 0], principal_components[indices, 1], 
                   color=colors[i], label=label, s=100, alpha=0.7)

    ax.set_title('PCA of Conv6 Weights: Meta-Trained vs. Vanilla-Trained', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(title='Training Type', fontsize=10)
    
    # Add annotations for each point
    for i, label in enumerate(labels):
        seed_index = i % len(seeds)
        ax.text(principal_components[i, 0], principal_components[i, 1] + 0.05, f'seed {seeds[seed_index]}', fontsize=8, ha='center')

    output_dir = 'visualizations/pca_analysis'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pca_meta_vs_vanilla_conv6.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"PCA plot saved to {save_path}")

if __name__ == '__main__':
    main() 