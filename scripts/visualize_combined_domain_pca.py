import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add project root to allow importing the model architecture
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from meta_baseline.models.conv6lr import SameDifferentCNN

def load_model_weights(path, device):
    """Loads a model and returns its flattened weights."""
    model = SameDifferentCNN()
    try:
        # Load the checkpoint. Using weights_only=True is safer but may fail
        # with older checkpoints. We'll handle both.
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Determine the correct state dictionary
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel)
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
    """Main function to perform combined PCA and plot results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Define Paths & Seeds for all Four Groups ---
    # PB-Trained Models
    pb_meta_seeds = [42, 43, 44, 45, 46]
    pb_vanilla_seeds = [47, 48, 49, 50, 51]
    pb_meta_paths = [f'/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/seed_{s}/best_model.pt' for s in pb_meta_seeds]
    pb_vanilla_paths = [f'/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular/seed_{s}/best_model.pt' for s in pb_vanilla_seeds]

    # Naturalistic-Trained Models
    nat_meta_seeds = [111, 222, 333]
    nat_vanilla_seeds = [123, 555, 42, 999, 789]
    nat_meta_paths = [f"/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_meta/conv6lr/seed_{s}/conv6lr/seed_{s}/conv6lr_best.pth" for s in nat_meta_seeds]
    nat_vanilla_paths = [f"/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/conv6lr/seed_{s}/best_model.pt" for s in nat_vanilla_seeds]

    model_groups = {
        'PB Meta': pb_meta_paths,
        'PB Vanilla': pb_vanilla_paths,
        'Naturalistic Meta': nat_meta_paths,
        'Naturalistic Vanilla': nat_vanilla_paths
    }

    # --- Load All Weights ---
    all_weights = []
    labels = []

    for group_label, paths in model_groups.items():
        print(f"\nLoading {group_label} models...")
        for path in tqdm(paths, desc=f"Loading {group_label}"):
            weights = load_model_weights(path, device)
            if weights is not None:
                all_weights.append(weights)
                labels.append(group_label)

    if len(all_weights) < 2:
        print("Error: Not enough models loaded to perform PCA. Check model paths.")
        return
        
    print(f"\nTotal models loaded: {len(all_weights)}")

    # --- Perform PCA on Combined Weights ---
    print("Performing PCA on all weights...")
    all_weights_np = np.array(all_weights)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_weights_np)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # --- Plot Combined Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define colors for domain and markers for training method
    colors = {'PB': 'blue', 'Naturalistic': 'green'}
    markers = {'Meta': '^', 'Vanilla': 'o'}
    
    # Create a dummy scatter for the legend
    for domain in colors:
        for method in markers:
            ax.scatter([], [], c=colors[domain], marker=markers[method], label=f'{domain} {method}')
            
    for i, label in enumerate(labels):
        domain, method = label.split()
        ax.scatter(principal_components[i, 0], principal_components[i, 1], 
                   color=colors[domain], marker=markers[method], s=150, alpha=0.8)

    ax.set_title('Combined PCA of Conv6 Weights (PB vs. Naturalistic Domains)', fontsize=20, pad=20)
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=16)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=16)
    ax.legend(title='Model Type', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_dir = 'visualizations/pca_analysis'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pca_combined_domains_final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\nCombined PCA plot saved to {save_path}")

if __name__ == '__main__':
    main() 