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

# Both model types now use the same, corrected architecture
from meta_baseline.models.conv6lr import SameDifferentCNN

def load_model_weights(path, device):
    """Loads a model and returns its flattened weights."""
    model = SameDifferentCNN()
    try:
        # Use weights_only=True for security and to avoid pickle errors
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # The new meta models save the state_dict directly
        # The vanilla models save it inside a 'model_state_dict' key
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
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

    # --- Define Paths & Seeds ---
    meta_seeds = [42, 43, 44, 45, 46]
    vanilla_seeds = [47, 48, 49, 50, 51]
    
    meta_base_path = '/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/seed_{seed}/best_model.pt'
    vanilla_base_path = '/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular/seed_{seed}/best_model.pt'

    meta_paths = [meta_base_path.format(seed=s) for s in meta_seeds]
    vanilla_paths = [vanilla_base_path.format(seed=s) for s in vanilla_seeds]

    # --- Load Weights ---
    all_weights = []
    labels = []
    all_seeds = meta_seeds + vanilla_seeds

    print("Loading meta-trained models...")
    for path in meta_paths:
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Meta-Trained')

    print("Loading vanilla models...")
    for path in vanilla_paths:
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Vanilla')

    if len(all_weights) < 2:
        print("Error: Not enough models loaded to perform PCA. Check model paths.")
        return
        
    print(f"Total models loaded: {len(all_weights)}")

    # --- Perform PCA ---
    print("Performing PCA...")
    all_weights = np.array(all_weights)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_weights)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # --- Plot Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {'Meta-Trained': 'blue', 'Vanilla': 'red'}
    markers = {'Meta-Trained': '^', 'Vanilla': 'o'}
    
    for label_type in np.unique(labels):
        indices = [i for i, l in enumerate(labels) if l == label_type]
        ax.scatter(principal_components[indices, 0], principal_components[indices, 1], 
                   color=colors[label_type], label=label_type, s=120, alpha=0.8, marker=markers[label_type])

    ax.set_title('PCA of Conv6 Weights: Meta-Trained vs. Vanilla-Trained', fontsize=18, pad=20)
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=14)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=14)
    ax.legend(title='Training Type', fontsize=12)
    
    # Add annotations for each point
    for i, _ in enumerate(all_weights):
        ax.text(principal_components[i, 0], principal_components[i, 1] + 0.05, f'seed {all_seeds[i]}', fontsize=9, ha='center')

    output_dir = 'visualizations/pca_analysis'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pca_meta_vs_vanilla_conv6_final.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"PCA plot saved to {save_path}")

if __name__ == '__main__':
    main() 