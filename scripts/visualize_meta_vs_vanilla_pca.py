import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict

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
        # It's safer to load onto CPU first, then move to device if needed
        checkpoint = torch.load(path, map_location='cpu')
        
        state_dict = checkpoint
        # Handle different checkpoint saving conventions
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint: # Another common convention
                 state_dict = checkpoint['model']

        # Clean keys from DataParallel saving
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        state_dict = new_state_dict
        
        # Load weights, ignoring missing keys (like batchnorm running stats)
        model.load_state_dict(state_dict, strict=False)
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
    single_tasks = [
        'original', 'filled', 'irregular', 'arrows', 'random_color', 
        'scrambled', 'wider_line', 'open', 'lines', 'regular'
    ]
    
    meta_base_path = '/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/seed_{seed}/best_model.pt'
    vanilla_base_path = '/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular/seed_{seed}/best_model.pt'
    single_task_base_path = '/scratch/gpfs/mg7411/samedifferent/results/ideal_datapar_exp_seed42_archconv6_della/single_task_runs/{task}/conv6/seed_42/best_model.pt'

    meta_paths = [meta_base_path.format(seed=s) for s in meta_seeds]
    vanilla_paths = [vanilla_base_path.format(seed=s) for s in vanilla_seeds]
    single_task_paths = [single_task_base_path.format(task=t) for t in single_tasks]

    # --- Load Weights ---
    all_weights = []
    labels = []
    annotations = []

    print("Loading meta-trained models...")
    for path, seed in zip(meta_paths, meta_seeds):
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Meta-Trained')
            annotations.append(f'seed {seed}')

    print("Loading vanilla models...")
    for path, seed in zip(vanilla_paths, vanilla_seeds):
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Vanilla')
            annotations.append(f'seed {seed}')
            
    print("Loading single-task models...")
    for path, task in zip(single_task_paths, single_tasks):
        weights = load_model_weights(path, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Single-Task')
            annotations.append(task)

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
    fig, ax = plt.subplots(figsize=(14, 12))
    
    colors = {'Meta-Trained': 'blue', 'Vanilla': 'red', 'Single-Task': 'green'}
    markers = {'Meta-Trained': '^', 'Vanilla': 'o', 'Single-Task': 's'}
    
    for label_type in np.unique(labels):
        indices = [i for i, l in enumerate(labels) if l == label_type]
        ax.scatter(principal_components[indices, 0], principal_components[indices, 1], 
                   color=colors[label_type], label=label_type, s=130, alpha=0.8, marker=markers[label_type])

    ax.set_title('PCA of Conv6 Weights: Meta vs. Vanilla vs. Single-Task', fontsize=20, pad=20)
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=16)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=16)
    ax.legend(title='Training Type', fontsize=14, title_fontsize=14)
    
    # Add annotations for each point
    for i, annotation in enumerate(annotations):
        ax.text(principal_components[i, 0], principal_components[i, 1] + 20, annotation, fontsize=9, ha='center')

    output_dir = 'visualizations/pca_analysis'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pca_meta_vs_vanilla_vs_singletask.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"PCA plot saved to {save_path}")

if __name__ == '__main__':
    main() 