import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

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
    single_task_base_path = '/scratch/gpfs/mg7411/samedifferent/results/ideal_datapar_exp_seed42_archconv6_della/single_task_runs/{task}/conv6/seed_42/best_model.pth'

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

    # Add annotations for each point
    for i, annotation in enumerate(annotations):
        # Only add annotations for the more spread out 'Vanilla' points on the main plot
        if labels[i] == 'Vanilla':
            ax.text(principal_components[i, 0], principal_components[i, 1] + 50, annotation, fontsize=9, ha='center')

    # --- Create a zoomed-in inset plot for the cluster ---
    # Define the region of interest based on the cluster
    cluster_indices = [i for i, l in enumerate(labels) if l in ['Meta-Trained', 'Single-Task']]
    if cluster_indices:
        x_min_zoom = principal_components[cluster_indices, 0].min()
        x_max_zoom = principal_components[cluster_indices, 0].max()
        y_min_zoom = principal_components[cluster_indices, 1].min()
        y_max_zoom = principal_components[cluster_indices, 1].max()

        # Add some padding
        x_padding = (x_max_zoom - x_min_zoom) * 0.1
        y_padding = (y_max_zoom - y_min_zoom) * 0.1
        x_min_zoom -= x_padding
        x_max_zoom += x_padding
        y_min_zoom -= y_padding
        y_max_zoom += y_padding

        # Create the inset axes
        # The location (4) is 'lower right'. You can change this to 1, 2, or 3.
        ax_inset = zoomed_inset_axes(ax, zoom=5, loc='lower left', borderpad=2)

        # Plot the clustered data on the inset axes
        for label_type in ['Meta-Trained', 'Single-Task']:
            indices = [i for i, l in enumerate(labels) if l == label_type]
            ax_inset.scatter(principal_components[indices, 0], principal_components[indices, 1],
                             color=colors[label_type], marker=markers[label_type], s=130, alpha=0.8)

        # Annotate points within the inset
        for i in cluster_indices:
            ax_inset.text(principal_components[i, 0], principal_components[i, 1] + 5, annotations[i],
                          fontsize=9, ha='center')

        # Set the limits and hide ticks for the inset
        ax_inset.set_xlim(x_min_zoom, x_max_zoom)
        ax_inset.set_ylim(y_min_zoom, y_max_zoom)
        ax_inset.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        # Draw a box showing the zoomed area and connection lines
        mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_title('PCA of Conv6 Weights: Meta vs. Vanilla vs. Single-Task', fontsize=20, pad=20)
    
    output_dir = 'visualizations/pca_analysis'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pca_with_inset.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"PCA plot with inset saved to {save_path}")

if __name__ == '__main__':
    main() 