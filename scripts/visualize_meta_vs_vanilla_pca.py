import os
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

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

def generate_initial_weights(seed, device):
    """Generate initial randomly initialized weights using the given seed."""
    # Set the random seed for reproducible initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create a new model with random initialization
    model = SameDifferentCNN()
    model.to(device)
    
    # Flatten all weights into a single vector
    weights = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    return np.concatenate(weights)

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
    
    print("Generating initial weights for vanilla models...")
    for seed in vanilla_seeds:
        weights = generate_initial_weights(seed, device)
        if weights is not None:
            all_weights.append(weights)
            labels.append('Initial')
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
    
    colors = {'Meta-Trained': 'blue', 'Vanilla': 'red', 'Single-Task': 'green', 'Initial': 'orange'}
    markers = {'Meta-Trained': '^', 'Vanilla': 'o', 'Single-Task': 's', 'Initial': 'x'}
    
    # Plot all data points on the main axes
    for label_type in np.unique(labels):
        indices = [i for i, l in enumerate(labels) if l == label_type]
        ax.scatter(principal_components[indices, 0], principal_components[indices, 1], 
                   color=colors[label_type], label=label_type, s=130, alpha=0.8, marker=markers[label_type])

    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=16)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=16)
    
    # --- Create a zoomed-in inset plot for the cluster ---
    # Manually place the inset in the upper left, with a balanced size
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper left')

    # Plot the meta-trained models in the inset
    meta_indices = [i for i, l in enumerate(labels) if l == 'Meta-Trained']
    if meta_indices:
        ax_inset.scatter(principal_components[meta_indices, 0], principal_components[meta_indices, 1],
                         color=colors['Meta-Trained'], marker=markers['Meta-Trained'], s=100, alpha=0.8)

    # Define the zoom area based on ALL meta-trained and single-task models
    cluster_indices = [i for i, l in enumerate(labels) if l in ['Meta-Trained', 'Single-Task']]
    if cluster_indices:
        # Get the PCs for all points that should be in the inset
        inset_pcs = principal_components[cluster_indices]
        
        x_min_zoom, x_max_zoom = inset_pcs[:, 0].min(), inset_pcs[:, 0].max()
        y_min_zoom, y_max_zoom = inset_pcs[:, 1].min(), inset_pcs[:, 1].max()

        # Add padding
        x_padding = (x_max_zoom - x_min_zoom) * 0.2
        y_padding = (y_max_zoom - y_min_zoom) * 0.2
        
        ax_inset.set_xlim(x_min_zoom - x_padding, x_max_zoom + x_padding)
        ax_inset.set_ylim(y_min_zoom - y_padding, y_max_zoom + y_padding)

    # Now, calculate a stronger jitter based on the inset's visible scale
    single_task_indices = [i for i, l in enumerate(labels) if l == 'Single-Task']
    if single_task_indices:
        single_task_pcs = principal_components[single_task_indices]
        
        inset_xlim = ax_inset.get_xlim()
        inset_ylim = ax_inset.get_ylim()
        x_range = inset_xlim[1] - inset_xlim[0]
        y_range = inset_ylim[1] - inset_ylim[0]
        
        # Jitter is now 5% of the inset's width/height for better visibility
        x_jitter = np.random.normal(0, x_range * 0.05, size=len(single_task_indices))
        y_jitter = np.random.normal(0, y_range * 0.05, size=len(single_task_indices))
        
        jittered_x = single_task_pcs[:, 0] + x_jitter
        jittered_y = single_task_pcs[:, 1] + y_jitter
        
        ax_inset.scatter(jittered_x, jittered_y,
                         color=colors['Single-Task'], marker=markers['Single-Task'], s=100, alpha=0.8)
        
        # Add annotations next to the jittered points
        single_task_annotations = [ann for i, ann in enumerate(annotations) if labels[i] == 'Single-Task']
        for i, txt in enumerate(single_task_annotations):
             ax_inset.text(jittered_x[i] + x_range * 0.015, jittered_y[i], txt, fontsize=8, ha='left', va='center')

    ax_inset.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # Draw a box showing the zoomed area on the main plot
    mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

    # Add the legend to the main plot, moved up
    ax.legend(title='Training Type', fontsize=14, title_fontsize=14, loc='center right')

    output_dir = 'visualizations/pca_analysis'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pca_with_inset_and_initial_weights.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"PCA plot with initial weights saved to {save_path}")

if __name__ == '__main__':
    main()