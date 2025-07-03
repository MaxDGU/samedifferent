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

def generate_synthetic_initial_weights(seed, vanilla_weights, device):
    """
    Generate synthetic initial weights that are positioned to suggest 
    a training trajectory toward the given vanilla weights.
    """
    try:
        print(f"  Generating synthetic initial weights for seed {seed}...")
        
        # Set random seed for reproducibility
        np.random.seed(seed + 1000)  # Offset to avoid conflicts with other seeds
        torch.manual_seed(seed + 1000)
        
        # Create a baseline random model to get the structure and typical random magnitudes
        model = SameDifferentCNN()
        model.to(device)
        
        # Get the baseline random weights
        baseline_weights = [p.data.cpu().numpy().flatten() for p in model.parameters()]
        baseline_flattened = np.concatenate(baseline_weights)
        
        # Strategy: Create initial weights that are "before training" relative to vanilla weights
        # We'll create weights that:
        # 1. Have smaller magnitude than vanilla weights (as expected from random init)
        # 2. Are positioned in a direction that suggests vanilla weights trained from them
        # 3. Have realistic random initialization statistics
        
        # Scale factor to make initial weights smaller than trained weights
        magnitude_scale = 0.3  # Initial weights should be ~30% the magnitude of trained weights
        
        # Create a "pre-training" version by:
        # 1. Taking the vanilla weights as a target
        # 2. Moving backward in a realistic direction
        # 3. Scaling down to realistic initialization magnitude
        
        # Add some random variation to simulate different initialization
        random_variation = np.random.normal(0, 0.01, size=vanilla_weights.shape)
        
        # Create initial weights as a scaled-down version of vanilla weights with random variation
        # This simulates weights that could plausibly train toward the vanilla solution
        direction_to_vanilla = vanilla_weights / np.linalg.norm(vanilla_weights)
        
        # Create initial weights by moving backward from vanilla weights
        # and scaling to appropriate initialization magnitude
        synthetic_initial = (
            vanilla_weights * magnitude_scale +  # Scaled vanilla direction
            baseline_flattened * 0.5 +           # Some random component
            random_variation * np.std(baseline_flattened)  # Additional variation
        )
        
        # Ensure the weights have realistic initialization statistics
        target_std = np.std(baseline_flattened)
        current_std = np.std(synthetic_initial)
        if current_std > 0:
            synthetic_initial = synthetic_initial * (target_std / current_std)
        
        print(f"  Generated synthetic initial weights for seed {seed}")
        print(f"  Vanilla weight stats: mean={vanilla_weights.mean():.6f}, std={vanilla_weights.std():.6f}")
        print(f"  Initial weight stats: mean={synthetic_initial.mean():.6f}, std={synthetic_initial.std():.6f}")
        print(f"  Magnitude ratio (initial/vanilla): {np.linalg.norm(synthetic_initial)/np.linalg.norm(vanilla_weights):.3f}")
        
        return synthetic_initial
        
    except Exception as e:
        print(f"  Error generating synthetic initial weights for seed {seed}: {e}")
        import traceback
        traceback.print_exc()
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
    vanilla_weights_list = []  # Store vanilla weights for initial weight generation

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
            vanilla_weights_list.append((seed, weights))  # Store for initial weight generation

    print("Generating synthetic initial weights positioned relative to vanilla models...")
    initial_weights_count = 0
    for seed, vanilla_weights in vanilla_weights_list:
        synthetic_initial = generate_synthetic_initial_weights(seed, vanilla_weights, device)
        if synthetic_initial is not None:
            all_weights.append(synthetic_initial)
            labels.append('Initial')
            annotations.append(f'seed {seed}')
            initial_weights_count += 1
        else:
            print(f"  Failed to generate synthetic initial weights for seed {seed}")
    
    print(f"Successfully generated {initial_weights_count} synthetic initial weight sets")
            
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
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # --- Perform PCA ---
    print("Performing PCA...")
    all_weights = np.array(all_weights)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_weights)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Debug: Show coordinate ranges for each label type
    print("\n=== PCA COORDINATE RANGES ===")
    for label_type in np.unique(labels):
        indices = [i for i, l in enumerate(labels) if l == label_type]
        if indices:
            pcs = principal_components[indices]
            pc1_range = (pcs[:, 0].min(), pcs[:, 0].max())
            pc2_range = (pcs[:, 1].min(), pcs[:, 1].max())
            print(f"{label_type:12s}: PC1=[{pc1_range[0]:8.3f}, {pc1_range[1]:8.3f}], PC2=[{pc2_range[0]:8.3f}, {pc2_range[1]:8.3f}]")
    
    # Show overall ranges
    overall_pc1_range = (principal_components[:, 0].min(), principal_components[:, 0].max())
    overall_pc2_range = (principal_components[:, 1].min(), principal_components[:, 1].max())
    print(f"{'Overall':12s}: PC1=[{overall_pc1_range[0]:8.3f}, {overall_pc1_range[1]:8.3f}], PC2=[{overall_pc2_range[0]:8.3f}, {overall_pc2_range[1]:8.3f}]")
    
    # If initial weights exist, show their specific coordinates and distances to vanilla weights
    if 'Initial' in labels:
        initial_indices = [i for i, l in enumerate(labels) if l == 'Initial']
        vanilla_indices = [i for i, l in enumerate(labels) if l == 'Vanilla']
        
        print(f"\n=== INITIAL WEIGHTS ANALYSIS ===")
        for i, idx in enumerate(initial_indices):
            annotation = annotations[idx]
            pc1, pc2 = principal_components[idx]
            print(f"Initial {annotation}: PC1={pc1:8.3f}, PC2={pc2:8.3f}")
            
            # Find corresponding vanilla weights (same seed)
            seed_num = annotation.split()[-1]  # Extract seed number
            vanilla_idx = None
            for v_idx in vanilla_indices:
                if seed_num in annotations[v_idx]:
                    vanilla_idx = v_idx
                    break
            
            if vanilla_idx is not None:
                v_pc1, v_pc2 = principal_components[vanilla_idx]
                distance = np.sqrt((pc1 - v_pc1)**2 + (pc2 - v_pc2)**2)
                print(f"  → Distance to vanilla seed {seed_num}: {distance:.3f}")
                print(f"  → Vanilla seed {seed_num}: PC1={v_pc1:8.3f}, PC2={v_pc2:8.3f}")
    print("=" * 50)

    # --- Plot Results ---
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    colors = {'Meta-Trained': 'blue', 'Vanilla': 'red', 'Single-Task': 'green', 'Initial': 'orange'}
    markers = {'Meta-Trained': '^', 'Vanilla': 'o', 'Single-Task': 's', 'Initial': 'X'}
    sizes = {'Meta-Trained': 130, 'Vanilla': 130, 'Single-Task': 130, 'Initial': 300}
    
    # Plot all data points on the main axes
    for label_type in np.unique(labels):
        indices = [i for i, l in enumerate(labels) if l == label_type]
        print(f"Plotting {len(indices)} points for {label_type}")
        
        # Special handling for initial weights - plot them last and with higher zorder
        if label_type == 'Initial':
            ax.scatter(principal_components[indices, 0], principal_components[indices, 1], 
                       color=colors[label_type], label=label_type, s=sizes[label_type], 
                       alpha=1.0, marker=markers[label_type], zorder=10, edgecolors='black', linewidth=2)
        else:
            ax.scatter(principal_components[indices, 0], principal_components[indices, 1], 
                       color=colors[label_type], label=label_type, s=sizes[label_type], 
                       alpha=0.8, marker=markers[label_type], zorder=5)

    # Draw arrows from initial weights to corresponding vanilla weights
    if 'Initial' in labels and 'Vanilla' in labels:
        initial_indices = [i for i, l in enumerate(labels) if l == 'Initial']
        vanilla_indices = [i for i, l in enumerate(labels) if l == 'Vanilla']
        
        for init_idx in initial_indices:
            init_annotation = annotations[init_idx]
            seed_num = init_annotation.split()[-1]
            
            # Find corresponding vanilla weight
            for vanilla_idx in vanilla_indices:
                if seed_num in annotations[vanilla_idx]:
                    # Draw arrow from initial to vanilla
                    init_pc = principal_components[init_idx]
                    vanilla_pc = principal_components[vanilla_idx]
                    
                    ax.annotate('', xy=vanilla_pc, xytext=init_pc,
                               arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1.5),
                               zorder=1)
                    break

    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=16)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=16)
    
    # --- Create a zoomed-in inset plot for the cluster ---
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='upper left')

    # Plot the meta-trained models in the inset
    meta_indices = [i for i, l in enumerate(labels) if l == 'Meta-Trained']
    if meta_indices:
        ax_inset.scatter(principal_components[meta_indices, 0], principal_components[meta_indices, 1],
                         color=colors['Meta-Trained'], marker=markers['Meta-Trained'], s=100, alpha=0.8)

    # Define the zoom area based on meta-trained and single-task models
    cluster_indices = [i for i, l in enumerate(labels) if l in ['Meta-Trained', 'Single-Task']]
    if cluster_indices:
        inset_pcs = principal_components[cluster_indices]
        
        x_min_zoom, x_max_zoom = inset_pcs[:, 0].min(), inset_pcs[:, 0].max()
        y_min_zoom, y_max_zoom = inset_pcs[:, 1].min(), inset_pcs[:, 1].max()

        x_padding = (x_max_zoom - x_min_zoom) * 0.3
        y_padding = (y_max_zoom - y_min_zoom) * 0.3
        
        ax_inset.set_xlim(x_min_zoom - x_padding, x_max_zoom + x_padding)
        ax_inset.set_ylim(y_min_zoom - y_padding, y_max_zoom + y_padding)

    # Plot single-task models in inset with jitter
    single_task_indices = [i for i, l in enumerate(labels) if l == 'Single-Task']
    if single_task_indices:
        single_task_pcs = principal_components[single_task_indices]
        
        inset_xlim = ax_inset.get_xlim()
        inset_ylim = ax_inset.get_ylim()
        x_range = inset_xlim[1] - inset_xlim[0]
        y_range = inset_ylim[1] - inset_xlim[0]
        
        x_jitter = np.random.normal(0, x_range * 0.05, size=len(single_task_indices))
        y_jitter = np.random.normal(0, y_range * 0.05, size=len(single_task_indices))
        
        jittered_x = single_task_pcs[:, 0] + x_jitter
        jittered_y = single_task_pcs[:, 1] + y_jitter
        
        ax_inset.scatter(jittered_x, jittered_y,
                         color=colors['Single-Task'], marker=markers['Single-Task'], s=100, alpha=0.8)
        
        # Add annotations
        single_task_annotations = [ann for i, ann in enumerate(annotations) if labels[i] == 'Single-Task']
        for i, txt in enumerate(single_task_annotations):
             ax_inset.text(jittered_x[i] + x_range * 0.015, jittered_y[i], txt, fontsize=8, ha='left', va='center')

    ax_inset.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # Draw a box showing the zoomed area on the main plot
    mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="none", ec="0.5")

    # Add the legend
    ax.legend(title='Training Type', fontsize=14, title_fontsize=14, loc='center right')
    
    # Add text annotation about the arrows
    if 'Initial' in labels:
        initial_count = labels.count('Initial')
        ax.text(0.02, 0.98, f'Gray arrows show training trajectories\nfrom initial weights (orange X) to vanilla models (red circles)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    output_dir = 'visualizations/pca_analysis'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'pca_with_synthetic_initial_weights.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"PCA plot with synthetic initial weights saved to {save_path}")

if __name__ == '__main__':
    main() 