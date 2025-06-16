import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import os

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Model Imports ---
try:
    from baselines.models.conv2 import SameDifferentCNN as StandardConv2
    from baselines.models.conv4 import SameDifferentCNN as StandardConv4
    from baselines.models.conv6 import SameDifferentCNN as StandardConv6
    from scripts.temp_model import PB_Conv2, PB_Conv4, PB_Conv6
    print("Successfully imported all model architectures.")
except ImportError as e:
    print(f"Fatal Error importing models: {e}. A dummy class will be used.")
    class StandardConv2: pass
    class StandardConv4: pass
    class StandardConv6: pass
    class PB_Conv2: pass
    class PB_Conv4: pass
    class PB_Conv6: pass

def flatten_weights(model):
    """Flattens all parameters of a model into a single 1D numpy array."""
    return np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()])

def plot_results(results, labels, method_name, output_path, title, xlim=None, ylim=None):
    """Generates and saves a scatter plot for PCA or t-SNE results."""
    fig, ax = plt.subplots(figsize=(22, 20))
    
    unique_groups = sorted(list(set([l.split('-seed')[0] for l in labels])))
    colors = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, len(unique_groups)))
    color_map = {group: colors[i] for i, group in enumerate(unique_groups)}

    for i, label in enumerate(labels):
        group_label = label.split('-seed')[0]
        if xlim and not (xlim[0] <= results[i, 0] <= xlim[1]): continue
        if ylim and not (ylim[0] <= results[i, 1] <= ylim[1]): continue

        ax.scatter(results[i, 0], results[i, 1], c=[color_map[group_label]], s=180, alpha=0.7)
        
        font_size = 12
        label_text = label # Show full label (e.g., with seed) on individual plots
        
        # Adjust label display based on plot type
        if "PCA" in title:
            label_text = group_label if "Zoomed" not in title else label
            font_size = 8 if "Zoomed" in title else 12
        elif "t-SNE" in title and len(unique_groups) > 1: # Combined t-SNE (if ever used)
             label_text = group_label
        
        ax.text(results[i, 0], results[i, 1] + 0.05, label_text, fontsize=font_size, ha='center')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=group, markerfacecolor=color_map[group], markersize=14) for group in unique_groups]
    ax.legend(handles=legend_elements, loc='best', fontsize=14)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=16)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    print(f"\nPlot saved to {output_path}")

def get_model_and_path(exp_type, model_arch, seed):
    """Returns the correct model class and file path."""
    if exp_type == 'Meta-PB':
        model_map = {'Conv2': PB_Conv2, 'Conv4': PB_Conv4, 'Conv6': PB_Conv6}
        model_class = model_map[model_arch]
        path_templates = {
            'Conv2': f'/scratch/gpfs/mg7411/exp1_conv2lr_runs_20250125_111902/seed_{seed}/model_seed_{seed}_pretesting.pt',
            'Conv4': f'/scratch/gpfs/mg7411/exp1_(finished)conv4lr_runs_20250126_201548/seed_{seed}/model_seed_{seed}_pretesting.pt',
            'Conv6': f'/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6/model_seed_{seed}_pretesting.pt'
        }
        path = path_templates.get(model_arch)
    else: # All other models use the standard architectures
        model_map = {'Conv2': StandardConv2, 'Conv4': StandardConv4, 'Conv6': StandardConv6}
        model_class = model_map[model_arch]
        
        if exp_type == 'Meta-Naturalistic':
            model_name_lower = model_arch.lower() + 'lr'
            path = f'/scratch/gpfs/mg7411/samedifferent/naturalistic/results_meta_della/{model_name_lower}/seed_{seed}/{model_name_lower}/seed_{seed}/{model_name_lower}_best.pth'
        elif exp_type == 'Vanilla-Naturalistic':
            model_name_lower = model_arch.lower() + 'lr'
            path = f'/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/{model_name_lower}/seed_{seed}/best_model.pt'
        elif exp_type == 'Vanilla-PB':
            model_name_lower_no_lr = model_arch.lower()
            path = f'/scratch/gpfs/mg7411/results/pb_baselines/regular/{model_name_lower_no_lr}/seed_{seed}/best_model.pt'
        else:
            path = None 
    
    if path is None:
        raise ValueError(f"Could not determine path for {exp_type} {model_arch}")
    return model_class, path

def main(args):
    # More granular seed configuration to handle special cases
    seed_config = {
        'Meta-PB': {
            'Conv2': [0, 1, 2, 3, 4],
            'Conv4': [0, 1, 2, 3, 4],
            'Conv6': [3, 4, 5, 6, 7],
        },
        'Meta-Naturalistic': {
            'Conv2': [0, 1, 2, 3, 4],
            'Conv4': [0, 1, 2, 3, 4],
            'Conv6': [0, 1, 2, 3, 4],
        },
        'Vanilla-Naturalistic': {
            'Conv2': [789, 42, 999, 555, 123],
            'Conv4': [789, 42, 999, 555, 123],
            'Conv6': [789, 42, 999, 555, 123],
        },
        'Vanilla-PB': {
            'Conv2': [46, 47, 48, 49, 50],
            'Conv4': [46, 47, 48, 49, 50],
            'Conv6': [46, 47, 48, 49, 50],
        }
    }
    
    all_weights, all_labels = [], []
    grouped_weights = {} # New dictionary to group weights by model type
    
    print("\n--- Loading and Flattening Model Weights ---")
    for exp_type, arch_seeds in seed_config.items():
        for model_arch, seeds in arch_seeds.items():
            group_label = f"{exp_type}-{model_arch}"
            grouped_weights[group_label] = {'weights': [], 'labels': []}

            for seed in seeds:
                try:
                    model_class, path = get_model_and_path(exp_type, model_arch, seed)
                    if not os.path.exists(path):
                        print(f"    WARNING: Path not found. Skipping: {path}")
                        continue
                    print(f"  Processing {path}...")
                    
                    model = model_class()
                    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
                    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
                    
                    model.load_state_dict(state_dict, strict=False)
                    flat_weights = flatten_weights(model)
                    label = f"{group_label}-seed{seed}"

                    # Add to master list for PCA
                    all_weights.append(flat_weights)
                    all_labels.append(label)
                    
                    # Add to grouped dictionary for individual t-SNE plots
                    grouped_weights[group_label]['weights'].append(flat_weights)
                    grouped_weights[group_label]['labels'].append(label)

                except Exception as e:
                    print(f"    ERROR processing {exp_type}-{model_arch}-seed{seed}: {e}")

    def pad_and_stack(weights_list):
        if not weights_list: return None
        max_len = max(len(w) for w in weights_list)
        return np.vstack([np.pad(w, (0, max_len - len(w)), 'constant') for w in weights_list])

    # --- PCA Section (on all weights) ---
    if all_weights:
        weights_matrix_pca = pad_and_stack(all_weights)
        print(f"\n--- Running PCA on {weights_matrix_pca.shape[0]} weight vectors (all seeds) ---")
        pca = PCA(n_components=2, random_state=42)
        results = pca.fit_transform(weights_matrix_pca)
        
        plot_results(results, all_labels, "PCA", args.output_dir / "weights_pca_all_seeds.png", "PCA Projection of All Model Weights (All Seeds)")

        x_coords, y_coords = results[:, 0], results[:, 1]
        x_mean, x_std = np.mean(x_coords), np.std(x_coords)
        y_mean, y_std = np.mean(y_coords), np.std(y_coords)
        zoom_mask = (np.abs(x_coords - x_mean) < x_std) & (np.abs(y_coords - y_mean) < y_std)
        zoom_x_min, zoom_x_max = results[zoom_mask, 0].min() - 5, results[zoom_mask, 0].max() + 5
        zoom_y_min, zoom_y_max = results[zoom_mask, 1].min() - 5, results[zoom_mask, 1].max() + 5
        plot_results(results, all_labels, "PCA", args.output_dir / "weights_pca_all_seeds_zoomed.png", "PCA Projection (Zoomed In)", xlim=(zoom_x_min, zoom_x_max), ylim=(zoom_y_min, zoom_y_max))

    # --- Individual t-SNE Section ---
    print("\n--- Generating individual t-SNE plots for each model group ---")
    for group, data in grouped_weights.items():
        group_weights = data['weights']
        group_labels = data['labels']
        
        if len(group_weights) < 2:
            print(f"  Group '{group}' has < 2 data points. Skipping t-SNE.")
            continue
            
        print(f"  Running t-SNE for group: '{group}' ({len(group_weights)} seeds)")
        weights_matrix_group = pad_and_stack(group_weights)
        perplexity = min(30, weights_matrix_group.shape[0] - 1)
        
        tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=1000, random_state=42)
        results = tsne.fit_transform(weights_matrix_group)
            
        output_filename = f"tsne_individual_{group}.png"
        plot_title = f"t-SNE Projection for {group} (All Seeds)"
        plot_results(results, group_labels, "t-SNE", args.output_dir / output_filename, plot_title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model weight space.")
    parser.add_argument('--output_dir', type=Path, default=Path.cwd() / 'visualizations' / 'weight_space',
                        help="Directory to save the plots.")
    args = parser.parse_args()
    main(args) 