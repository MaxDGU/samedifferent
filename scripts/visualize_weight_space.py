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

def get_layer_weights_for_tsne(model):
    """Returns a list of flattened layer weights and a corresponding list of layer names."""
    weights_list = []
    labels_list = []
    state_dict = model.state_dict()
    # Sort by key to ensure consistent order and only include tensors with actual data
    for key, param in sorted(state_dict.items()):
        if isinstance(param, torch.Tensor) and param.ndim > 0:
            weights_list.append(param.detach().cpu().numpy().flatten())
            labels_list.append(key)
    return weights_list, labels_list

def plot_pca_results(results, labels, output_path, title, xlim=None, ylim=None):
    """Generates and saves a scatter plot specifically for the overall PCA results."""
    fig, ax = plt.subplots(figsize=(22, 20))
    
    unique_groups = sorted(list(set([l.split('-seed')[0] for l in labels])))
    colors = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, len(unique_groups)))
    color_map = {group: colors[i] for i, group in enumerate(unique_groups)}

    for i, label in enumerate(labels):
        group_label = label.split('-seed')[0]
        if xlim and not (xlim[0] <= results[i, 0] <= xlim[1]): continue
        if ylim and not (ylim[0] <= results[i, 1] <= ylim[1]): continue

        ax.scatter(results[i, 0], results[i, 1], c=[color_map[group_label]], s=180, alpha=0.7)
        
        font_size = 8 if "Zoomed" in title else 12
        label_text = label if "Zoomed" in title else group_label
        ax.text(results[i, 0], results[i, 1] + 0.05, label_text, fontsize=font_size, ha='center')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=group, markerfacecolor=color_map[group], markersize=14) for group in unique_groups]
    ax.legend(handles=legend_elements, loc='best', fontsize=14)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("PCA Dimension 1", fontsize=16)
    ax.set_ylabel("PCA Dimension 2", fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\nPCA plot saved to {output_path}")

def plot_layer_tsne(results, labels, output_path, title):
    """Generates and saves a t-SNE plot for a single model's layers."""
    fig, ax = plt.subplots(figsize=(22, 20))
    ax.scatter(results[:, 0], results[:, 1], s=250, alpha=0.8)
    for i, label in enumerate(labels):
        ax.text(results[i, 0], results[i, 1] + 0.1, label, fontsize=10, ha='center')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=16)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\nLayer t-SNE plot saved to {output_path}")

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
    # Seed configuration
    seed_config = {
        'Meta-PB': {'Conv2': [0], 'Conv4': [0], 'Conv6': [3]},
        'Meta-Naturalistic': {'Conv2': [0], 'Conv4': [0], 'Conv6': [0]},
        'Vanilla-Naturalistic': {'Conv2': [789], 'Conv4': [789], 'Conv6': [789]},
        'Vanilla-PB': {'Conv2': [46], 'Conv4': [46], 'Conv6': [46]}
    }
    
    all_weights_for_pca, all_labels_for_pca = [], []
    
    print("\n--- Loading Model Weights for PCA and Layer-wise t-SNE ---")
    for exp_type, arch_seeds in seed_config.items():
        for model_arch, seeds in arch_seeds.items():
            for seed in seeds:
                try:
                    model_class, path = get_model_and_path(exp_type, model_arch, seed)
                    if not os.path.exists(path):
                        print(f"    WARNING: Path not found. Skipping: {path}")
                        continue
                    
                    print(f"\nProcessing {exp_type}-{model_arch}-seed{seed}")
                    print(f"  Path: {path}")
                    
                    model = model_class()
                    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
                    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint
                    model.load_state_dict(state_dict, strict=False)
                    
                    # --- 1. Prepare data for overall PCA ---
                    full_flat_weights = flatten_weights(model)
                    all_weights_for_pca.append(full_flat_weights)
                    all_labels_for_pca.append(f"{exp_type}-{model_arch}") # Use group label for PCA

                    # --- 2. Generate individual t-SNE plot for this model's layers ---
                    layer_weights, layer_labels = get_layer_weights_for_tsne(model)
                    
                    if len(layer_weights) < 2:
                        print("    Not enough layers for t-SNE. Skipping.")
                        continue
                    
                    print(f"  Running t-SNE on {len(layer_weights)} layers...")
                    def pad_and_stack_layers(weights_list):
                        """Pads a list of 1D arrays to the same length and stacks them."""
                        if not weights_list: return None
                        max_len = max(len(w) for w in weights_list)
                        return np.vstack([np.pad(w, (0, max_len - len(w)), 'constant') for w in weights_list])
                    
                    weights_matrix_layer = pad_and_stack_layers(layer_weights)
                    tsne = TSNE(n_components=2, verbose=0, perplexity=min(30, len(layer_weights)-1), n_iter=1000, random_state=42)
                    results_tsne = tsne.fit_transform(weights_matrix_layer)
                    
                    output_filename = f"tsne_layers_{exp_type}-{model_arch}-seed{seed}.png"
                    plot_title = f"t-SNE of Layers for {exp_type}-{model_arch} (Seed {seed})"
                    plot_layer_tsne(results_tsne, layer_labels, args.output_dir / output_filename, plot_title)

                except Exception as e:
                    print(f"    ERROR processing {exp_type}-{model_arch}-seed{seed}: {e}")
        
    # --- Overall PCA Section (on all models) ---
    def pad_and_stack_models(weights_list):
        if not weights_list: return None
        max_len = max(len(w) for w in weights_list)
        return np.vstack([np.pad(w, (0, max_len - len(w)), 'constant') for w in weights_list])

    if all_weights_for_pca:
        weights_matrix_pca = pad_and_stack_models(all_weights_for_pca)
        print(f"\n--- Running PCA on {weights_matrix_pca.shape[0]} total models ---")
        pca = PCA(n_components=2, random_state=42)
        results_pca = pca.fit_transform(weights_matrix_pca)
    
        plot_pca_results(results_pca, all_labels_for_pca, args.output_dir / "weights_pca_all_models.png", "PCA Projection of All Model Weights")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model weight space, including layer-wise t-SNE.")
    parser.add_argument('--output_dir', type=Path, default=Path.cwd() / 'visualizations' / 'weight_space',
                        help="Directory to save the plots.")
    args = parser.parse_args()
    main(args) 