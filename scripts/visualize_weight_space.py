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
    from scripts.temp_model import SameDifferentCNN as PB_Hybrid_CNN
    print("Successfully imported all model architectures.")
except ImportError as e:
    print(f"Fatal Error importing models: {e}. A dummy class will be used.")
    class StandardConv2: pass
    class StandardConv4: pass
    class StandardConv6: pass
    class PB_Hybrid_CNN: pass

def flatten_weights(model):
    """Flattens all parameters of a model into a single 1D numpy array."""
    return np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()])

def plot_results(results, labels, method_name, output_path, title):
    """Generates and saves a scatter plot for PCA or t-SNE results."""
    fig, ax = plt.subplots(figsize=(22, 20))
    
    unique_groups = sorted(list(set([l.split('-seed')[0] for l in labels])))
    colors = plt.cm.get_cmap('tab20', len(unique_groups))
    color_map = {group: colors(i) for i, group in enumerate(unique_groups)}

    for i, label in enumerate(labels):
        group_label = label.split('-seed')[0]
        ax.scatter(results[i, 0], results[i, 1], c=[color_map[group_label]], s=180, alpha=0.7)
        
        is_tsne = "t-SNE" in title
        label_text = group_label if is_tsne else label
        if is_tsne:
            if label not in labels[:i]:
                 ax.text(results[i, 0], results[i, 1] + 0.05, label_text, fontsize=12, ha='center')
        else:
            ax.text(results[i, 0], results[i, 1] + 0.05, label_text, fontsize=9, ha='center')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=group, markerfacecolor=color_map[group], markersize=14) for group in unique_groups]
    ax.legend(handles=legend_elements, loc='best', fontsize=14)
    
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=16)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=16)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n{method_name} plot saved to {output_path}")

def get_model_and_path(exp_type, model_arch, seed):
    """Returns the correct model class and file path."""
    if exp_type == 'Meta-PB':
        model_class = PB_Hybrid_CNN
        path_templates = {
            'Conv2': f'/scratch/gpfs/mg7411/exp1_conv2lr_runs_20250125_111902/seed_{seed}/model_seed_{seed}_pretesting.pt',
            'Conv4': f'/scratch/gpfs/mg7411/exp1_(finished)conv4lr_runs_20250126_201548/seed_{seed}/model_seed_{seed}_pretesting.pt',
            'Conv6': f'/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6/model_seed_{seed}_pretesting.pt'
        }
        path = path_templates[model_arch]
    else:
        model_map = {'Conv2': StandardConv2, 'Conv4': StandardConv4, 'Conv6': StandardConv6}
        model_class = model_map[model_arch]
        model_name_lower = model_arch.lower() + 'lr'
        if exp_type == 'Meta-Naturalistic':
            path = f'/scratch/gpfs/mg7411/samedifferent/naturalistic/results_meta_della/{model_name_lower}/seed_{seed}/{model_name_lower}/seed_{seed}/{model_name_lower}_best.pth'
        elif exp_type == 'Vanilla-Naturalistic':
            path = f'/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/{model_name_lower}/seed_{seed}/best_model.pt'
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")
    return model_class, path

def main(args):
    seed_config = {
        'Meta-PB': [0, 1, 2, 3, 4],
        'Meta-Naturalistic': [0, 1, 2, 3, 4],
        'Vanilla-Naturalistic': [789, 42, 999, 555, 123]
    }
    architectures = ['Conv2', 'Conv4', 'Conv6']

    pca_weights, pca_labels = [], []
    tsne_weights, tsne_labels = [], []
    
    print("\n--- Loading and Flattening Model Weights ---")
    for exp_type, seeds in seed_config.items():
        for model_arch in architectures:
            added_for_tsne = False
            for seed in seeds:
                try:
                    model_class, path = get_model_and_path(exp_type, model_arch, seed)
                    if not os.path.exists(path):
                        print(f"    WARNING: Path not found. Skipping: {path}")
                        continue
                    print(f"  Processing {path}...")
                    
                    model = model_class()
                    checkpoint = torch.load(path, map_location=torch.device("cpu"))
                    
                    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict') or checkpoint.get('model') or (checkpoint.get('net') and {k.replace('net.', ''): v for k, v in checkpoint['net'].items()}) or checkpoint
                    
                    model.load_state_dict(state_dict, strict=False)
                    flat_weights = flatten_weights(model)
                    
                    label = f"{exp_type}-{model_arch}-seed{seed}"
                    pca_weights.append(flat_weights)
                    pca_labels.append(label)

                    if not added_for_tsne:
                        tsne_weights.append(flat_weights)
                        tsne_labels.append(label)
                        added_for_tsne = True
                except Exception as e:
                    print(f"    ERROR processing {exp_type}-{model_arch}-seed{seed}: {e}")

    def pad_and_stack(weights_list):
        if not weights_list: return None
        max_len = max(len(w) for w in weights_list)
        return np.vstack([np.pad(w, (0, max_len - len(w)), 'constant') for w in weights_list])

    if pca_weights:
        weights_matrix_pca = pad_and_stack(pca_weights)
        print(f"\n--- Running PCA on {weights_matrix_pca.shape[0]} weight vectors (all seeds) ---")
        pca = PCA(n_components=2, random_state=42)
        results = pca.fit_transform(weights_matrix_pca)
        plot_results(results, pca_labels, "PCA", args.output_dir / "weights_pca_all_seeds.png", "PCA Projection of All Model Weights (All Seeds)")
    
    if tsne_weights:
        weights_matrix_tsne = pad_and_stack(tsne_weights)
        print(f"\n--- Running t-SNE on {weights_matrix_tsne.shape[0]} weight vectors (one seed each) ---")
        perplexity = min(30, weights_matrix_tsne.shape[0] - 1)
        if perplexity > 0:
            tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000, random_state=42)
            results = tsne.fit_transform(weights_matrix_tsne)
            plot_results(results, tsne_labels, "t-SNE", args.output_dir / "weights_tsne_one_seed.png", "t-SNE Projection of Model Weights (One Seed per Type)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model weight space.")
    parser.add_argument('--output_dir', type=Path, default=Path('/scratch/gpfs/mg7411/samedifferent/visualizations'),
                        help='Directory to save the output plots.')
    args = parser.parse_args()
    main(args) 