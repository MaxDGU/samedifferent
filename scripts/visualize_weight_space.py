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

def flatten_weights(model):
    """Flattens all parameters of a model into a single 1D numpy array."""
    return np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()])

def plot_results(results, labels, method_name, output_path):
    """Generates and saves a scatter plot for PCA or t-SNE results."""
    fig, ax = plt.subplots(figsize=(16, 14))
    
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}
    
    for i, label in enumerate(labels):
        ax.scatter(results[i, 0], results[i, 1], c=[color_map[label]], s=150, alpha=0.8, label=label)
        ax.text(results[i, 0], results[i, 1] + 0.05, label, fontsize=9)

    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color_map[label], markersize=12) for label in unique_labels]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)
    
    ax.set_title(f"{method_name} Projection of Model Weights", fontsize=18)
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=14)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n{method_name} plot saved to {output_path}")

def main(args):
    """Main function to run visualizations."""
    
    # Using a single seed for each experiment type for consistency
    pb_seed = 4
    nat_seed = 4
    vanilla_seed = 42

    config = {
        'Meta-PB': {
            'models': {
                'Conv2': f'/scratch/gpfs/mg7411/exp1_conv2lr_runs_20250125_111902/seed_{pb_seed}/model_seed_{pb_seed}_pretesting.pt',
                'Conv4': f'/scratch/gpfs/mg7411/exp1_(finished)conv4lr_runs_20250126_201548/seed_{pb_seed}/model_seed_{pb_seed}_pretesting.pt',
                'Conv6': f'/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6/model_seed_{pb_seed}_pretesting.pt',
            }
        },
        'Meta-Naturalistic': {
            'models': {
                'Conv2': f'/scratch/gpfs/mg7411/samedifferent/naturalistic/results_meta_della/conv2lr/seed_{nat_seed}/conv2lr/seed_{nat_seed}/conv2lr_best.pth',
                'Conv4': f'/scratch/gpfs/mg7411/samedifferent/naturalistic/results_meta_della/conv4lr/seed_{nat_seed}/conv4lr/seed_{nat_seed}/conv4lr_best.pth',
                'Conv6': f'/scratch/gpfs/mg7411/samedifferent/naturalistic/results_meta_della/conv6lr/seed_{nat_seed}/conv6lr/seed_{nat_seed}/conv6lr_best.pth',
            }
        },
        'Vanilla-Naturalistic': {
            'models': {
                'Conv2': f'/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/conv2lr/seed_{vanilla_seed}/best_model.pt',
                'Conv4': f'/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/conv4lr/seed_{vanilla_seed}/best_model.pt',
                'Conv6': f'/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/conv6lr/seed_{vanilla_seed}/best_model.pt',
            }
        }
    }

    # Dynamically import model classes from baselines
    try:
        from baselines.models.conv2 import SameDifferentCNN as ConvNet2LR
        from baselines.models.conv4 import SameDifferentCNN as ConvNet4LR
        from baselines.models.conv6 import SameDifferentCNN as ConvNet6LR
        model_classes = {
            'Conv2': ConvNet2LR,
            'Conv4': ConvNet4LR,
            'Conv6': ConvNet6LR
        }
        print("Successfully imported all model architectures.")
    except ImportError as e:
        print(f"Error importing model architectures: {e}")
        sys.exit(1)

    all_weights = []
    labels = []

    print("\n--- Loading and Flattening Model Weights ---")
    for exp_type, group in config.items():
        for model_arch, path in group['models'].items():
            print(f"  Processing {path}...")
            if not os.path.exists(path):
                print("    WARNING: Weight file not found. Skipping.")
                continue

            try:
                model_class = model_classes[model_arch]
                # Do not pass num_classes, as the base class doesn't accept it.
                model = model_class()
                
                device = torch.device("cpu")
                model.to(device)
                
                checkpoint = torch.load(path, map_location=device)
                
                state_dict = None
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'net' in checkpoint:
                    state_dict = {k.replace('net.', ''): v for k, v in checkpoint['net'].items()}
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
                
                # Load weights with strict=False to tolerate mismatches between model and checkpoint
                model.load_state_dict(state_dict, strict=False)

                flat_weights = flatten_weights(model)
                all_weights.append(flat_weights)
                labels.append(f"{exp_type}-{model_arch}")

            except Exception as e:
                print(f"    ERROR processing {path}: {e}")

    if len(all_weights) < 2:
        print("\nNot enough models found to perform analysis. Aborting.")
        return
        
    # Pad vectors to be the same length for models with different architectures
    max_len = max(len(w) for w in all_weights)
    padded_weights = [np.pad(w, (0, max_len - len(w)), 'constant') for w in all_weights]
    weights_matrix = np.vstack(padded_weights)
    
    # --- Run & Plot Analysis ---
    if args.run_pca:
        print(f"\n--- Running PCA on {weights_matrix.shape[0]} weight vectors ---")
        pca = PCA(n_components=2, random_state=42)
        results = pca.fit_transform(weights_matrix)
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        plot_results(results, labels, "PCA", args.output_dir / "weights_pca.png")

    if args.run_tsne:
        print(f"\n--- Running t-SNE on {weights_matrix.shape[0]} weight vectors ---")
        perplexity = min(30, len(all_weights) - 1)
        if perplexity > 0:
            tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=1000, random_state=42)
            results = tsne.fit_transform(weights_matrix)
            plot_results(results, labels, "t-SNE", args.output_dir / "weights_tsne.png")
        else:
            print("Not enough data points for t-SNE. Skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model weight space.")
    parser.add_argument('--output_dir', type=Path, default=Path('/scratch/gpfs/mg7411/samedifferent/visualizations'),
                        help='Directory to save the output plots.')
    parser.add_argument('--run_pca', action='store_true', default=True, help='Run PCA analysis.')
    parser.add_argument('--run_tsne', action='store_true', default=True, help='Run t-SNE analysis.')
    
    # For future flexibility, though currently unused as paths are hardcoded
    parser.add_argument('--base_dir', type=str, default='/scratch/gpfs/mg7411',
                        help='The base directory for experiment data.')

    args = parser.parse_args()
    main(args) 