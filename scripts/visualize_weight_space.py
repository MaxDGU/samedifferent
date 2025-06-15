import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Import Model Architectures ---
# We need to import all potential architectures.
# The user can select which one to use via command-line args.
try:
    from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
    from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
    from baselines.models.conv6 import SameDifferentCNN as Conv6CNN
    # The 'PB' weights have a custom architecture we discovered.
    from scripts.temp_model import SameDifferentCNN as PB_Hybrid_CNN
    print("Successfully imported all model architectures.")
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Please ensure all model definition files exist.")
    sys.exit(1)

def load_and_flatten_weights(model_path, model_class, strict=False):
    """Loads a state_dict and returns flattened weights."""
    print(f"  Processing {model_path}...")
    if not Path(model_path).exists():
        print(f"    WARNING: Weight file not found. Skipping.")
        return None
        
    model = model_class()
    # It's safer to load to CPU first
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Handle nested state dicts from different saving conventions
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    model.load_state_dict(state_dict, strict=strict)
    
    return np.concatenate([p.cpu().numpy().flatten() for p in model.state_dict().values()])

def plot_results(results, labels, method_name, output_path):
    """Generates and saves a scatter plot for PCA or t-SNE results."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define colors and markers for the four groups
    color_map = {'PB': '#1f77b4', 'Nat': '#ff7f0e'}
    marker_map = {'MAML': 'o', 'Vanilla': 'X'}
    
    for i, label in enumerate(labels):
        dataset_type, model_type = label.split('-')
        color = color_map[dataset_type]
        marker = marker_map[model_type]
        
        ax.scatter(results[i, 0], results[i, 1], c=color, marker=marker, s=150, alpha=0.8, label=label if not ax.get_legend() else "")
        ax.text(results[i, 0] + 0.1, results[i, 1] + 0.1, label, fontsize=9)
        
    # Create a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='MAML-PB', markerfacecolor=color_map['PB'], markersize=12),
        Line2D([0], [0], marker='o', color='w', label='MAML-Naturalistic', markerfacecolor=color_map['Nat'], markersize=12),
        # Line2D([0], [0], marker='X', color='w', label='Vanilla-PB', markerfacecolor=color_map['PB'], markersize=12, markeredgewidth=1.5, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Vanilla-Naturalistic', markerfacecolor=color_map['Nat'], markersize=12, markeredgewidth=1.5, markeredgecolor='k')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)
    
    ax.set_title(f"{method_name} Projection of Model Weights", fontsize=18)
    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=14)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n{method_name} plot saved to {output_path}")

def main(args):
    """Main function to run visualizations."""
    
    # --- Define All Models to Compare ---
    # A list of tuples: (Group Label, Architecture Label, Model Class, Path, strict_loading)
    models_to_compare = [
        # MAML Models
        ("PB-MAML", "conv2", PB_Hybrid_CNN, f"{args.base_dir}/exp1_conv2lr_runs_20250125_111902/seed_42/model_seed_42_pretesting.pt", False),
        ("PB-MAML", "conv4", PB_Hybrid_CNN, f"{args.base_dir}/exp1_(finished)conv4lr_runs_20250126_201548/seed_42/model_seed_42_pretesting.pt", False),
        ("PB-MAML", "conv6", PB_Hybrid_CNN, f"{args.base_dir}/maml_pbweights_conv6/model_seed_42_pretesting.pt", False),
        ("Nat-MAML", "conv2", Conv2CNN, f"{args.base_dir}/naturalistic/results_meta_della/conv2lr/seed_42/conv2lr_best.pth", True),
        ("Nat-MAML", "conv4", Conv4CNN, f"{args.base_dir}/naturalistic/results_meta_della/conv4lr/seed_42/conv4lr_best.pth", True),
        ("Nat-MAML", "conv6", Conv6CNN, f"{args.base_dir}/naturalistic/results_meta_della/conv6lr/seed_42/conv6lr_best.pth", True),
        
        # Vanilla SGD Models (PB temporarily disabled)
        # ("PB-Vanilla", "conv2", Conv2CNN, f"{args.vanilla_pb_dir}/all_tasks/conv2/test_regular/seed_42/best_model.pt", True),
        # ("PB-Vanilla", "conv4", Conv4CNN, f"{args.vanilla_pb_dir}/all_tasks/conv4/test_regular/seed_42/best_model.pt", True),
        # ("PB-Vanilla", "conv6", Conv6CNN, f"{args.vanilla_pb_dir}/all_tasks/conv6/test_regular/seed_42/best_model.pt", True),
        ("Nat-Vanilla", "conv2", Conv2CNN, f"{args.vanilla_nat_dir}/conv2lr/seed_42/best_model.pt", True),
        ("Nat-Vanilla", "conv4", Conv4CNN, f"{args.vanilla_nat_dir}/conv4lr/seed_42/best_model.pt", True),
        ("Nat-Vanilla", "conv6", Conv6CNN, f"{args.vanilla_nat_dir}/conv6lr/seed_42/best_model.pt", True),
    ]

    all_weights = []
    all_labels = []

    print("--- Loading and Flattening Model Weights ---")
    for group_label, arch_label, model_class, path, strict in models_to_compare:
        flat_weights = load_and_flatten_weights(path, model_class, strict=strict)
        if flat_weights is not None:
            all_weights.append(flat_weights)
            # Combine group label with architecture
            all_labels.append(f"{group_label}-{arch_label}")

    if len(all_weights) < 2:
        print("Not enough models found to perform analysis. Aborting.")
        return
        
    # --- Pad vectors to be the same length ---
    max_len = max(len(w) for w in all_weights)
    padded_weights = [np.pad(w, (0, max_len - len(w)), 'constant') for w in all_weights]
    weights_matrix = np.vstack(padded_weights)
    
    # --- Run PCA ---
    print(f"\n--- Running PCA on {weights_matrix.shape[0]} weight vectors ---")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(weights_matrix)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    plot_results(pca_results, all_labels, "PCA", args.output_pca)

    # --- Run t-SNE ---
    print(f"\n--- Running t-SNE on {weights_matrix.shape[0]} weight vectors ---")
    tsne = TSNE(n_components=2, verbose=1, perplexity=min(5, len(all_weights) - 1), n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(weights_matrix)
    plot_results(tsne_results, all_labels, "t-SNE", args.output_tsne)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model weight space using PCA and t-SNE.")
    parser.add_argument('--base_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent',
                        help='The base directory for MAML experiment data.')
    parser.add_argument('--vanilla_pb_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/results/pb_baselines',
                        help='The directory for vanilla PB-trained models.')
    parser.add_argument('--vanilla_nat_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla',
                        help='The directory for vanilla Naturalistic-trained models.')
    parser.add_argument('--output_pca', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/weights_pca.png',
                        help='Path to save the output PCA plot.')
    parser.add_argument('--output_tsne', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/weights_tsne.png',
                        help='Path to save the output t-SNE plot.')
    args = parser.parse_args()
    main(args) 