import torch
import numpy as np
import os
import sys
from pathlib import Path
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import importlib

# --- Setup Project Path ---
# This ensures that we can import modules from the project root (e.g., meta_baseline)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_weights(model):
    """Flattens and returns the weights of a model as a numpy array."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def find_weight_file(directory, possible_names):
    """Searches a directory for the first matching file from a list of names."""
    for name in possible_names:
        path = directory / name
        if path.exists():
            return path
    return None

def load_weights_from_path(model_path, model_class, device):
    """Loads a model's state_dict from a file and returns its flattened weights."""
    model = model_class().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint structures
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    return get_model_weights(model)

def import_model_class(architecture):
    """Dynamically imports the model class from the meta_baseline.models module."""
    try:
        # Correctly form the module name, e.g., 'conv6' -> 'conv6lr'
        module_name = architecture + 'lr' if 'conv' in architecture else architecture
        module_path = f"meta_baseline.models.{module_name}"
        module = importlib.import_module(module_path)
        return module.SameDifferentCNN
    except ImportError:
        print(f"Error: Could not import model for architecture '{architecture}'.")
        print(f"Attempted to import from '{module_path}'. Please check the file and class names.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize model weight space using PCA.")
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'], help='Model architecture to visualize.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 555, 789, 999], help='List of random seeds to process.')
    
    # Directories for different model types
    parser.add_argument('--naturalistic_results_dir', type=str, default="logs_naturalistic_meta", help="Directory for naturalistic MAML models.")
    parser.add_argument('--vanilla_results_dir', type=str, default="logs_naturalistic_vanilla", help="Directory for naturalistic Vanilla models.")
    parser.add_argument('--pb_results_dir', type=str, default="results/pb_retrained_conv6lr", help="Directory for PB-retrained MAML models.")
    
    parser.add_argument('--output_dir', type=str, default="visualizations/multi_task_pca", help="Directory to save the PCA plots.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = import_model_class(args.architecture)
    arch_name = args.architecture + 'lr' if 'conv' in args.architecture else args.architecture

    all_weights = []
    all_labels = []

    # --- 1. Load MAML (Naturalistic) Weights ---
    print("--- Loading MAML (Naturalistic) Weights ---")
    nat_maml_dir = Path(args.naturalistic_results_dir)
    for seed in args.seeds:
        print(f"\n- Seed {seed}:")
        # Re-create initial state since we can't find a saved file reliably
        set_seed(seed)
        initial_model = model_class().to(device)
        all_weights.append(get_model_weights(initial_model))
        all_labels.append("MAML (Nat) Initial")
        print("  Created initial MAML (Nat) weights on-the-fly.")

        # Construct the specific, nested path for the final model
        final_path = nat_maml_dir / arch_name / f"seed_{seed}" / arch_name / f"seed_{seed}" / f"{arch_name}_best.pth"
        print(f"  Checking for final model: {final_path}")
        if final_path.exists():
            all_weights.append(load_weights_from_path(final_path, model_class, device))
            all_labels.append("MAML (Nat) Final")
            print("    -> Found and loaded.")
        else:
            print("    -> Not found.")

    # --- 2. Load Vanilla (Naturalistic) Weights ---
    print("\n--- Loading Vanilla (Naturalistic) Weights ---")
    vanilla_dir = Path(args.vanilla_results_dir)
    for seed in args.seeds:
        print(f"\n- Seed {seed}:")
        # Re-create initial state to match MAML's start for a fair comparison
        set_seed(seed)
        initial_model = model_class().to(device)
        all_weights.append(get_model_weights(initial_model))
        all_labels.append("Vanilla (Nat) Initial")
        print("  Created initial Vanilla (Nat) weights on-the-fly.")

        vanilla_seed_dir = vanilla_dir / arch_name / f"seed_{seed}"
        final_path = find_weight_file(vanilla_seed_dir, ["best_model.pt", "final_model.pth"])
        print(f"  Searching for final model in: {vanilla_seed_dir}")
        if final_path:
            all_weights.append(load_weights_from_path(final_path, model_class, device))
            all_labels.append("Vanilla (Nat) Final")
            print(f"    -> Found and loaded: {final_path.name}")
        else:
            print("    -> Not found.")

    # --- 3. Load MAML (PB-Trained) Weights ---
    print("\n--- Loading MAML (PB-Trained) Weights ---")
    pb_dir = Path(args.pb_results_dir) / args.architecture
    for seed in args.seeds:
        print(f"\n- Seed {seed}:")
        set_seed(seed)
        initial_model = model_class().to(device)
        all_weights.append(get_model_weights(initial_model))
        all_labels.append("MAML (PB) Initial")
        print("  Created initial PB model weights.")

        final_path = pb_dir / f"seed_{seed}" / "best_model.pt"
        print(f"  Checking for final model: {final_path}")
        if final_path.exists():
            all_weights.append(load_weights_from_path(final_path, model_class, device))
            all_labels.append("MAML (PB) Final")
            print("    -> Found and loaded.")
        else:
            print("    -> Not found.")
            
    # --- Shape Consistency Check ---
    print("\n--- Checking for Shape Consistency ---")
    if not all_weights:
        print("Error: No weights were loaded. Exiting.")
        return
        
    shapes = [w.shape for w in all_weights]
    most_common_shape = max(set(shapes), key=shapes.count)
    
    filtered_weights = [w for w, s in zip(all_weights, shapes) if s == most_common_shape]
    filtered_labels = [l for l, s in zip(all_labels, shapes) if s == most_common_shape]
    
    if len(filtered_weights) != len(all_weights):
        print(f"Warning: Original weights count: {len(all_weights)}. Filtered count: {len(filtered_weights)}.")
        print(f"Keeping weights with shape: {most_common_shape}")

    if len(filtered_weights) < 2:
        print("Error: Fewer than 2 valid weight vectors loaded. Cannot perform PCA.")
        return

    # --- PCA and Plotting ---
    print("\n--- Performing PCA and Plotting ---")
    pca = PCA(n_components=2)
    projected_weights = pca.fit_transform(np.vstack(filtered_weights))

    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_styles = [
        ("MAML (PB) Initial", "x", "#ff7f0e"),      # Orange X
        ("MAML (PB) Final", "P", "#ff7f0e"),        # Orange Plus
        ("MAML (Nat) Initial", "x", "#1f77b4"),     # Blue X
        ("MAML (Nat) Final", "o", "#1f77b4"),       # Blue Circle
        ("Vanilla (Nat) Initial", "x", "#2ca02c"),  # Green X (Same as MAML Nat Initial)
        ("Vanilla (Nat) Final", "s", "#2ca02c"),    # Green Square
    ]
    style_map = {label: (marker, color) for label, marker, color in unique_styles}

    for i, label in enumerate(filtered_labels):
        marker, color = style_map.get(label, ("d", "grey")) # Default: diamond, grey
        ax.scatter(projected_weights[i, 0], projected_weights[i, 1],
                   marker=marker, color=color, s=100, alpha=0.8, label=label)

    # Add lines connecting initial to final states
    for seed in args.seeds:
        # PB line
        try:
            initial_pb_idx = filtered_labels.index("MAML (PB) Initial")
            final_pb_idx = filtered_labels.index("MAML (PB) Final")
            ax.plot([projected_weights[initial_pb_idx, 0], projected_weights[final_pb_idx, 0]],
                    [projected_weights[initial_pb_idx, 1], projected_weights[final_pb_idx, 1]],
                    color='#ff7f0e', linestyle='--', alpha=0.5)
        except ValueError:
            pass # A point might be missing
            
        # MAML Naturalistic line
        try:
            initial_nat_maml_idx = filtered_labels.index("MAML (Nat) Initial")
            final_nat_maml_idx = filtered_labels.index("MAML (Nat) Final")
            ax.plot([projected_weights[initial_nat_maml_idx, 0], projected_weights[final_nat_maml_idx, 0]],
                    [projected_weights[initial_nat_maml_idx, 1], projected_weights[final_nat_maml_idx, 1]],
                    color='#1f77b4', linestyle='--', alpha=0.5)
        except ValueError:
            pass

        # Vanilla Naturalistic line
        try:
            initial_nat_van_idx = filtered_labels.index("Vanilla (Nat) Initial")
            final_nat_van_idx = filtered_labels.index("Vanilla (Nat) Final")
            ax.plot([projected_weights[initial_nat_van_idx, 0], projected_weights[final_nat_van_idx, 0]],
                    [projected_weights[initial_nat_van_idx, 1], projected_weights[final_nat_van_idx, 1]],
                    color='#2ca02c', linestyle='--', alpha=0.5)
        except ValueError:
            pass
            
    ax.set_title(f"PCA of Model Weights for {args.architecture.upper()} Architecture", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker=marker, color='w', label=label, markerfacecolor=color, markersize=10)
        for label, marker, color in unique_styles
    ]
    ax.legend(handles=legend_elements, title="Weight States", loc="best")

    plt.grid(True, linestyle='--', alpha=0.6)
    output_path = Path(args.output_dir) / f"pca_weights_{arch_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"\nPCA plot saved to {output_path}")

if __name__ == "__main__":
    main() 