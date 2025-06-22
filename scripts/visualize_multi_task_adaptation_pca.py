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
    state_dict = checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    
    # Load the state dict with strict=False to ignore mismatches like missing BN stats
    model.load_state_dict(state_dict, strict=False)
        
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

def import_legacy_model_class(architecture):
    """Dynamically imports the LEGACY model class from the baselines.models module."""
    try:
        # Legacy models are not named with 'lr'
        module_path = f"baselines.models.{architecture}"
        module = importlib.import_module(module_path)
        # The class name is different in legacy files
        return module.SameDifferentCNN
    except ImportError:
        print(f"Error: Could not import legacy model for architecture '{architecture}'.")
        print(f"Attempted to import from '{module_path}'.")
        sys.exit(1)
    except AttributeError:
        print(f"Error: Class 'SameDifferentCNN' not found in '{module_path}'.")
        sys.exit(1)

def generate_pca_plot(weights, labels, arch_name, output_dir, suffix):
    """Performs PCA and generates a plot for a given set of weights."""
    print(f"\n--- Generating PCA for {suffix} Architecture ---")
    
    # 1. Shape Consistency Check
    if not weights:
        print("No weights provided. Skipping plot.")
        return
        
    shapes = [w.shape for w in weights]
    most_common_shape = max(set(shapes), key=shapes.count)
    
    filtered_weights = [w for w, s in zip(weights, shapes) if s == most_common_shape]
    filtered_labels = [l for l, s in zip(labels, shapes) if s == most_common_shape]
    
    if len(filtered_weights) != len(weights):
        print(f"  Warning: Original weights count: {len(weights)}. Filtered count: {len(filtered_weights)}.")
        print(f"  Keeping weights with shape: {most_common_shape}")

    if len(filtered_weights) < 2:
        print("  Error: Fewer than 2 valid weight vectors. Cannot perform PCA.")
        return

    # 2. PCA
    pca = PCA(n_components=2)
    projected_weights = pca.fit_transform(np.vstack(filtered_weights))
    print(f"  PCA complete. Explained variance: {pca.explained_variance_ratio_}")

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_styles = [
        ("MAML (PB) Initial", "x", "#ff7f0e"),
        ("MAML (PB) Final", "P", "#ff7f0e"),
        ("MAML (Nat) Initial", "x", "#1f77b4"),
        ("MAML (Nat) Final", "o", "#1f77b4"),
        ("Vanilla (Nat) Initial", "x", "#1f77b4"), # Use same color as MAML Nat for shared initial state
        ("Vanilla (Nat) Final", "s", "#2ca02c"),
    ]
    style_map = {label: (marker, color) for label, marker, color in unique_styles}

    # Plot points
    for i, label in enumerate(filtered_labels):
        marker, color = style_map.get(label, ("d", "grey"))
        ax.scatter(projected_weights[i, 0], projected_weights[i, 1],
                   marker=marker, color=color, s=120, alpha=0.8, zorder=5)

    # Plot lines connecting initial to final states
    unique_prefixes = sorted(list(set(l.rsplit(' ', 1)[0] for l in filtered_labels)))
    for prefix in unique_prefixes:
        initial_label = f"{prefix} Initial"
        final_label = f"{prefix} Final"
        
        initial_indices = [i for i, l in enumerate(filtered_labels) if l == initial_label]
        final_indices = [i for i, l in enumerate(filtered_labels) if l == final_label]
        
        # Draw lines between corresponding initial and final points
        for i_idx in initial_indices:
            # Simple heuristic: connect to nearest final point of same type
            if not final_indices: continue
            distances = [np.linalg.norm(projected_weights[i_idx] - projected_weights[f_idx]) for f_idx in final_indices]
            f_idx = final_indices[np.argmin(distances)]

            _, color = style_map.get(initial_label, ("d", "grey"))
            ax.plot([projected_weights[i_idx, 0], projected_weights[f_idx, 0]],
                    [projected_weights[i_idx, 1], projected_weights[f_idx, 1]],
                    color=color, linestyle='--', alpha=0.4, zorder=1)

    ax.set_title(f"PCA of Model Weights for {arch_name.upper()} ({suffix} Arch)", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)

    # Create legend from the styles relevant to this plot
    relevant_styles = [style for style in unique_styles if any(style[0] in l for l in filtered_labels)]
    legend_elements = [
        Line2D([0], [0], marker=marker, color='w', label=label, markerfacecolor=color, markersize=10)
        for label, marker, color in relevant_styles
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, title="Weight States", loc="best")

    plt.grid(True, linestyle='--', alpha=0.6)
    output_path = Path(output_dir) / f"pca_weights_{arch_name}_{suffix.lower()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"PCA plot saved to {output_path}")

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
    # Import both new and old model classes
    new_model_class = import_model_class(args.architecture)
    legacy_model_class = import_legacy_model_class(args.architecture)
    arch_name = args.architecture + 'lr' if 'conv' in args.architecture else args.architecture

    all_weights = []
    all_labels = []

    # --- 1. Load MAML (Naturalistic) Weights ---
    print("--- Loading MAML (Naturalistic) Weights ---")
    nat_maml_dir = Path(args.naturalistic_results_dir)
    for seed in args.seeds:
        print(f"\n- Seed {seed}:")
        # Re-create initial state using the LEGACY model for apples-to-apples comparison
        set_seed(seed)
        initial_model = legacy_model_class().to(device)
        all_weights.append(get_model_weights(initial_model))
        all_labels.append("MAML (Nat) Initial")
        print("  Created initial MAML (Nat) weights on-the-fly (using legacy model class).")

        # Construct the specific, nested path for the final model
        final_path = nat_maml_dir / arch_name / f"seed_{seed}" / arch_name / f"seed_{seed}" / f"{arch_name}_best.pth"
        print(f"  Checking for final model: {final_path}")
        if final_path.exists():
            # Load these weights using the LEGACY model class
            all_weights.append(load_weights_from_path(final_path, legacy_model_class, device))
            all_labels.append("MAML (Nat) Final")
            print("    -> Found and loaded.")
        else:
            print("    -> Not found.")

    # --- 2. Load Vanilla (Naturalistic) Weights ---
    print("\n--- Loading Vanilla (Naturalistic) Weights ---")
    vanilla_dir = Path(args.vanilla_results_dir)
    for seed in args.seeds:
        print(f"\n- Seed {seed}:")
        # Re-create initial state to match MAML's start, using the LEGACY model class
        set_seed(seed)
        initial_model = legacy_model_class().to(device)
        all_weights.append(get_model_weights(initial_model))
        all_labels.append("Vanilla (Nat) Initial")
        print("  Created initial Vanilla (Nat) weights on-the-fly (using legacy model class).")

        vanilla_seed_dir = vanilla_dir / arch_name / f"seed_{seed}"
        final_path = find_weight_file(vanilla_seed_dir, ["best_model.pt", "final_model.pth"])
        print(f"  Searching for final model in: {vanilla_seed_dir}")
        if final_path:
            # Vanilla models were also trained with the LEGACY architecture
            all_weights.append(load_weights_from_path(final_path, legacy_model_class, device))
            all_labels.append("Vanilla (Nat) Final")
            print(f"    -> Found and loaded: {final_path.name}")
        else:
            print("    -> Not found.")

    # --- 3. Load MAML (PB-Trained) Weights ---
    print("\n--- Loading MAML (PB-Trained) Weights ---")
    pb_dir = Path(args.pb_results_dir) / args.architecture
    for seed in args.seeds:
        print(f"\n- Seed {seed}:")
        # These were trained with the NEW architecture
        set_seed(seed)
        initial_model = new_model_class().to(device)
        all_weights.append(get_model_weights(initial_model))
        all_labels.append("MAML (PB) Initial")
        print("  Created initial PB model weights (using new model class).")

        final_path = pb_dir / f"seed_{seed}" / "best_model.pt"
        print(f"  Checking for final model: {final_path}")
        if final_path.exists():
            # Load these weights using the NEW model class
            all_weights.append(load_weights_from_path(final_path, new_model_class, device))
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

    # --- Partition weights by architecture ---
    legacy_weights, legacy_labels = [], []
    new_arch_weights, new_arch_labels = [], []

    for w, l in zip(all_weights, all_labels):
        if "PB" in l:
            new_arch_weights.append(w)
            new_arch_labels.append(l)
        else:
            legacy_weights.append(w)
            legacy_labels.append(l)
            
    # --- Generate a plot for each architecture ---
    generate_pca_plot(legacy_weights, legacy_labels, arch_name, args.output_dir, "Legacy")
    generate_pca_plot(new_arch_weights, new_arch_labels, arch_name, args.output_dir, "New")

if __name__ == "__main__":
    main() 