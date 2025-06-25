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
import torch.nn.functional as F

# --- Custom Dataset Import ---
# This is a bit of a hack to ensure we can import from the parent `naturalistic` dir
try:
    from naturalistic.train_vanilla import NaturalisticDataset
except ImportError:
    print("Warning: Could not import NaturalisticDataset. A dummy class will be used.", file=sys.stderr)
    class NaturalisticDataset: pass # Dummy class to avoid crashing

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_weights(model):
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def load_weights_from_path(model_path, model_class, device):
    model = model_class().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    
    model.load_state_dict(state_dict, strict=False)
    return get_model_weights(model)

def find_weight_file(directory, possible_names):
    for name in possible_names:
        path = directory / name
        if path.exists():
            return path
    return None

def import_legacy_model_class(architecture):
    """Dynamically imports the LEGACY model class from the baselines.models module."""
    try:
        module_path = f"baselines.models.{architecture}"
        module = importlib.import_module(module_path)
        return module.SameDifferentCNN
    except ImportError as e:
        print(f"Error: Could not import legacy model for architecture '{architecture}'.\n{e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize model weight space using PCA for models with matching architectures.")
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'], help='Model architecture to visualize.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 555, 789, 999], help='List of random seeds to process.')
    
    parser.add_argument('--naturalistic_meta_dir', type=str, default="logs_naturalistic_meta", help="Directory for naturalistic MAML models.")
    parser.add_argument('--naturalistic_vanilla_dir', type=str, default="logs_naturalistic_vanilla", help="Directory for naturalistic Vanilla models.")
    parser.add_argument('--pb_dir', type=str, default="results/pb_retrained_legacy_conv6", help="Directory for the correctly retrained PB MAML models.")
    
    # New arguments for adaptation
    parser.add_argument('--naturalistic_data_dir', type=str, default="data/naturalistic_new/meta", help="Directory for the naturalistic dataset for adaptation.")
    parser.add_argument('--adaptation_lr', type=float, default=0.01, help="Learning rate for the adaptation process.")
    parser.add_argument('--adaptation_steps', type=int, default=5, help="Number of gradient steps for adaptation.")

    parser.add_argument('--output_dir', type=str, default="visualizations/final_pca", help="Directory to save the final PCA plot.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class = import_legacy_model_class(args.architecture)
    arch_name_lr = args.architecture + 'lr'

    # --- Load adaptation data once ---
    print(f"--- Loading Naturalistic data for adaptation from {args.naturalistic_data_dir} ---")
    try:
        adaptation_dataset = NaturalisticDataset(args.naturalistic_data_dir, num_tasks=10, num_examples=50) # Small subset for speed
        adaptation_loader = torch.utils.data.DataLoader(adaptation_dataset, batch_size=32, shuffle=True)
        adaptation_batch = next(iter(adaptation_loader))
        print("Adaptation data loaded successfully.")
    except Exception as e:
        print(f"CRITICAL: Could not load adaptation data. The new analysis will be skipped. Error: {e}")
        adaptation_batch = None

    all_weights = []
    all_labels = []

    # --- Load All Weights (they all use the same legacy architecture now) ---
    print("\n--- Loading All Model Weights (using Legacy Architecture) ---")
    for seed in args.seeds:
        print(f"\n- Seed {seed}:")
        set_seed(seed)
        initial_weights = get_model_weights(model_class().to(device))
        
        # 1. MAML (Naturalistic)
        final_nat_maml_path = Path(args.naturalistic_meta_dir) / arch_name_lr / f"seed_{seed}" / arch_name_lr / f"seed_{seed}" / f"{arch_name_lr}_best.pth"
        if final_nat_maml_path.exists():
            all_weights.extend([initial_weights, load_weights_from_path(final_nat_maml_path, model_class, device)])
            all_labels.extend(["MAML (Nat) Initial", "MAML (Nat) Final"])
            print(f"  Loaded MAML (Nat) initial (re-created) and final from {final_nat_maml_path}")
        else:
            print(f"  MAML (Nat) final weights not found at: {final_nat_maml_path}")

        # 2. Vanilla (Naturalistic)
        vanilla_seed_dir = Path(args.naturalistic_vanilla_dir) / arch_name_lr / f"seed_{seed}"
        final_vanilla_path = find_weight_file(vanilla_seed_dir, ["best_model.pt", "final_model.pth"])
        if final_vanilla_path:
            all_weights.extend([initial_weights, load_weights_from_path(final_vanilla_path, model_class, device)])
            all_labels.extend(["Vanilla (Nat) Initial", "Vanilla (Nat) Final"])
            print(f"  Loaded Vanilla (Nat) initial (re-created) and final from {final_vanilla_path}")
        else:
            print(f"  Vanilla (Nat) final weights not found in: {vanilla_seed_dir}")
            
        # 3. MAML (PB) and On-The-Fly Adaptation
        pb_seed_dir = Path(args.pb_dir) / args.architecture / f"seed_{seed}"
        final_pb_path = find_weight_file(pb_seed_dir, ["best_model.pt"])
        if final_pb_path:
            # Load the base PB-trained model
            pb_model = model_class().to(device)
            pb_model.load_state_dict(torch.load(final_pb_path, map_location=device).get('model_state_dict'), strict=False)
            
            all_weights.extend([initial_weights, get_model_weights(pb_model)])
            all_labels.extend(["MAML (PB) Initial", "MAML (PB) Final"])
            print(f"  Loaded MAML (PB) initial (re-created) and final from {final_pb_path}")

            # Perform adaptation if data is available
            if adaptation_batch:
                print("    -> Adapting MAML (PB) on Naturalistic data...")
                learner = pb_model.clone()
                optimizer = torch.optim.SGD(learner.parameters(), lr=args.adaptation_lr)
                
                images, labels = adaptation_batch
                images, labels = images.to(device), labels.to(device).float()

                for step in range(args.adaptation_steps):
                    optimizer.zero_grad()
                    preds = learner(images)
                    loss = F.binary_cross_entropy_with_logits(preds.squeeze(), labels)
                    loss.backward()
                    optimizer.step()
                
                all_weights.append(get_model_weights(learner))
                all_labels.append("MAML (PB->Nat) Adapted")
                print(f"    -> Adaptation complete. Final loss: {loss.item():.4f}")

        else:
            print(f"  MAML (PB) final weights not found in: {pb_seed_dir}")

    # --- PCA and Plotting ---
    if not all_weights:
        print("\nNo weights were loaded. Cannot generate plot.")
        return
        
    print(f"\n--- Performing PCA on {len(all_weights)} total weight vectors ---")
    pca = PCA(n_components=2)
    projected_weights = pca.fit_transform(np.vstack(all_weights))

    fig, ax = plt.subplots(figsize=(14, 12))
    
    styles = {
        "MAML (PB) Initial": {"marker": "x", "color": "#ff7f0e", "label": "MAML (PB) Initial"},
        "MAML (PB) Final": {"marker": "P", "color": "#ff7f0e", "label": "MAML (PB) Final"},
        "MAML (PB->Nat) Adapted": {"marker": "*", "color": "#ff7f0e", "label": "MAML (PB->Nat) Adapted"},
        "MAML (Nat) Initial": {"marker": "x", "color": "#1f77b4", "label": "MAML (Nat) Initial"},
        "MAML (Nat) Final": {"marker": "o", "color": "#1f77b4", "label": "MAML (Nat) Final"},
        "Vanilla (Nat) Initial": {"marker": "x", "color": "#1f77b4", "label": "Vanilla (Nat) Initial"},
        "Vanilla (Nat) Final": {"marker": "s", "color": "#2ca02c", "label": "Vanilla (Nat) Final"},
    }

    for i, label in enumerate(all_labels):
        style = styles.get(label)
        if style:
            ax.scatter(projected_weights[i, 0], projected_weights[i, 1], **style, s=150, alpha=0.8, zorder=5)

    # Add lines connecting initial to final states
    for i in range(0, len(projected_weights), 2):
        if i + 1 < len(projected_weights):
            start_label = all_labels[i]
            end_label = all_labels[i+1]
            if start_label.replace("Initial", "") == end_label.replace("Final", ""):
                style = styles.get(start_label, styles.get(end_label))
                ax.plot([projected_weights[i, 0], projected_weights[i+1, 0]],
                        [projected_weights[i, 1], projected_weights[i+1, 1]],
                        color=style["color"], linestyle='--', alpha=0.5, zorder=1)

    ax.set_title(f"PCA of Model Weights for {args.architecture.upper()} Architecture", fontsize=18, pad=20)
    ax.set_xlabel("Principal Component 1", fontsize=14)
    ax.set_ylabel("Principal Component 2", fontsize=14)
    
    legend_elements = [Line2D([0], [0], marker=s["marker"], color='w', label=s["label"], markerfacecolor=s["color"], markersize=12) for s in styles.values()]
    ax.legend(handles=legend_elements, title="Weight States", loc="best", fontsize=12, title_fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_path = Path(args.output_dir) / f"pca_weights_{arch_name_lr}_unified.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"\nUnified PCA plot saved to {output_path}")

if __name__ == "__main__":
    main() 