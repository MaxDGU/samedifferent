import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- Setup Project Path ---
# This is a bit of a hack to make sure we can import the necessary modules
# without having to install the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# --- Model Imports ---
# Now we can import the models from the project
from meta_baseline.models import Conv2LR, Conv4LR, Conv6LR

# --- Constants ---
ARCHS = ['conv2lr', 'conv4lr', 'conv6lr']
SEEDS = [123, 456, 789, 555, 999]

def get_model_from_arch(arch):
    """Returns the model class constructor for a given architecture string."""
    if arch == 'conv2lr':
        return Conv2LR()
    elif arch == 'conv4lr':
        return Conv4LR()
    elif arch == 'conv6lr':
        return Conv6LR()
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def flatten_weights(model):
    """Flattens all parameters of a model into a single numpy array."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def load_initial_pb_weights(arch):
    """Load initial weights from a pre-trained PB model."""
    model = get_model_from_arch(arch)
    # This path is based on where the PB-trained models were saved in previous steps.
    path = PROJECT_ROOT / f"maml_pbweights_conv6/{arch}_pb_model.pth"
    if not path.exists():
        print(f"Warning: Initial PB model not found at {path}")
        return None
    try:
        # The PB models were saved as entire model files, not state_dicts
        model = torch.load(path, map_location=torch.device('cpu'))
        return flatten_weights(model)
    except Exception as e:
        print(f"Warning: Could not load initial PB model from {path}. Error: {e}")
        return None

def load_fully_trained_vanilla_weights(arch, seed):
    """Load final weights of a vanilla model fully trained on naturalistic data."""
    model = get_model_from_arch(arch)
    path = PROJECT_ROOT / f"logs_naturalistic_vanilla/{arch}/seed_{seed}/final_model.pth"
    if not path.exists():
        print(f"Warning: Fully trained vanilla model not found at {path}")
        return None
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return flatten_weights(model)
    except Exception as e:
        print(f"Warning: Could not load fully trained vanilla model from {path}. Error: {e}")
        return None

def load_maml_adapted_weights(arch, seed):
    """Load weights from a MAML-adapted model after fine-tuning on naturalistic data."""
    model = get_model_from_arch(arch)
    path = PROJECT_ROOT / f"results_naturalistic_meta_test/{arch}/seed_{seed}/adapted_model.pth"
    if not path.exists():
        print(f"Warning: MAML-adapted model not found at {path}")
        return None
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        return flatten_weights(model)
    except Exception as e:
        print(f"Warning: Could not load MAML-adapted model from {path}. Error: {e}")
        return None

def load_maml_naturalistic_trained_weights(arch, seed):
    """Load final weights of a MAML model trained on naturalistic data."""
    model = get_model_from_arch(arch)
    # This path points to the logs from the MAML training on the naturalistic dataset
    path = PROJECT_ROOT / f"logs_naturalistic_meta/{arch}/seed_{seed}/final_model.pt"
    if not path.exists():
        print(f"Warning: MAML naturalistic trained model not found at {path}")
        return None
    try:
        # These checkpoints contain the model's state_dict
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        return flatten_weights(model)
    except Exception as e:
        print(f"Warning: Could not load MAML naturalistic trained model from {path}. Error: {e}")
        return None

def load_vanilla_adapted_weights(arch, seed):
    """
    Loads initial and final weights for a randomly initialized model
    adapted on the naturalistic task.
    """
    model = get_model_from_arch(arch)
    initial_path = PROJECT_ROOT / f"logs_naturalistic_vanilla/{arch}/seed_{seed}/initial_model.pth"
    adapted_path = PROJECT_ROOT / f"logs_naturalistic_vanilla/{arch}/seed_{seed}/adapted_model.pth"

    if not initial_path.exists() or not adapted_path.exists():
        print(f"Warning: Vanilla adaptation weights not found for {arch} seed {seed}")
        return None, None
    try:
        model.load_state_dict(torch.load(initial_path, map_location=torch.device('cpu')))
        initial_weights = flatten_weights(model)

        model.load_state_dict(torch.load(adapted_path, map_location=torch.device('cpu')))
        adapted_weights = flatten_weights(model)
        return initial_weights, adapted_weights
    except Exception as e:
        print(f"Warning: Could not load vanilla adaptation weights for {arch} seed {seed}. Error: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Visualize adaptation vectors using PCA.")
    parser.add_argument('--output_dir', type=str, default='visualizations/adaptation_pca',
                        help='Directory to save the PCA plot.')
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_adaptation_vectors = []
    meta_adaptation_vectors = []
    vanilla_adaptation_vectors = []

    print("Loading weights and calculating adaptation vectors...")
    for arch in ARCHS:
        initial_pb_weights = load_initial_pb_weights(arch)
        if initial_pb_weights is None:
            print(f"Skipping arch {arch} due to missing PB weights.")
            continue

        for seed in SEEDS:
            # 1. Meta-adaptation vectors
            maml_adapted = load_maml_adapted_weights(arch, seed)
            if maml_adapted is not None:
                meta_vec = maml_adapted - initial_pb_weights
                meta_adaptation_vectors.append(meta_vec)

            # 2. Vanilla-adaptation vectors
            vanilla_initial, vanilla_adapted = load_vanilla_adapted_weights(arch, seed)
            if vanilla_initial is not None and vanilla_adapted is not None:
                vanilla_vec = vanilla_adapted - vanilla_initial
                vanilla_adaptation_vectors.append(vanilla_vec)

    if not meta_adaptation_vectors and not vanilla_adaptation_vectors:
        print("No adaptation vectors could be calculated. Exiting.")
        return

    all_adaptation_vectors = vanilla_adaptation_vectors + meta_adaptation_vectors

    # --- PCA Fitting and Transformation ---
    print("Performing PCA...")
    pca = PCA(n_components=2)
    pca.fit(all_adaptation_vectors)

    transformed_vanilla_vectors = pca.transform(vanilla_adaptation_vectors)
    transformed_meta_vectors = pca.transform(meta_adaptation_vectors)

    # Now, load the MAML-Naturalistic trained models and project them
    all_maml_nat_vectors = []
    for arch in ARCHS:
        initial_pb_weights = load_initial_pb_weights(arch)
        if initial_pb_weights is None:
            continue

        for seed in SEEDS:
            maml_nat_weights = load_maml_naturalistic_trained_weights(arch, seed)
            if maml_nat_weights is not None:
                # Vector from the common PB start point to the MAML-Nat final point
                vector = maml_nat_weights - initial_pb_weights
                all_maml_nat_vectors.append(vector)

    transformed_maml_nat_points = None
    if all_maml_nat_vectors:
        transformed_maml_nat_points = pca.transform(all_maml_nat_vectors)

    # --- Plotting ---
    print("Generating plot...")
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot Vanilla Adaptation vectors (blue)
    for i in range(len(transformed_vanilla_vectors)):
        ax.arrow(0, 0, transformed_vanilla_vectors[i, 0], transformed_vanilla_vectors[i, 1],
                 head_width=0.3, head_length=0.5, fc='royalblue', ec='royalblue', length_includes_head=True, alpha=0.7)

    # Plot Meta Adaptation vectors (orange)
    for i in range(len(transformed_meta_vectors)):
        ax.arrow(0, 0, transformed_meta_vectors[i, 0], transformed_meta_vectors[i, 1],
                 head_width=0.3, head_length=0.5, fc='darkorange', ec='darkorange', length_includes_head=True, alpha=0.7)

    # Plot the projected MAML-Naturalistic trained weights if they exist
    if transformed_maml_nat_points is not None:
        ax.scatter(transformed_maml_nat_points[:, 0], transformed_maml_nat_points[:, 1],
                   c='red', marker='*', s=150, label='MAML-Nat Final Endpoint', zorder=5, edgecolors='black')

    # Plot shared start point
    ax.plot(0, 0, 'o', markersize=12, color='black', label='Shared Start Point', zorder=6)

    # --- Legend and Labels ---
    vanilla_patch = mpatches.Patch(color='royalblue', label='Vanilla Adaptation', alpha=0.7)
    meta_patch = mpatches.Patch(color='darkorange', label='Meta Adaptation', alpha=0.7)
    start_point = plt.Line2D([0], [0], marker='o', color='w', label='Shared Start Point',
                             markerfacecolor='black', markersize=10)
    
    handles = [vanilla_patch, meta_patch, start_point]
    if transformed_maml_nat_points is not None:
        maml_nat_handle = plt.Line2D([], [], marker='*', color='red', label='MAML-Nat Final Endpoint',
                                     linestyle='None', markersize=12, markeredgecolor='black')
        handles.insert(2, maml_nat_handle)

    ax.legend(handles=handles)
    ax.grid(True)
    ax.set_title("PCA of Adaptation Vectors from a Common Origin", fontsize=16)
    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"Principal Component of Adaptation 1 ({pc1_var:.2f}%)", fontsize=12)
    ax.set_ylabel(f"Principal Component of Adaptation 2 ({pc2_var:.2f}%)", fontsize=12)

    # Save the figure
    output_path = output_dir / "pca_adaptation_vectors_with_maml_nat_endpoints.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main() 