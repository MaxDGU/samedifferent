#!/usr/bin/env python
# naturalistic/plot_weight_pca.py

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import OrderedDict

# Add project root to sys.path to allow importing 'baselines'
# Assumes the script is in 'naturalistic' directory, and 'baselines' is in parent.
try:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from baselines.models import ConvNet2LR, ConvNet4LR, ConvNet6LR
    MODEL_CLASSES = {
        'conv2lr': ConvNet2LR,
        'conv4lr': ConvNet4LR,
        'conv6lr': ConvNet6LR,
    }
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model classes from 'baselines.models': {e}", file=sys.stderr)
    print("Ensure 'baselines' directory is at the project root and contains model definitions (ConvNet2LR, etc.).", file=sys.stderr)
    print(f"Project root determined as: {project_root}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    # Define dummy classes if import fails, to prevent script from crashing immediately,
    # but it won't function correctly.
    class DummyModel:
        def __init__(self, *args, **kwargs): pass
        def eval(self): pass
        def state_dict(self): return {}
    MODEL_CLASSES = {'conv2lr': DummyModel, 'conv4lr': DummyModel, 'conv6lr': DummyModel}
    # It's better to exit if models can't be loaded, as the script is useless.
    # However, for tool execution flow, this might be problematic.
    # For now, it will try to run but likely fail at model instantiation.

def parse_args():
    parser = argparse.ArgumentParser(description="Generate PCA plots of initial and final model weights for MAML and Vanilla SGD.")
    parser.add_argument('--maml_log_dir_base', type=str, required=True, help='Base directory for MAML logs (e.g., /scratch/gpfs/mg7411/samedifferent/logs_naturalistic_meta).')
    parser.add_argument('--vanilla_log_dir_base', type=str, required=True, help='Base directory for Vanilla SGD logs (e.g., /scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla).')
    parser.add_argument('--output_dir', type=str, default='pca_weight_plots', help='Directory to save the PCA plots (default: pca_weight_plots).')
    
    default_architectures = ['conv2lr', 'conv4lr', 'conv6lr']
    parser.add_argument('--architectures', nargs='+', default=default_architectures, 
                        help=f'List of model architectures to process (default: {" ".join(default_architectures)}).')
    
    default_seeds = [42, 123, 789, 555, 999]
    parser.add_argument('--seeds', nargs='+', type=int, default=default_seeds,
                        help=f'List of seeds to process (default: {" ".join(map(str, default_seeds))}).')
    
    parser.add_argument('--num_input_channels', type=int, default=1, help='Number of input channels for the models (default: 1).')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes for the models (default: 2).')
    
    return parser.parse_args()

def get_initial_flattened_weights(model_class, seed, num_input_channels=1, num_classes=2):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # For consistent GPU initialization if model uses it
    np.random.seed(seed) # For any numpy-based randomness in model initialization

    # Instantiate the model (expected to be on CPU by default)
    model = model_class(channels_in=num_input_channels, num_classes=num_classes)
    model.eval() # Set to evaluation mode (e.g., for dropout, batchnorm)
    
    with torch.no_grad(): # Ensure no gradients are computed
        # Flatten the state dict values and concatenate them
        flat_weights = torch.cat([p.cpu().flatten() for p in model.state_dict().values()])
    return flat_weights

def load_and_flatten_weights(model_path):
    # Load checkpoint to CPU to avoid issues if the script is run on a machine without GPU
    # or with a different GPU setup than where the model was saved.
    checkpoint = torch.load(model_path, map_location='cpu')
    
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: # Some frameworks use 'state_dict'
            state_dict = checkpoint['state_dict']
        else:
            # If it's a dict but no known keys, assume the dict itself is the state_dict
            # This is less common for checkpoints but possible for direct state_dict saves
            state_dict = checkpoint 
    else:
        # If checkpoint is not a dict, assume it's the state_dict itself
        # (e.g., saved directly with torch.save(model.state_dict(), path))
        state_dict = checkpoint

    if state_dict is None:
        raise ValueError(f"Could not extract state_dict from checkpoint at {model_path}")

    # The MAML and Vanilla training scripts save model.module.state_dict(),
    # so the keys should already be clean (no 'module.' prefix).
    # If a raw DataParallel state_dict was saved, prefix removal would be needed:
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] if k.startswith('module.') else k 
    #     new_state_dict[name] = v
    # state_dict = new_state_dict
            
    flat_weights = torch.cat([p.cpu().flatten() for p in state_dict.values()])
    return flat_weights

def main():
    args = parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if MODEL_CLASSES was populated correctly (i.e., no DummyModel instances)
    if any(mc == DummyModel for mc in MODEL_CLASSES.values()): # Check class itself
        print("CRITICAL ERROR: Model classes were not imported correctly. Aborting.", file=sys.stderr)
        print("Please check the import logic for 'baselines.models' and ensure the path is correct.", file=sys.stderr)
        return 1 # Indicate an error

    for arch_name in args.architectures:
        if arch_name not in MODEL_CLASSES:
            print(f"WARNING: Architecture '{arch_name}' not found in MODEL_CLASSES. Skipping.")
            continue
        
        print(f"\nProcessing architecture: {arch_name.upper()}")
        model_class = MODEL_CLASSES[arch_name]
        if model_class == DummyModel: # Double check, if the earlier check passed but somehow it's still Dummy
             print(f"CRITICAL ERROR: Model class for {arch_name} is a DummyModel. Cannot proceed. Aborting.", file=sys.stderr)
             return 1


        all_weights_for_pca_list = []
        # Stores dicts: {'method': str, 'state': str, 'seed': int, 'arch': str}
        pca_point_metadata = [] 

        # --- Collect Initial MAML Weights ---
        for seed in args.seeds:
            print(f"  Getting initial MAML weights for seed {seed}...")
            try:
                weights = get_initial_flattened_weights(model_class, seed, args.num_input_channels, args.num_classes)
                all_weights_for_pca_list.append(weights.numpy())
                pca_point_metadata.append({'method': 'MAML', 'state': 'Initial', 'seed': seed, 'arch': arch_name})
            except Exception as e:
                print(f"    ERROR generating initial MAML weights for seed {seed}, arch {arch_name}: {e}")

        # --- Collect Final MAML Weights ---
        for seed in args.seeds:
            print(f"  Loading final MAML weights for seed {seed}...")
            maml_model_path = Path(args.maml_log_dir_base) / arch_name / f"seed_{seed}" / arch_name / f"seed_{seed}" / f"{arch_name}_best.pth"
            if maml_model_path.exists():
                try:
                    weights = load_and_flatten_weights(maml_model_path)
                    all_weights_for_pca_list.append(weights.numpy())
                    pca_point_metadata.append({'method': 'MAML', 'state': 'Final', 'seed': seed, 'arch': arch_name})
                except Exception as e:
                    print(f"    ERROR loading final MAML weights from {maml_model_path}: {e}")
            else:
                print(f"    WARNING: MAML model not found at {maml_model_path}")
        
        # --- Collect Initial Vanilla Weights ---
        for seed in args.seeds:
            print(f"  Getting initial Vanilla SGD weights for seed {seed}...")
            try:
                weights = get_initial_flattened_weights(model_class, seed, args.num_input_channels, args.num_classes)
                all_weights_for_pca_list.append(weights.numpy())
                pca_point_metadata.append({'method': 'Vanilla', 'state': 'Initial', 'seed': seed, 'arch': arch_name})
            except Exception as e:
                print(f"    ERROR generating initial Vanilla weights for seed {seed}, arch {arch_name}: {e}")

        # --- Collect Final Vanilla Weights ---
        for seed in args.seeds:
            print(f"  Loading final Vanilla SGD weights for seed {seed}...")
            vanilla_model_path = Path(args.vanilla_log_dir_base) / arch_name / f"seed_{seed}" / "best_model.pt"
            if vanilla_model_path.exists():
                try:
                    weights = load_and_flatten_weights(vanilla_model_path)
                    all_weights_for_pca_list.append(weights.numpy())
                    pca_point_metadata.append({'method': 'Vanilla', 'state': 'Final', 'seed': seed, 'arch': arch_name})
                except Exception as e:
                    print(f"    ERROR loading final Vanilla weights from {vanilla_model_path}: {e}")
            else:
                print(f"    WARNING: Vanilla SGD model not found at {vanilla_model_path}")

        if not all_weights_for_pca_list:
            print(f"  No weights collected for architecture {arch_name}. Skipping PCA plot.")
            continue

        weights_matrix = np.array(all_weights_for_pca_list)
        
        if weights_matrix.shape[0] < 2: # PCA needs at least 2 samples
            print(f"  Not enough weight samples ({weights_matrix.shape[0]}) for PCA for architecture {arch_name} (need at least 2). Skipping.")
            continue
        if weights_matrix.ndim != 2 or weights_matrix.shape[1] == 0:
             print(f"  Weight matrix for {arch_name} is not suitable for PCA (shape: {weights_matrix.shape}). Skipping.")
             continue

        print(f"  Performing PCA on {weights_matrix.shape[0]} weight vectors of dimension {weights_matrix.shape[1]}...")
        pca = PCA(n_components=2, random_state=42) # Ensure reproducibility of PCA
        try:
            transformed_weights = pca.fit_transform(weights_matrix)
        except ValueError as e: # Handles cases like n_samples < n_components
            print(f"  PCA failed for {arch_name}: {e}. Weight matrix shape: {weights_matrix.shape}")
            continue
        
        # --- Plotting ---
        plt.figure(figsize=(12, 10))
        colors = {'MAML': '#1f77b4', 'Vanilla': '#ff7f0e'} 
        markers = {'Initial': 'o', 'Final': 'x'}
        
        # To ensure unique legend entries
        plotted_legend_labels = {} 

        for i, meta in enumerate(pca_point_metadata):
            method = meta['method']
            state = meta['state']
            
            pc1 = transformed_weights[i, 0]
            pc2 = transformed_weights[i, 1]
            
            # Create a unique key for the legend for this combination of method and state
            label_key = f"{method} {state}"
            
            plt.scatter(pc1, pc2, 
                        color=colors[method], 
                        marker=markers[state], 
                        s=100, # marker size
                        alpha=0.7,
                        label=label_key if label_key not in plotted_legend_labels else None) # Add label only if it's new
            
            if label_key not in plotted_legend_labels:
                plotted_legend_labels[label_key] = True # Mark this label as plotted for legend purposes
        
        plt.title(f'PCA of Model Weights: {arch_name.upper()}', fontsize=16)
        plt.xlabel(f'Principal Component 1 (Explains {pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 (Explains {pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Create legend from unique labels automatically gathered by scatter
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles: 
            # Define desired order for legend items
            desired_order = ["MAML Initial", "MAML Final", "Vanilla Initial", "Vanilla Final"]
            ordered_handles = []
            ordered_labels = []
            label_to_handle_map = dict(zip(labels, handles))

            for lbl in desired_order:
                if lbl in label_to_handle_map:
                    ordered_labels.append(lbl)
                    ordered_handles.append(label_to_handle_map[lbl])
            
            # Add any other labels that might have been generated if not in desired_order
            # (e.g. if some data was missing and a category wasn't plotted)
            for lbl, hdl in label_to_handle_map.items():
                if lbl not in ordered_labels:
                    ordered_labels.append(lbl)
                    ordered_handles.append(hdl)

            plt.legend(ordered_handles, ordered_labels, title="Weight Group", fontsize=10, title_fontsize=12)
        
        plot_filename = output_path / f"pca_weights_{arch_name}.png"
        plt.tight_layout() # Adjust plot to ensure everything fits without overlapping
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free up memory
        print(f"  PCA plot saved to {plot_filename}")

    print("\nFinished generating PCA plots.")
    return 0

if __name__ == '__main__':
    # Ensure that if main returns an error code, it's propagated
    sys.exit(main()) 