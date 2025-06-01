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

# Define DummyModel at the top level so it's always available
class DummyModel:
    def __init__(self, *args, **kwargs): pass
    def eval(self): pass
    def state_dict(self): return {}

# Add project root to sys.path to allow importing 'baselines'
# Assumes the script is in 'naturalistic' directory, and 'baselines' is in parent.
try:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Use conv2, conv4, conv6 as per user confirmation
    from baselines.models import conv2, conv4, conv6
    MODEL_CLASSES = {
        'conv2lr': conv2,
        'conv4lr': conv4,
        'conv6lr': conv6,
    }
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model classes from 'baselines.models': {e}", file=sys.stderr)
    print("Ensure 'baselines' directory is at the project root and contains model definitions (conv2, conv4, conv6).", file=sys.stderr)
    try:
        print(f"Project root determined as: {project_root}", file=sys.stderr)
    except NameError:
        print("Project root could not be determined.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    MODEL_CLASSES = {'conv2lr': DummyModel, 'conv4lr': DummyModel, 'conv6lr': DummyModel}
    print("Fell back to using DummyModel instances due to import error.", file=sys.stderr)

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

    model = model_class(channels_in=num_input_channels, num_classes=num_classes)
    model.eval() 
    
    with torch.no_grad(): 
        flat_weights = torch.cat([p.cpu().flatten() for p in model.state_dict().values()])
    return flat_weights.numpy()

def load_and_flatten_weights(model_path):
    # Revert to weights_only=False as checkpoints contain argparse.Namespace
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False) 
    
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint: 
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'): # MAML saves model object
             model_loaded = checkpoint['model']
             # If model was saved with DataParallel, keys might have 'module.' prefix
             # Creating a new OrderedDict ensures we handle this potential prefix correctly,
             # though if your training scripts already saved model.module.state_dict(), it's not strictly needed.
             new_state_dict = OrderedDict()
             for k, v in model_loaded.state_dict().items():
                 name = k[7:] if k.startswith('module.') else k 
                 new_state_dict[name] = v
             state_dict = new_state_dict
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()): 
            # If it's a dict of tensors, assume it IS the state_dict
            # This is less likely for your MAML/Vanilla checkpoints but good to have
            state_dict = checkpoint
        else:
            # Fallback: if the checkpoint is a dict but no known keys, assume it is the state_dict itself.
            # This might happen if torch.save(model.state_dict(), path) was used directly
            # and the state_dict itself was a plain dict (not OrderedDict).
            # Check if all values are tensors to be safer.
            if all(isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in checkpoint.values()):
                 state_dict = checkpoint
            else:
                raise ValueError(f"Checkpoint at {model_path} is a dict but not a recognized state_dict format or simple model wrapper.")
    elif isinstance(checkpoint, OrderedDict) and all(isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in checkpoint.values()):
        # Directly saved state_dict (often an OrderedDict)
        state_dict = checkpoint
    elif hasattr(checkpoint, 'state_dict'): # If the checkpoint is the model object itself
        # This can happen if torch.save(model, path) was used
        new_state_dict = OrderedDict()
        for k, v in checkpoint.state_dict().items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        raise ValueError(f"Could not extract state_dict from checkpoint at {model_path}. Loaded object type: {type(checkpoint)}")

    if state_dict is None:
        raise ValueError(f"State_dict is None after attempting to load from {model_path}")
            
    flat_weights = torch.cat([p.cpu().flatten() for p in state_dict.values()])
    return flat_weights.numpy()

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
        # Store initial and final PCA points to draw lines later
        # Structure: {(method, seed): {'initial': (pc1, pc2), 'final': (pc1, pc2)}}
        trajectory_points = {} 

        # --- Collect Initial MAML Weights ---
        for seed in args.seeds:
            print(f"  Getting initial MAML weights for seed {seed}...")
            try:
                weights = get_initial_flattened_weights(model_class, seed, args.num_input_channels, args.num_classes)
                all_weights_for_pca_list.append(weights)
                pca_point_metadata.append({'method': 'MAML', 'state': 'Initial', 'seed': seed, 'arch': arch_name})
                if ('MAML', seed) not in trajectory_points: trajectory_points[('MAML', seed)] = {}
                # Temporarily store raw weights; will be replaced by PCA coords later
                trajectory_points[('MAML', seed)]['initial_raw'] = weights 
            except Exception as e:
                print(f"    ERROR generating initial MAML weights for seed {seed}, arch {arch_name}: {e}")

        # --- Collect Final MAML Weights ---
        for seed in args.seeds:
            print(f"  Loading final MAML weights for seed {seed}...")
            maml_model_path = Path(args.maml_log_dir_base) / arch_name / f"seed_{seed}" / arch_name / f"seed_{seed}" / f"{arch_name}_best.pth"
            if maml_model_path.exists():
                try:
                    weights = load_and_flatten_weights(maml_model_path)
                    all_weights_for_pca_list.append(weights)
                    pca_point_metadata.append({'method': 'MAML', 'state': 'Final', 'seed': seed, 'arch': arch_name})
                    if ('MAML', seed) not in trajectory_points: trajectory_points[('MAML', seed)] = {}
                    trajectory_points[('MAML', seed)]['final_raw'] = weights
                except Exception as e:
                    print(f"    ERROR loading final MAML weights from {maml_model_path}: {e}")
            else:
                print(f"    WARNING: MAML model not found at {maml_model_path}")
        
        # --- Collect Initial Vanilla Weights ---
        for seed in args.seeds:
            print(f"  Getting initial Vanilla SGD weights for seed {seed}...")
            try:
                weights = get_initial_flattened_weights(model_class, seed, args.num_input_channels, args.num_classes)
                all_weights_for_pca_list.append(weights)
                pca_point_metadata.append({'method': 'Vanilla', 'state': 'Initial', 'seed': seed, 'arch': arch_name})
                if ('Vanilla', seed) not in trajectory_points: trajectory_points[('Vanilla', seed)] = {}
                trajectory_points[('Vanilla', seed)]['initial_raw'] = weights
            except Exception as e:
                print(f"    ERROR generating initial Vanilla weights for seed {seed}, arch {arch_name}: {e}")

        # --- Collect Final Vanilla Weights ---
        for seed in args.seeds:
            print(f"  Loading final Vanilla SGD weights for seed {seed}...")
            vanilla_model_path = Path(args.vanilla_log_dir_base) / arch_name / f"seed_{seed}" / "best_model.pt"
            if vanilla_model_path.exists():
                try:
                    weights = load_and_flatten_weights(vanilla_model_path)
                    all_weights_for_pca_list.append(weights)
                    pca_point_metadata.append({'method': 'Vanilla', 'state': 'Final', 'seed': seed, 'arch': arch_name})
                    if ('Vanilla', seed) not in trajectory_points: trajectory_points[('Vanilla', seed)] = {}
                    trajectory_points[('Vanilla', seed)]['final_raw'] = weights
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
        
        # Populate trajectory_points with PCA coordinates
        current_pca_idx = 0
        for meta_idx, meta in enumerate(pca_point_metadata):
            method_seed_key = (meta['method'], meta['seed'])
            state_key_pca = 'initial' if meta['state'] == 'Initial' else 'final' # To store PCA coords
            
            # Check if this point corresponds to one we stored raw weights for
            # This check is to ensure we only try to get PCA for points that were actually loaded
            # And that we don't go out of bounds if some raw weights were missing
            if method_seed_key in trajectory_points and f'{state_key_pca}_raw' in trajectory_points[method_seed_key]:
                if current_pca_idx < len(transformed_weights):
                    trajectory_points[method_seed_key][state_key_pca] = (transformed_weights[current_pca_idx, 0], transformed_weights[current_pca_idx, 1])
                    current_pca_idx += 1
                else:
                    print(f"    WARNING: Ran out of transformed_weights for {method_seed_key}, {state_key_pca}. Skipping for trajectory line.")
            
        # --- Plotting ---
        plt.figure(figsize=(14, 10)) # Slightly wider for lines
        colors = {'MAML': '#1f77b4', 'Vanilla': '#ff7f0e'} 
        markers = {'Initial': 'o', 'Final': 'x'}
        line_styles = {'MAML': ':', 'Vanilla': '--'} # Different line styles for MAML and Vanilla trajectories
        
        plotted_legend_labels = {} 

        # First, plot all scatter points
        # Iterate through pca_point_metadata to ensure correct association with transformed_weights
        scatter_idx = 0
        for meta in pca_point_metadata:
            method = meta['method']
            state = meta['state']
            
            # Check if this point was successfully transformed by PCA
            # (This implies it was successfully loaded initially)
            method_seed_key = (meta['method'], meta['seed'])
            state_key_pca = 'initial' if meta['state'] == 'Initial' else 'final'
            
            if method_seed_key in trajectory_points and state_key_pca in trajectory_points[method_seed_key]:
                pc1, pc2 = trajectory_points[method_seed_key][state_key_pca]
                label_key = f"{method} {state}"
                
                plt.scatter(pc1, pc2, 
                            color=colors[method], 
                            marker=markers[state], 
                            s=120, # marker size
                            alpha=0.8,
                            label=label_key if label_key not in plotted_legend_labels else None, 
                            zorder=3) # Ensure points are above lines
                
                if label_key not in plotted_legend_labels:
                    plotted_legend_labels[label_key] = True
            else:
                # This case should ideally not happen if PCA was successful and trajectory_points was populated correctly
                # But as a fallback, try to use scatter_idx if this point did contribute to PCA
                # This path is less robust, relying on matching order.
                if scatter_idx < len(transformed_weights):
                    pc1 = transformed_weights[scatter_idx, 0]
                    pc2 = transformed_weights[scatter_idx, 1]
                    label_key = f"{method} {state}" # May create duplicate legend items if not careful
                    plt.scatter(pc1, pc2, color=colors[method], marker=markers[state], s=120, alpha=0.8, label=label_key if label_key not in plotted_legend_labels else None, zorder=3)
                    if label_key not in plotted_legend_labels: plotted_legend_labels[label_key] = True
                scatter_idx +=1 # Increment only if used
            

        # Then, draw trajectory lines
        for (method, seed), points in trajectory_points.items():
            if 'initial' in points and 'final' in points:
                initial_pt = points['initial']
                final_pt = points['final']
                plt.plot([initial_pt[0], final_pt[0]], [initial_pt[1], final_pt[1]],
                         linestyle=line_styles[method],
                         color=colors[method],
                         alpha=0.5, 
                         linewidth=1.5,
                         zorder=2) # Lines behind points
        
        plt.title(f'PCA of Model Weights: {arch_name.upper()}', fontsize=16)
        plt.xlabel(f'Principal Component 1 (Explains {pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 (Explains {pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles: 
            desired_order = ["MAML Initial", "MAML Final", "Vanilla Initial", "Vanilla Final"]
            ordered_handles = []
            ordered_labels = []
            label_to_handle_map = dict(zip(labels, handles))

            for lbl in desired_order:
                if lbl in label_to_handle_map:
                    ordered_labels.append(lbl)
                    ordered_handles.append(label_to_handle_map[lbl])
            
            for lbl, hdl in label_to_handle_map.items():
                if lbl not in ordered_labels:
                    ordered_labels.append(lbl)
                    ordered_handles.append(hdl)

            plt.legend(ordered_handles, ordered_labels, title="Weight Group", fontsize=10, title_fontsize=12, loc='best')
        
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