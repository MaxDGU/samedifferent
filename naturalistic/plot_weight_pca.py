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
    # Correctly import the aliased class names from baselines.models
    from baselines.models import Conv2CNN, Conv4CNN, Conv6CNN 
    MODEL_CLASSES = {
        'conv2lr': Conv2CNN,
        'conv4lr': Conv4CNN,
        'conv6lr': Conv6CNN,
    }
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model classes from 'baselines.models': {e}", file=sys.stderr)
    print("Ensure 'baselines.models/__init__.py' exports Conv2CNN, Conv4CNN, Conv6CNN.", file=sys.stderr)
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
    print(f"    Attempting to generate initial weights for model class {model_class.__name__} with seed {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    try:
        model = model_class(channels_in=num_input_channels, num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            flat_weights = torch.cat([p.cpu().flatten() for p in model.state_dict().values()])
        print(f"    Successfully generated initial weights on the fly.")
        return flat_weights.numpy()
    except Exception as e:
        print(f"    ERROR during on-the-fly initial weight generation for {model_class.__name__}, seed {seed}: {e}")
        raise # Re-raise the exception to be caught by the main loop

def load_and_flatten_weights(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
            model_loaded = checkpoint['model']
            new_state_dict = OrderedDict()
            for k, v in model_loaded.state_dict().items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif all(isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in checkpoint.values()): 
            state_dict = checkpoint
        else:
            # Check if it's the args Namespace issue for MAML initial models
            if 'args' in checkpoint and isinstance(checkpoint['args'], argparse.Namespace) and 'model_state_dict' not in checkpoint and 'state_dict' not in checkpoint:
                 # This case might be an older MAML initial save that saved the entire training script's checkpoint
                 # We need to find the actual model weights within this structure if possible,
                 # or this format is not directly usable for just weights.
                 # For now, assume this is not the primary path for initial_model.pth if it *only* contains weights.
                 # If initial_model.pth *does* follow this complex structure, this logic needs enhancement.
                 print(f"    WARNING: Checkpoint at {model_path} contains 'args' but no clear state_dict. Trying to find nested model data.")
                 # Add specific logic here if you know how the model is nested in this case.
                 # For example, if it was checkpoint['learner'].module.state_dict()
                 if hasattr(checkpoint.get('learner'), 'module') and hasattr(checkpoint['learner'].module, 'state_dict'):
                     state_dict = checkpoint['learner'].module.state_dict()
                 elif hasattr(checkpoint.get('learner'), 'state_dict'):
                     state_dict = checkpoint['learner'].state_dict()
                 else:
                     raise ValueError(f"Checkpoint at {model_path} is a complex dict with 'args' and no recognized model state_dict path.")
            else:
                raise ValueError(f"Checkpoint at {model_path} is a dict but not a recognized state_dict format or simple model wrapper.")
    elif isinstance(checkpoint, OrderedDict) and all(isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in checkpoint.values()):
        state_dict = checkpoint
    elif hasattr(checkpoint, 'state_dict'): 
        new_state_dict = OrderedDict()
        for k, v in checkpoint.state_dict().items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        # This case could be a raw state_dict saved directly that isn't an OrderedDict
        # Let's assume if it's not a dict and not an OrderedDict, but has values that are tensors, it *is* the state_dict
        try:
            if all(isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in checkpoint.values()):
                 state_dict = checkpoint
            else:
                 raise ValueError(f"Could not extract state_dict from checkpoint at {model_path}. Loaded object type: {type(checkpoint)} and values are not all tensors.")
        except AttributeError: # if checkpoint doesn't have .values()
             raise ValueError(f"Could not extract state_dict from checkpoint at {model_path}. Loaded object type: {type(checkpoint)} is not a dict-like structure.")


    if state_dict is None:
        raise ValueError(f"State_dict is None after attempting to load from {model_path}")
            
    flat_weights = torch.cat([p.cpu().flatten() for p in state_dict.values()])
    return flat_weights.numpy()

def main():
    args = parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if MODEL_CLASSES was populated correctly (i.e., no DummyModel instances)
    if any(isinstance(MODEL_CLASSES.get(arch_name), type) and issubclass(MODEL_CLASSES.get(arch_name), DummyModel) for arch_name in args.architectures):
        print("CRITICAL ERROR: Model classes were not imported correctly (DummyModel found). Aborting.", file=sys.stderr)
        return 1

    loaded_initial_maml_weights = {}

    for arch_name in args.architectures:
        print(f"\nProcessing architecture: {arch_name.upper()}")
        model_class = MODEL_CLASSES.get(arch_name)
        if model_class == DummyModel or model_class is None:
             print(f"CRITICAL ERROR: Model class for {arch_name} is a DummyModel or missing. Cannot proceed. Aborting.", file=sys.stderr)
             return 1

        current_arch_weights_list = []
        current_arch_pca_metadata = []
        current_arch_trajectory_points = {}

        for seed in args.seeds:
            print(f"  Loading initial MAML weights for arch {arch_name}, seed {seed}...")
            initial_maml_path = Path(args.maml_log_dir_base) / arch_name / f"seed_{seed}" / arch_name / f"seed_{seed}" / "initial_model.pth"
            if initial_maml_path.exists():
                try:
                    weights = load_and_flatten_weights(initial_maml_path)
                    current_arch_weights_list.append(weights)
                    current_arch_pca_metadata.append({'method': 'MAML', 'state': 'Initial', 'seed': seed, 'arch': arch_name})
                    loaded_initial_maml_weights[(arch_name, seed)] = weights
                    if ('MAML', seed) not in current_arch_trajectory_points: current_arch_trajectory_points[('MAML', seed)] = {}
                    current_arch_trajectory_points[('MAML', seed)]['initial_raw'] = weights
                    print(f"    Successfully loaded initial MAML weights from {initial_maml_path}")
                except Exception as e:
                    print(f"    ERROR loading initial MAML weights from {initial_maml_path}: {e}")
            else:
                print(f"    WARNING: Initial MAML model not found at {initial_maml_path}.")

        for seed in args.seeds:
            print(f"  Loading final MAML weights for arch {arch_name}, seed {seed}...")
            final_maml_path = Path(args.maml_log_dir_base) / arch_name / f"seed_{seed}" / arch_name / f"seed_{seed}" / f"{arch_name}_best.pth"
            if final_maml_path.exists():
                try:
                    weights = load_and_flatten_weights(final_maml_path)
                    current_arch_weights_list.append(weights)
                    current_arch_pca_metadata.append({'method': 'MAML', 'state': 'Final', 'seed': seed, 'arch': arch_name})
                    if ('MAML', seed) not in current_arch_trajectory_points: current_arch_trajectory_points[('MAML', seed)] = {}
                    current_arch_trajectory_points[('MAML', seed)]['final_raw'] = weights
                    print(f"    Successfully loaded final MAML weights from {final_maml_path}")
                except Exception as e:
                    print(f"    ERROR loading final MAML weights from {final_maml_path}: {e}")
            else:
                print(f"    WARNING: Final MAML model not found at {final_maml_path}")
        
        for seed in args.seeds:
            print(f"  Assigning initial Vanilla SGD weights for arch {arch_name}, seed {seed} (from MAML initial)...")
            if (arch_name, seed) in loaded_initial_maml_weights:
                weights = loaded_initial_maml_weights[(arch_name, seed)]
                current_arch_weights_list.append(weights)
                current_arch_pca_metadata.append({'method': 'Vanilla', 'state': 'Initial', 'seed': seed, 'arch': arch_name})
                if ('Vanilla', seed) not in current_arch_trajectory_points: current_arch_trajectory_points[('Vanilla', seed)] = {}
                current_arch_trajectory_points[('Vanilla', seed)]['initial_raw'] = weights
                print(f"    Successfully assigned initial Vanilla weights (from MAML seed {seed}).")
            else:
                print(f"    WARNING: Initial MAML weights for arch {arch_name}, seed {seed} were not loaded. Cannot assign initial Vanilla weights.")

        for seed in args.seeds:
            print(f"  Loading final Vanilla SGD weights for arch {arch_name}, seed {seed}...")
            final_vanilla_path = Path(args.vanilla_log_dir_base) / arch_name / f"seed_{seed}" / "best_model.pt"
            if final_vanilla_path.exists():
                try:
                    weights = load_and_flatten_weights(final_vanilla_path)
                    current_arch_weights_list.append(weights)
                    current_arch_pca_metadata.append({'method': 'Vanilla', 'state': 'Final', 'seed': seed, 'arch': arch_name})
                    if ('Vanilla', seed) not in current_arch_trajectory_points: current_arch_trajectory_points[('Vanilla', seed)] = {}
                    current_arch_trajectory_points[('Vanilla', seed)]['final_raw'] = weights
                    print(f"    Successfully loaded final Vanilla weights from {final_vanilla_path}")
                except Exception as e:
                    print(f"    ERROR loading final Vanilla weights from {final_vanilla_path}: {e}")
            else:
                print(f"    WARNING: Final Vanilla SGD model not found at {final_vanilla_path}")

        if not current_arch_weights_list:
            print(f"  No weights collected for architecture {arch_name}. Skipping PCA plot.")
            continue

        try:
            weights_matrix = np.array(current_arch_weights_list)
            if weights_matrix.size == 0 and len(current_arch_weights_list) > 0:
                 print(f"  WARNING: weights_matrix is empty for {arch_name} even though list was not. This indicates all appends failed or list contained non-convertible types.")
                 continue
        except ValueError as e:
            print(f"  ERROR creating numpy array from weights list for {arch_name}: {e}")
            # Debugging: print types and shapes if ValueError occurs due to inhomogeneous list
            if any(not isinstance(item, np.ndarray) for item in current_arch_weights_list):
                print("    Detected non-NumPy array items in current_arch_weights_list:")
                for i, item in enumerate(current_arch_weights_list):
                    print(f"      Item {i} type: {type(item)}")
            else: # All are ndarrays, but shapes might differ
                print("    All items are NumPy arrays, checking shapes for inhomogeneity:")
                first_shape = current_arch_weights_list[0].shape if current_arch_weights_list else None
                for i, item in enumerate(current_arch_weights_list):
                    if item.shape != first_shape:
                        print(f"      Item {i} shape: {item.shape} (DIFFERENT from first shape {first_shape})")
                    # else: print(f"      Item {i} shape: {item.shape}") # optionally print all shapes
            continue
        
        if weights_matrix.shape[0] < 2: 
            print(f"  Not enough weight samples ({weights_matrix.shape[0]}) for PCA for architecture {arch_name} (need at least 2). Skipping.")
            continue
        if weights_matrix.ndim != 2 or (weights_matrix.ndim == 2 and weights_matrix.shape[1] == 0):
             print(f"  Weight matrix for {arch_name} is not suitable for PCA (shape: {weights_matrix.shape}). Skipping.")
             continue

        print(f"  Performing PCA on {weights_matrix.shape[0]} weight vectors of dimension {weights_matrix.shape[1]} for {arch_name}...")
        pca = PCA(n_components=2, random_state=42)
        try:
            transformed_weights = pca.fit_transform(weights_matrix)
        except ValueError as e:
            print(f"  PCA failed for {arch_name}: {e}. Weight matrix shape: {weights_matrix.shape}")
            continue
        
        current_pca_idx = 0
        for meta_idx, meta in enumerate(current_arch_pca_metadata):
            method_seed_key = (meta['method'], meta['seed'])
            state_key_pca = 'initial' if meta['state'] == 'Initial' else 'final'
            if method_seed_key in current_arch_trajectory_points and f'{state_key_pca}_raw' in current_arch_trajectory_points[method_seed_key]:
                if current_pca_idx < len(transformed_weights):
                    current_arch_trajectory_points[method_seed_key][state_key_pca] = (transformed_weights[current_pca_idx, 0], transformed_weights[current_pca_idx, 1])
                    current_pca_idx += 1
                else:
                    print(f"    WARNING: Ran out of transformed_weights for {method_seed_key}, {state_key_pca} (arch {arch_name}). Skipping for trajectory line.")
            
        plt.figure(figsize=(14, 10))
        colors = {'MAML': '#1f77b4', 'Vanilla': '#ff7f0e'}
        markers = {'Initial': 'o', 'Final': 'x'}
        line_styles = {'MAML': ':', 'Vanilla': '--'}
        plotted_legend_labels = {}

        for (method, seed), points_data in current_arch_trajectory_points.items():
            for state, coords in points_data.items():
                if state.endswith('_raw'): continue
                pc1, pc2 = coords
                label_key = f"{method} {state.capitalize()}"
                actual_marker_state = state.capitalize()
                plt.scatter(pc1, pc2,
                            color=colors[method],
                            marker=markers[actual_marker_state],
                            s=120, alpha=0.8,
                            label=label_key if label_key not in plotted_legend_labels else None,
                            zorder=3)
                if label_key not in plotted_legend_labels:
                    plotted_legend_labels[label_key] = True

        for (method, seed), points_data in current_arch_trajectory_points.items():
            if 'initial' in points_data and 'final' in points_data:
                initial_pt = points_data['initial']
                final_pt = points_data['final']
                plt.plot([initial_pt[0], final_pt[0]], [initial_pt[1], final_pt[1]],
                         linestyle=line_styles[method], color=colors[method],
                         alpha=0.5, linewidth=1.5, zorder=2)
        
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
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close()
        print(f"  PCA plot saved to {plot_filename} for architecture {arch_name}")

    print("\nFinished generating PCA plots.")
    return 0

if __name__ == '__main__':
    # Ensure that if main returns an error code, it's propagated
    sys.exit(main()) 