import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import h5py
import learn2learn as l2l
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# --- Import Model Architectures ---
# Import all necessary model classes
try:
    from baselines.models.conv2 import SameDifferentCNN as StandardConv2
    from baselines.models.conv4 import SameDifferentCNN as StandardConv4
    from baselines.models.conv6 import SameDifferentCNN as StandardConv6
    # The MAML models use a specific architecture we've defined separately
    from scripts.temp_model import PB_Conv6 as MamlModel
    print("Successfully imported model architectures.")
except ImportError as e:
    print(f"Error importing model architectures: {e}. Please check file paths.")
    sys.exit(1)

# --- Dataset for Adaptation ---
class LazyHDF5Dataset(Dataset):
    """Lazily loads data from an HDF5 file for adaptation."""
    def __init__(self, h5_path, transform=None, max_samples=None):
        self.h5_path = h5_path
        self.transform = transform
        self.file = None
        self.index_map = []
        with h5py.File(self.h5_path, 'r') as hf:
            episode_keys = sorted([k for k in hf.keys() if k.startswith('episode_')])
            for key in episode_keys:
                if 'support_images' in hf[key]:
                    num_samples = hf[key]['support_images'].shape[0]
                    for i in range(num_samples):
                        self.index_map.append((key, i))
                        if max_samples and len(self.index_map) >= max_samples: break
                    if max_samples and len(self.index_map) >= max_samples: break
    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.file is None: self.file = h5py.File(self.h5_path, 'r')
        ep_key, item_idx = self.index_map[idx]
        image = self.file[ep_key]['support_images'][item_idx]
        label = self.file[ep_key]['support_labels'][item_idx]
        if self.transform: image = self.transform(image)
        return image, torch.from_numpy(np.array(label)).long()

# --- Helper Functions ---
def flatten_weights(model_or_statedict):
    """Flattens weights from a model or a state_dict into a single numpy vector."""
    all_weights = []
    state_dict = model_or_statedict.state_dict() if hasattr(model_or_statedict, 'state_dict') else model_or_statedict
    for key in sorted(state_dict.keys()):
        param = state_dict[key]
        if isinstance(param, torch.Tensor):
            all_weights.append(param.detach().cpu().numpy().flatten())
    return np.concatenate(all_weights) if all_weights else None

def load_and_flatten(model_path, model_class):
    """Loads a model, flattens its weights, and returns the vector."""
    try:
        model = model_class()
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        return flatten_weights(model)
    except Exception as e:
        print(f"  - Warning: Could not load {model_path}. Error: {e}")
        return None

def main(args):
    # --- Paths and Config ---
    single_task_dir = Path(args.single_task_dir)
    maml_dir = Path(args.maml_dir)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Collection ---
    all_weights = []
    all_labels = []
    all_colors = []
    
    # Define colors
    cmap = plt.get_cmap('tab10')
    single_task_color = cmap(0)
    maml_pre_color = cmap(1)
    maml_post_color = cmap(2)

    # 1. Load Single-Task Weights
    print("\n--- Loading Single-Task Model Weights ---")
    PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 'random_color', 'arrows', 'irregular', 'filled', 'original']
    SEEDS_PER_TASK_IN_ORIG_EXP = 10 # This reflects the original experiment's folder structure
    SEEDS_TO_LOAD = range(5) # We only want to load the first 5 seeds for this analysis

    for task_idx, task_name in enumerate(PB_TASKS):
        for seed in SEEDS_TO_LOAD:
            # This logic replicates the folder structure from the original training script,
            # where seed folders had globally unique names.
            globally_unique_seed_for_folder = (task_idx * SEEDS_PER_TASK_IN_ORIG_EXP) + seed
            model_path = single_task_dir / task_name / 'conv6' / f'seed_{globally_unique_seed_for_folder}' / 'best_model.pth'

            if not model_path.exists(): continue
            
            weights = load_and_flatten(model_path, StandardConv6)
            if weights is not None:
                all_weights.append(weights)
                all_labels.append(f'Single-Task ({task_name})')
                all_colors.append(single_task_color)

    # 2. Load MAML Weights and Adapt Them
    print("\n--- Loading and Adapting MAML Model Weights ---")
    adaptation_loader = DataLoader(
        LazyHDF5Dataset(data_path, transform=T.Compose([T.ToPILImage(), T.Resize((32, 32)), T.ToTensor()]), max_samples=args.max_adaptation_samples),
        batch_size=args.adaptation_batch_size, shuffle=True
    )
    loss_func = torch.nn.CrossEntropyLoss()
    
    pre_adaptation_weights = []
    post_adaptation_weights = []

    for seed in range(3, 8): # Seeds 3-7 for this MAML model
        model_path = maml_dir / f'model_seed_{seed}_pretesting.pt'
        if not model_path.exists():
            print(f"Warning: MAML weight file for seed {seed} not found at {model_path}. Skipping.")
            continue
        
        print(f"Processing MAML seed {seed}...")
        model = MamlModel()
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        
        # Store pre-adaptation weights
        pre_weights = flatten_weights(model)
        pre_adaptation_weights.append(pre_weights)
        
        # Adapt model
        model.to(device)
        maml = l2l.algorithms.MAML(model, lr=args.lr, first_order=True)
        for _ in range(args.steps):
            for batch_images, batch_labels in adaptation_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                error = loss_func(maml(batch_images), batch_labels)
                maml.adapt(error)
        
        # Store post-adaptation weights
        post_weights = flatten_weights(maml.module)
        post_adaptation_weights.append(post_weights)

    # Add MAML weights to the main list
    all_weights.extend(pre_adaptation_weights)
    all_weights.extend(post_adaptation_weights)
    all_labels.extend(['MAML (Pre-Adaptation)'] * len(pre_adaptation_weights))
    all_labels.extend(['MAML (Post-Adaptation)'] * len(post_adaptation_weights))
    all_colors.extend([maml_pre_color] * len(pre_adaptation_weights))
    all_colors.extend([maml_post_color] * len(post_adaptation_weights))

    # --- PCA and Visualization ---
    if not all_weights:
        print("No weights were loaded. Aborting.")
        return

    print("\n--- Performing PCA on All Collected Weights ---")
    max_len = max(len(w) for w in all_weights)
    padded_weights = np.vstack([np.pad(w, (0, max_len - len(w)), 'constant') for w in all_weights])
    
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(padded_weights)

    print("\n--- Generating Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 14))

    # Plot the points
    ax.scatter(principal_components[:, 0], principal_components[:, 1], c=all_colors, alpha=0.6, s=80)

    # Draw arrows for adaptation
    num_maml_seeds = len(pre_adaptation_weights)
    pre_adapt_pca = principal_components[len(principal_components) - 2*num_maml_seeds : len(principal_components) - num_maml_seeds]
    post_adapt_pca = principal_components[len(principal_components) - num_maml_seeds:]

    for i in range(num_maml_seeds):
        ax.arrow(
            pre_adapt_pca[i, 0], pre_adapt_pca[i, 1],
            post_adapt_pca[i, 0] - pre_adapt_pca[i, 0],
            post_adapt_pca[i, 1] - pre_adapt_pca[i, 1],
            color='black', linestyle='--', lw=1.2, head_width=0.5, length_includes_head=True
        )

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Single-Task Models', markerfacecolor=single_task_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='MAML (Pre-Adaptation)', markerfacecolor=maml_pre_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='MAML (Post-Adaptation)', markerfacecolor=maml_post_color, markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    ax.set_title('PCA of Weight Space: Single-Task vs. MAML Adaptation', fontsize=18)
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=14)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=14)
    
    plot_path = output_dir / 'full_adaptation_pca.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize the MAML adaptation trajectory against single-task models.")
    parser.add_argument('--single_task_dir', type=str, default='/scratch/gpfs/mg7411/results/pb_baselines', help='Base directory for single-task model weights.')
    parser.add_argument('--maml_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6', help='Directory for the pre-trained MAML models.')
    parser.add_argument('--data_path', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/naturalistic/test.h5', help='Path to the naturalistic HDF5 data for adaptation.')
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/adaptation_pca', help='Directory to save the output plot.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for adaptation.')
    parser.add_argument('--steps', type=int, default=1, help='Number of adaptation epochs.')
    parser.add_argument('--adaptation_batch_size', type=int, default=128, help='Batch size for adaptation.')
    parser.add_argument('--max_adaptation_samples', type=int, default=2000, help='Max support samples to use for adaptation.')
    args = parser.parse_args()
    main(args) 