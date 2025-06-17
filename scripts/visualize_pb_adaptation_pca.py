import torch
import torch.nn as nn
import learn2learn as l2l
import numpy as np
import h5py
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import copy

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Model Imports ---
try:
    from baselines.models.conv6 import SameDifferentCNN as VanillaModel
    from scripts.temp_model import PB_Conv6 as MetaModel
    print("Successfully imported model architectures.")
except ImportError as e:
    print(f"Fatal Error importing models: {e}. A dummy class will be used.")
    sys.exit(1)

# --- Dataset for Adaptation ---
class SimpleHDF5Dataset(Dataset):
    """
    Loads data from an HDF5 file where support images and labels are
    stored in top-level datasets. The first dimension of these datasets
    is treated as the episode index.
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.file = None # For lazy opening
        
        with h5py.File(self.h5_path, 'r') as hf:
            # The total number of items is num_episodes * num_images_per_episode
            self.num_episodes, self.num_support_per_ep = hf['support_images'].shape[:2]
            self.total_samples = self.num_episodes * self.num_support_per_ep

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            
        # Calculate which episode and which item in that episode this index corresponds to
        episode_idx = idx // self.num_support_per_ep
        item_idx_in_episode = idx % self.num_support_per_ep
        
        image = self.file['support_images'][episode_idx, item_idx_in_episode]
        label = self.file['support_labels'][episode_idx, item_idx_in_episode]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.from_numpy(np.array(label)).long()

# --- Helper Functions ---
def flatten_weights(model):
    """Flattens weights from a model into a single numpy vector."""
    return np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()])

def adapt_model(model, loader, device, lr, steps):
    """Performs adaptation on a cloned model and returns the adapted clone."""
    learner = copy.deepcopy(model)
    learner.to(device)
    
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    learner.train()
    for _ in range(steps):
        for batch_images, batch_labels in loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            predictions = learner(batch_images)
            error = loss_func(predictions, batch_labels)
            error.backward()
            optimizer.step()
            
    learner.eval()
    return learner

def check_and_report_nan(weights, context_msg):
    """Checks for NaN in a numpy array and prints a message if found."""
    if np.isnan(weights).any():
        print(f"    WARNING: NaN values detected {context_msg}")
        return True
    return False

def main(args):
    # --- Paths and Config ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the seeds to iterate over
    vanilla_seeds = [0, 1, 2, 3, 4]
    meta_seeds = [3, 4, 5, 6, 7]
    print(f"Processing Vanilla seeds: {vanilla_seeds}")
    print(f"Processing Meta seeds: {meta_seeds}")

    # --- Load Data ---
    transform_vanilla = T.Compose([T.ToPILImage(), T.Resize((128, 128)), T.ToTensor()])
    transform_meta = T.Compose([T.ToPILImage(), T.Resize((35, 35)), T.ToTensor()])
    loader_vanilla = DataLoader(SimpleHDF5Dataset(args.data_path, transform=transform_vanilla), batch_size=args.adaptation_batch_size, shuffle=True)
    loader_meta = DataLoader(SimpleHDF5Dataset(args.data_path, transform=transform_meta), batch_size=args.adaptation_batch_size, shuffle=True)

    # Lists to store all weight vectors
    all_weights_vanilla_pre = []
    all_weights_vanilla_post = []
    all_weights_meta_pre = []
    all_weights_meta_post = []

    # --- Loop and Adapt Vanilla Models ---
    print("\n--- Processing Vanilla-PB Models ---")
    for seed in vanilla_seeds:
        try:
            path = Path(args.vanilla_models_dir) / f"regular/conv6/seed_{seed}/initial_model.pth"
            print(f"  Loading model for seed {seed} from {path}...")
            model = VanillaModel()
            model.load_state_dict(torch.load(path, map_location='cpu'))

            # Check for NaNs in initial weights
            initial_weights = flatten_weights(model)
            check_and_report_nan(initial_weights, f"in initial Vanilla model, seed {seed}")
            all_weights_vanilla_pre.append(initial_weights)
            
            adapted_model = adapt_model(model, loader_vanilla, device, args.vanilla_lr, args.steps)

            # Check for NaNs in adapted weights
            adapted_weights = flatten_weights(adapted_model)
            check_and_report_nan(adapted_weights, f"after adapting Vanilla model, seed {seed}")
            all_weights_vanilla_post.append(adapted_weights)
            print(f"  Finished adapting seed {seed}.")
        except Exception as e:
            print(f"    ERROR processing vanilla seed {seed}: {e}")

    # --- Loop and Adapt Meta Models ---
    print("\n--- Processing Meta-PB Models ---")
    for seed in meta_seeds:
        try:
            path = Path(args.meta_models_dir) / f"model_seed_{seed}_pretesting.pt"
            print(f"  Loading model for seed {seed} from {path}...")
            model = MetaModel()
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # Check for NaNs in initial weights
            initial_weights = flatten_weights(model)
            check_and_report_nan(initial_weights, f"in initial Meta model, seed {seed}")
            all_weights_meta_pre.append(initial_weights)

            adapted_model = adapt_model(model, loader_meta, device, args.meta_lr, args.steps)

            # Check for NaNs in adapted weights
            adapted_weights = flatten_weights(adapted_model)
            check_and_report_nan(adapted_weights, f"after adapting Meta model, seed {seed}")
            all_weights_meta_post.append(adapted_weights)
            print(f"  Finished adapting seed {seed}.")
        except Exception as e:
            print(f"    ERROR processing meta seed {seed}: {e}")

    # --- PCA Analysis ---
    print("\n--- Performing PCA ---")
    all_weights = all_weights_vanilla_pre + all_weights_vanilla_post + all_weights_meta_pre + all_weights_meta_post
    
    max_len = max(len(w) for w in all_weights)
    padded_weights = np.vstack([np.pad(w, (0, max_len - len(w)), 'constant') for w in all_weights])
    
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(padded_weights)
    
    # --- Plotting ---
    print("--- Generating Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 12))
    
    vanilla_color, meta_color = 'royalblue', 'darkorange'
    
    # Split PCs for plotting
    num_vanilla = len(all_weights_vanilla_pre)
    num_meta = len(all_weights_meta_pre)
    vanilla_pre_pc = pcs[:num_vanilla]
    vanilla_post_pc = pcs[num_vanilla:2*num_vanilla]
    meta_pre_pc = pcs[2*num_vanilla : 2*num_vanilla + num_meta]
    meta_post_pc = pcs[2*num_vanilla + num_meta:]

    # Plot arrows and points
    for i in range(num_vanilla):
        ax.scatter(vanilla_pre_pc[i, 0], vanilla_pre_pc[i, 1], c=vanilla_color, s=150, alpha=0.5)
        ax.scatter(vanilla_post_pc[i, 0], vanilla_post_pc[i, 1], c=vanilla_color, s=150, alpha=1.0, marker='X')
        ax.arrow(vanilla_pre_pc[i, 0], vanilla_pre_pc[i, 1], vanilla_post_pc[i, 0] - vanilla_pre_pc[i, 0], vanilla_post_pc[i, 1] - vanilla_pre_pc[i, 1], color=vanilla_color, ls='--', lw=1.5, head_width=0.1)

    for i in range(num_meta):
        ax.scatter(meta_pre_pc[i, 0], meta_pre_pc[i, 1], c=meta_color, s=150, alpha=0.5)
        ax.scatter(meta_post_pc[i, 0], meta_post_pc[i, 1], c=meta_color, s=150, alpha=1.0, marker='X')
        ax.arrow(meta_pre_pc[i, 0], meta_pre_pc[i, 1], meta_post_pc[i, 0] - meta_pre_pc[i, 0], meta_post_pc[i, 1] - meta_pre_pc[i, 1], color=meta_color, ls='--', lw=1.5, head_width=0.1)

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Vanilla Pre-Adapt', markerfacecolor=vanilla_color, markersize=12, alpha=0.5),
        Line2D([0], [0], marker='X', color=vanilla_color, label='Vanilla Post-Adapt', markersize=10, lw=0),
        Line2D([0], [0], marker='o', color='w', label='Meta Pre-Adapt', markerfacecolor=meta_color, markersize=12, alpha=0.5),
        Line2D([0], [0], marker='X', color=meta_color, label='Meta Post-Adapt', markersize=10, lw=0)
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    ax.set_title('PCA of Adaptation Trajectories: Vanilla vs. Meta-Learned Weights (Multiple Seeds)', fontsize=16)
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.grid(True)

    plot_path = output_dir / 'pb_adaptation_trajectories_pca_multiseed.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare adaptation of Vanilla vs. Meta-trained models across multiple seeds.")
    # Paths to model directories
    parser.add_argument('--vanilla_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/single_task/results/pb_single_task', help='Base directory for initial Vanilla-PB models.')
    parser.add_argument('--meta_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6', help='Directory for trained Meta-PB models.')
    # Path to adaptation data
    parser.add_argument('--data_path', type=str, default='/scratch/gpfs/mg7411/data/pb/pb/arrows_support6_train.h5', help='Path to the HDF5 data for adaptation.')
    # Output and training parameters
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/adaptation_pca', help='Directory to save the output plot.')
    parser.add_argument('--meta_lr', type=float, default=0.01, help='Learning rate for meta model adaptation.')
    parser.add_argument('--vanilla_lr', type=float, default=0.001, help='Learning rate for vanilla model adaptation.')
    parser.add_argument('--steps', type=int, default=3, help='Number of adaptation epochs.')
    parser.add_argument('--adaptation_batch_size', type=int, default=64, help='Batch size for adaptation.')
    args = parser.parse_args()
    main(args) 