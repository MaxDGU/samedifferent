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

    # --- Define Seeds ---
    vanilla_seeds = [0, 1, 2, 3, 4]
    meta_seeds = [3, 4, 5, 6, 7]
    print(f"Processing Vanilla seeds: {vanilla_seeds}")
    print(f"Processing Meta seeds: {meta_seeds}")

    # --- Load Data ---
    transform_vanilla = T.Compose([T.ToPILImage(), T.Resize((128, 128)), T.ToTensor()])
    transform_meta = T.Compose([T.ToPILImage(), T.Resize((35, 35)), T.ToTensor()])
    loader_vanilla = DataLoader(SimpleHDF5Dataset(args.data_path, transform=transform_vanilla), batch_size=args.adaptation_batch_size, shuffle=True)
    loader_meta = DataLoader(SimpleHDF5Dataset(args.data_path, transform=transform_meta), batch_size=args.adaptation_batch_size, shuffle=True)

    # --- Data Collection ---
    all_weights_vanilla_pre, all_weights_vanilla_post = [], []
    all_weights_meta_pre, all_weights_meta_post = [], []
    vanilla_distances, meta_distances = [], []

    # --- Process Vanilla Models ---
    print("\n--- Processing Vanilla-PB Models ---")
    for seed in vanilla_seeds:
        try:
            path = Path(args.vanilla_models_dir) / f"regular/conv6/seed_{seed}/initial_model.pth"
            print(f"  Loading model for seed {seed} from {path}...")
            model = VanillaModel()
            model.load_state_dict(torch.load(path, map_location='cpu'))
            
            initial_weights = flatten_weights(model)
            all_weights_vanilla_pre.append(initial_weights)
            
            adapted_model = adapt_model(model, loader_vanilla, device, args.lr, args.steps)
            adapted_weights = flatten_weights(adapted_model)
            all_weights_vanilla_post.append(adapted_weights)
            
            distance = np.linalg.norm(adapted_weights - initial_weights)
            vanilla_distances.append(distance)
            print(f"  Finished adapting seed {seed}. Distance: {distance:.4f}")

        except Exception as e:
            print(f"    ERROR processing vanilla seed {seed}: {e}")

    # --- Process Meta Models ---
    print("\n--- Processing Meta-PB Models ---")
    for seed in meta_seeds:
        try:
            path = Path(args.meta_models_dir) / f"model_seed_{seed}_pretesting.pt"
            print(f"  Loading model for seed {seed} from {path}...")
            model = MetaModel()
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            initial_weights = flatten_weights(model)
            all_weights_meta_pre.append(initial_weights)

            adapted_model = adapt_model(model, loader_meta, device, args.lr, args.steps)
            adapted_weights = flatten_weights(adapted_model)
            all_weights_meta_post.append(adapted_weights)

            distance = np.linalg.norm(adapted_weights - initial_weights)
            meta_distances.append(distance)
            print(f"  Finished adapting seed {seed}. Distance: {distance:.4f}")

        except Exception as e:
            print(f"    ERROR processing meta seed {seed}: {e}")

    # --- Analysis & Plotting ---
    if all_weights_vanilla_pre:
        print("\n--- Performing PCA and Plotting for Vanilla Models (focused on adaptation) ---")
        vanilla_deltas = [post - pre for pre, post in zip(all_weights_vanilla_pre, all_weights_vanilla_post)]
        max_len_vanilla = max(len(w) for w in vanilla_deltas)
        padded_vanilla_deltas = np.vstack([np.pad(w, (0, max_len_vanilla - len(w)), 'constant') for w in vanilla_deltas])

        pca_vanilla = PCA(n_components=2, random_state=42)
        pca_vanilla.fit(padded_vanilla_deltas) # Fit PCA on the adaptation vectors

        all_vanilla_weights = all_weights_vanilla_pre + all_weights_vanilla_post
        max_len_vanilla_all = max(len(w) for w in all_vanilla_weights)
        padded_vanilla_all = np.vstack([np.pad(w, (0, max_len_vanilla_all - len(w)), 'constant') for w in all_vanilla_weights])
        pcs_vanilla = pca_vanilla.transform(padded_vanilla_all) # Transform original points
        
        fig_vanilla, ax_vanilla = plt.subplots(figsize=(12, 10))
        num_vanilla = len(all_weights_vanilla_pre)
        vanilla_pre_pc = pcs_vanilla[:num_vanilla]
        vanilla_post_pc = pcs_vanilla[num_vanilla:]

        for i in range(num_vanilla):
            ax_vanilla.scatter(vanilla_pre_pc[i, 0], vanilla_pre_pc[i, 1], c='royalblue', s=150, alpha=0.6)
            ax_vanilla.scatter(vanilla_post_pc[i, 0], vanilla_post_pc[i, 1], c='royalblue', s=150, marker='X')
            ax_vanilla.arrow(vanilla_pre_pc[i, 0], vanilla_pre_pc[i, 1], vanilla_post_pc[i, 0] - vanilla_pre_pc[i, 0], vanilla_post_pc[i, 1] - vanilla_pre_pc[i, 1], color='royalblue', ls='--', lw=1.5, head_width=0.1)
        
        ax_vanilla.set_title('PCA of Adaptation Trajectories: Vanilla Models', fontsize=16)
        ax_vanilla.set_xlabel(f'Principal Component of Adaptation 1 ({pca_vanilla.explained_variance_ratio_[0]:.2%})', fontsize=12)
        ax_vanilla.set_ylabel(f'Principal Component of Adaptation 2 ({pca_vanilla.explained_variance_ratio_[1]:.2%})', fontsize=12)
        ax_vanilla.grid(True)
        vanilla_plot_path = output_dir / 'vanilla_adaptation_pca.png'
        plt.savefig(vanilla_plot_path, bbox_inches='tight')
        print(f"Vanilla plot saved to {vanilla_plot_path}")

    if all_weights_meta_pre:
        print("\n--- Performing PCA and Plotting for Meta Models (focused on adaptation) ---")
        meta_deltas = [post - pre for pre, post in zip(all_weights_meta_pre, all_weights_meta_post)]
        max_len_meta = max(len(w) for w in meta_deltas)
        padded_meta_deltas = np.vstack([np.pad(w, (0, max_len_meta - len(w)), 'constant') for w in meta_deltas])

        pca_meta = PCA(n_components=2, random_state=42)
        pca_meta.fit(padded_meta_deltas) # Fit PCA on the adaptation vectors

        all_meta_weights = all_weights_meta_pre + all_weights_meta_post
        max_len_meta_all = max(len(w) for w in all_meta_weights)
        padded_meta_all = np.vstack([np.pad(w, (0, max_len_meta_all - len(w)), 'constant') for w in all_meta_weights])
        pcs_meta = pca_meta.transform(padded_meta_all) # Transform original points

        fig_meta, ax_meta = plt.subplots(figsize=(12, 10))
        num_meta = len(all_weights_meta_pre)
        meta_pre_pc = pcs_meta[:num_meta]
        meta_post_pc = pcs_meta[num_meta:]

        for i in range(num_meta):
            ax_meta.scatter(meta_pre_pc[i, 0], meta_pre_pc[i, 1], c='darkorange', s=150, alpha=0.6)
            ax_meta.scatter(meta_post_pc[i, 0], meta_post_pc[i, 1], c='darkorange', s=150, marker='X')
            ax_meta.arrow(meta_pre_pc[i, 0], meta_pre_pc[i, 1], meta_post_pc[i, 0] - meta_pre_pc[i, 0], meta_post_pc[i, 1] - meta_pre_pc[i, 1], color='darkorange', ls='--', lw=1.5, head_width=0.1)

        ax_meta.set_title('PCA of Adaptation Trajectories: Meta-Learned Models', fontsize=16)
        ax_meta.set_xlabel(f'Principal Component of Adaptation 1 ({pca_meta.explained_variance_ratio_[0]:.2%})', fontsize=12)
        ax_meta.set_ylabel(f'Principal Component of Adaptation 2 ({pca_meta.explained_variance_ratio_[1]:.2%})', fontsize=12)
        ax_meta.grid(True)
        meta_plot_path = output_dir / 'meta_adaptation_pca.png'
        plt.savefig(meta_plot_path, bbox_inches='tight')
        print(f"Meta plot saved to {meta_plot_path}")

    # --- Final Quantitative Summary ---
    print("\n--- Quantitative Analysis ---")
    if vanilla_distances:
        print(f"Vanilla Mean Distance: {np.mean(vanilla_distances):.4f} (+/- {np.std(vanilla_distances):.4f})")
    if meta_distances:
        print(f"Meta Mean Distance:    {np.mean(meta_distances):.4f} (+/- {np.std(meta_distances):.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare adaptation of Vanilla vs. Meta-trained models across multiple seeds.")
    # Paths to model directories
    parser.add_argument('--vanilla_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/single_task/results/pb_single_task', help='Base directory for initial Vanilla-PB models.')
    parser.add_argument('--meta_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6', help='Directory for trained Meta-PB models.')
    # Path to adaptation data
    parser.add_argument('--data_path', type=str, default='/scratch/gpfs/mg7411/data/pb/pb/arrows_support6_train.h5', help='Path to the HDF5 data for adaptation.')
    # Output and training parameters
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/adaptation_pca', help='Directory to save the output plot.')
    parser.add_argument('--lr', type=float, default=0.001, help='Unified learning rate for adaptation.')
    parser.add_argument('--steps', type=int, default=5, help='Number of adaptation epochs.')
    parser.add_argument('--adaptation_batch_size', type=int, default=64, help='Batch size for adaptation.')
    args = parser.parse_args()
    main(args) 