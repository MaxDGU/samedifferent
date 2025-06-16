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
class LazyHDF5Dataset(Dataset):
    """Lazily loads data from an HDF5 file for adaptation."""
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.file = None
        self.index_map = []
        with h5py.File(self.h5_path, 'r') as hf:
            for key in [k for k in hf.keys() if k.startswith('episode_')]:
                if 'support_images' in hf[key]:
                    num_samples = hf[key]['support_images'].shape[0]
                    for i in range(num_samples):
                        self.index_map.append((key, i))
    
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
def flatten_weights(model):
    """Flattens weights from a model into a single numpy vector."""
    return np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()])

def adapt_model(model, loader, device, lr, steps):
    """Performs adaptation and returns the adapted model."""
    model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    maml_wrapper = l2l.algorithms.MAML(model, lr=lr, first_order=True)
    
    for _ in range(steps):
        for batch_images, batch_labels in loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            error = loss_func(maml_wrapper(batch_images), batch_labels)
            maml_wrapper.adapt(error, allow_unused=True)
            
    return maml_wrapper.module

def main(args):
    # --- Paths and Config ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    # The models expect different input sizes, so we need two different transforms/loaders.
    # StandardConv6 from baselines expects 128x128.
    # PB_Conv6 (MetaModel) from temp_model expects 35x35.
    transform_vanilla = T.Compose([T.ToPILImage(), T.Resize((128, 128)), T.ToTensor()])
    transform_meta = T.Compose([T.ToPILImage(), T.Resize((35, 35)), T.ToTensor()])
    
    loader_vanilla = DataLoader(LazyHDF5Dataset(args.data_path, transform=transform_vanilla), batch_size=args.adaptation_batch_size, shuffle=True)
    loader_meta = DataLoader(LazyHDF5Dataset(args.data_path, transform=transform_meta), batch_size=args.adaptation_batch_size, shuffle=True)

    # --- Load Models ---
    print("\n--- Loading Models ---")
    vanilla_model = VanillaModel()
    vanilla_model.load_state_dict(torch.load(args.vanilla_model_path, map_location='cpu'))
    print(f"Loaded Vanilla-PB model from: {args.vanilla_model_path}")

    meta_model = MetaModel()
    meta_model.load_state_dict(torch.load(args.meta_model_path, map_location='cpu'))
    print(f"Loaded Meta-PB model from: {args.meta_model_path}")

    # --- Get Pre-Adaptation Weights ---
    weights_vanilla_pre = flatten_weights(vanilla_model)
    weights_meta_pre = flatten_weights(meta_model)

    # --- Adapt Models ---
    print("\n--- Adapting Models ---")
    adapted_vanilla_model = adapt_model(vanilla_model, loader_vanilla, device, args.lr, args.steps)
    print("Vanilla-PB model adapted.")
    adapted_meta_model = adapt_model(meta_model, loader_meta, device, args.lr, args.steps)
    print("Meta-PB model adapted.")

    # --- Get Post-Adaptation Weights ---
    weights_vanilla_post = flatten_weights(adapted_vanilla_model)
    weights_meta_post = flatten_weights(adapted_meta_model)

    # --- PCA Analysis ---
    print("\n--- Performing PCA ---")
    all_weights = [weights_vanilla_pre, weights_vanilla_post, weights_meta_pre, weights_meta_post]
    
    # Pad weights to be the same length for PCA
    max_len = max(len(w) for w in all_weights)
    padded_weights = np.vstack([np.pad(w, (0, max_len - len(w)), 'constant') for w in all_weights])
    
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(padded_weights)
    
    # --- Plotting ---
    print("--- Generating Plot ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    vanilla_color, meta_color = 'royalblue', 'darkorange'
    
    # Plot points
    ax.scatter(pcs[0, 0], pcs[0, 1], c=vanilla_color, s=150, alpha=0.6, label='Vanilla Pre-Adapt')
    ax.scatter(pcs[1, 0], pcs[1, 1], c=vanilla_color, s=150, alpha=1.0, marker='X', label='Vanilla Post-Adapt')
    ax.scatter(pcs[2, 0], pcs[2, 1], c=meta_color, s=150, alpha=0.6, label='Meta Pre-Adapt')
    ax.scatter(pcs[3, 0], pcs[3, 1], c=meta_color, s=150, alpha=1.0, marker='X', label='Meta Post-Adapt')

    # Draw arrows
    ax.arrow(pcs[0, 0], pcs[0, 1], pcs[1, 0] - pcs[0, 0], pcs[1, 1] - pcs[0, 1], color=vanilla_color, ls='--', lw=1.5, head_width=0.1)
    ax.arrow(pcs[2, 0], pcs[2, 1], pcs[3, 0] - pcs[2, 0], pcs[3, 1] - pcs[2, 1], color=meta_color, ls='--', lw=1.5, head_width=0.1)

    ax.legend(loc='best', fontsize=12)
    ax.set_title('PCA of Adaptation Trajectories: Vanilla vs. Meta-Learned Weights', fontsize=16)
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.grid(True)

    plot_path = output_dir / 'pb_adaptation_trajectories_pca.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare adaptation of Vanilla vs. Meta-trained models.")
    # Paths to specific model files
    parser.add_argument('--vanilla_model_path', type=str, default='/scratch/gpfs/mg7411/samedifferent/single_task/results/pb_single_task/regular/conv6/seed_80/initial_model.pth', help='Path to the initial Vanilla-PB model weights.')
    parser.add_argument('--meta_model_path', type=str, default='/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6/model_seed_3_pretesting.pt', help='Path to the trained Meta-PB model weights.')
    # Path to adaptation data
    parser.add_argument('--data_path', type=str, default='/scratch/gpfs/mg7411/data/pb/pb/arrows_support6_test.h5', help='Path to the HDF5 data for adaptation.')
    # Output and training parameters
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/adaptation_pca', help='Directory to save the output plot.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for adaptation.')
    parser.add_argument('--steps', type=int, default=3, help='Number of adaptation epochs.')
    parser.add_argument('--adaptation_batch_size', type=int, default=64, help='Batch size for adaptation.')
    args = parser.parse_args()
    main(args) 