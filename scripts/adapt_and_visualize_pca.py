import torch
import learn2learn as l2l
import numpy as np
import h5py
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from torchvision import transforms as T
from torch.utils.data import TensorDataset, DataLoader, Dataset

# Ensure the project root is in the Python path
# This allows us to import from 'baselines' and 'meta_baseline'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Correctly import the model architecture
# The weights inspection revealed a hybrid architecture. We now import
# the temporary model definition that was built to match it exactly.
try:
    from scripts.temp_model import SameDifferentCNN as ModelCNN
except ImportError:
    print("Error: Could not import the model from 'scripts/temp_model.py'.")
    print("Please ensure the file exists and the PYTHONPATH is set correctly.")
    sys.exit(1)


class LazyHDF5Dataset(Dataset):
    """
    A PyTorch Dataset that loads data from an HDF5 file lazily.
    It builds an index of all samples across all episodes at initialization
    but only loads samples into memory when requested by __getitem__.
    This is crucial for datasets that are too large to fit into RAM.
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.file = None  # File handle will be opened in __getitem__ if not open

        # Build an index of all samples without loading the actual image data
        self.index_map = []
        with h5py.File(self.h5_path, 'r') as hf:
            episode_keys = sorted([key for key in hf.keys() if key.startswith('episode_')])
            for key in episode_keys:
                if 'support_images' in hf[key]:
                    num_samples = hf[key]['support_images'].shape[0]
                    for i in range(num_samples):
                        self.index_map.append((key, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # h5py file objects are not fork-safe, so we open the file here
        # if it's not already open in this worker process.
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        episode_key, item_idx_in_episode = self.index_map[idx]
        
        # Load one sample from disk
        image = self.file[episode_key]['support_images'][item_idx_in_episode]
        label = self.file[episode_key]['support_labels'][item_idx_in_episode]
        
        label_tensor = torch.from_numpy(np.array(label)).long()

        # The transform pipeline handles conversion from numpy/PIL to a normalized tensor
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image).float()
            
        return image_tensor, label_tensor


def flatten_weights(state_dict):
    """Flattens a model's state_dict into a single numpy vector."""
    return np.concatenate([p.cpu().numpy().flatten() for p in state_dict.values()])


def main():
    # --- Main Script ---
    parser = argparse.ArgumentParser(
        description="Adapt MAML models and visualize weight changes with PCA."
    )
    # Update default paths for the Della cluster environment
    parser.add_argument('--weights_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6',
                        help='Directory containing the pre-trained model weight files.')
    parser.add_argument('--data_path', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/naturalistic/test.h5',
                        help='Path to the HDF5 file with the naturalistic meta-testing data.')
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/adaptation_pca',
                        help='Directory where the PCA plot will be saved.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate for adaptation.')
    parser.add_argument('--steps', type=int, default=1,
                        help='Number of adaptation epochs over the support set.')
    parser.add_argument('--adaptation_batch_size', type=int, default=128,
                        help='Batch size for the adaptation phase.')

    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not weights_dir.exists():
        print(f"Error: Weights directory not found at {weights_dir}")
        return
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    adaptation_lr = args.lr
    adaptation_steps = args.steps
    adaptation_batch_size = args.adaptation_batch_size
    num_seeds_start = 3
    num_seeds_end = 10 # range goes up to, but does not include, this value

    # --- Load Data Lazily ---
    print(f"Initializing lazy loading from {data_path}...")

    # Define transformations to be applied to each image as it's loaded
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Instantiate the lazy dataset
    support_dataset = LazyHDF5Dataset(data_path, transform=transform)
    print(f"  Found a total of {len(support_dataset)} support images across all episodes.")

    # Create a DataLoader for batching. num_workers must be 0 for this h5py implementation.
    support_loader = DataLoader(support_dataset, batch_size=adaptation_batch_size, shuffle=True, num_workers=0)

    # We still need a loss function
    loss_func = torch.nn.CrossEntropyLoss()

    initial_weights_vectors = []
    adapted_weights_vectors = []
    seed_labels = []

    # --- Adaptation Loop ---
    for seed in range(num_seeds_start, num_seeds_end):
        seed_labels.append(seed)
        model_path = weights_dir / f'model_seed_{seed}_pretesting.pt'
        if not model_path.exists():
            print(f"Warning: Weight file for seed {seed} not found at {model_path}. Skipping.")
            continue

        print(f"\nProcessing Seed {seed}...")

        # 1. Load Model and Initial Weights
        # We now use the custom-built model and load with strict=False
        # to ignore the MAML-specific learning rate keys in the file.
        model = ModelCNN()
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            
            # The saved weights might be nested under 'model_state_dict'
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            model.load_state_dict(state_dict, strict=False)
            print(f"  Successfully loaded weights for seed {seed}.")
        except Exception as e:
            print(f"  Could not load weights for seed {seed}, error: {e}")
            continue

        initial_weights = flatten_weights(model.state_dict())
        initial_weights_vectors.append(initial_weights)

        model.to(device)
        maml = l2l.algorithms.MAML(model, lr=adaptation_lr, first_order=False)

        # 3. Adapt on Naturalistic Data using batches
        print(f"  Adapting model for {adaptation_steps} epoch(s) with batch size {adaptation_batch_size}...")
        for epoch in range(adaptation_steps):
            for batch_images, batch_labels in support_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                
                # Perform one adaptation step on the batch
                error = loss_func(maml(batch_images), batch_labels)
                maml.adapt(error, allow_unused=True)
        
        # 4. Store Final Weights Snapshot
        print("  Storing final weights snapshot...")
        adapted_weights = flatten_weights(maml.module.state_dict())
        adapted_weights_vectors.append(adapted_weights)


    # --- PCA and Visualization ---
    if not initial_weights_vectors or not adapted_weights_vectors:
        print("\nNo weights were processed. Cannot perform PCA.")
        return

    print("\nPerforming PCA on initial and adapted weights...")
    all_weights = np.array(initial_weights_vectors + adapted_weights_vectors)

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_weights)
    explained_variance = pca.explained_variance_ratio_

    # Split back into initial and adapted
    num_processed_seeds = len(initial_weights_vectors)
    initial_pca = principal_components[:num_processed_seeds]
    adapted_pca = principal_components[num_processed_seeds:]

    # Create Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot points and connecting lines
    for i in range(num_processed_seeds):
        seed = seed_labels[i]
        # Initial point (blue)
        ax.scatter(initial_pca[i, 0], initial_pca[i, 1], c='blue', s=80, alpha=0.7, label='Initial' if i == 0 else "")
        # Adapted point (red)
        ax.scatter(adapted_pca[i, 0], adapted_pca[i, 1], c='red', s=80, alpha=0.7, label='Adapted' if i == 0 else "")
        # Arrow connecting them
        ax.arrow(initial_pca[i, 0], initial_pca[i, 1],
                 adapted_pca[i, 0] - initial_pca[i, 0],
                 adapted_pca[i, 1] - initial_pca[i, 1],
                 color='gray', linestyle='--', lw=1.0,
                 head_width=0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), # Dynamic head width
                 length_includes_head=True)
        # Label the seed number
        ax.text(adapted_pca[i, 0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), adapted_pca[i, 1], str(seed), fontsize=9)


    ax.set_title(f'PCA of MAML Weights Before and After Adaptation on Naturalistic Data\n(Architecture: conv6lr)', fontsize=16)
    ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2%})', fontsize=12)
    ax.legend()
    ax.grid(True)

    plot_filename = output_dir / 'pca_adaptation_trajectory.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPCA plot saved to {plot_filename}")


if __name__ == '__main__':
    main() 