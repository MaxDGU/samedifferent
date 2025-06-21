import torch
import torch.nn as nn
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
from tqdm import tqdm

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Model Imports ---
try:
    from baselines.models.conv6 import SameDifferentCNN as VanillaModel
    from meta_baseline.models.conv6lr import SameDifferentCNN as MetaConv6Model
    print("Successfully imported model architectures.")
except ImportError as e:
    print(f"Fatal Error importing models: {e}. A dummy class will be used.")
    sys.exit(1)

class SimpleHDF5Dataset(Dataset):
    """
    Loads data from a PB HDF5 file. Assumes top-level datasets
    'support_images' and 'support_labels'.
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.file = None
        
        with h5py.File(self.h5_path, 'r') as hf:
            self.total_samples = hf['support_images'].shape[0] * hf['support_images'].shape[1]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
        
        num_support_per_ep = self.file['support_images'].shape[1]
        episode_idx = idx // num_support_per_ep
        item_idx_in_episode = idx % num_support_per_ep
        
        image = self.file['support_images'][episode_idx, item_idx_in_episode]
        label = self.file['support_labels'][episode_idx, item_idx_in_episode]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.from_numpy(np.array(label)).long()

def flatten_weights(model):
    return np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()])

def flatten_weights_from_state_dict(state_dict):
    """Flattens weights directly from a state_dict, bypassing model instantiation."""
    return np.concatenate([p.cpu().numpy().flatten() for p in state_dict.values()])

def adapt_model(model, loader, device, lr, steps):
    learner = copy.deepcopy(model)
    learner.to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    learner.train()
    for _ in range(steps):
        for batch_images, batch_labels in loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            predictions = learner(batch_images)
            error = loss_fn(predictions, batch_labels)
            error.backward()
            optimizer.step()
    learner.eval()
    return learner

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Base Models ---
    vanilla_path = Path(args.vanilla_models_dir) / f"regular/conv6/seed_{args.vanilla_seed}/initial_model.pth"
    vanilla_model = VanillaModel()
    vanilla_model.load_state_dict(torch.load(vanilla_path, map_location='cpu'))
    vanilla_pre_weights = flatten_weights(vanilla_model)

    # --- Load Pre-Trained PB Meta-Model ---
    pb_meta_path = Path(args.meta_models_dir) / f"seed_{args.meta_seed}/model_seed_{args.meta_seed}_checkpoint_epoch_80.pt"
    pb_meta_model = MetaConv6Model()
    pb_checkpoint = torch.load(pb_meta_path, map_location='cpu')
    pb_meta_model.load_state_dict(pb_checkpoint['model_state_dict'], strict=False)
    pb_meta_pre_weights = flatten_weights(pb_meta_model)

    # --- Load Naturalistic Meta-Trained Models ---
    naturalistic_deltas = []
    naturalistic_base_path = Path(args.naturalistic_models_dir)
    print("\n--- Loading Naturalistic Meta-Trained Models ---")
    for seed in args.naturalistic_seeds:
        try:
            model_path = naturalistic_base_path / f"conv6lr/seed_{seed}/conv6lr/seed_{seed}/conv6lr_best.pth"
            if not model_path.exists():
                print(f"  Warning: Could not find model file for naturalistic seed {seed} at {model_path}. Skipping.")
                continue
            
            print(f"Loading naturalistic model from seed {seed}")
            naturalistic_model = MetaConv6Model()
            checkpoint = torch.load(model_path, map_location='cpu')
            naturalistic_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            # This delta represents the entire naturalistic meta-training process
            # as a single vector from the common random start point.
            naturalistic_weights = flatten_weights(naturalistic_model)
            delta = naturalistic_weights - vanilla_pre_weights
            naturalistic_deltas.append(delta)
        except Exception as e:
            print(f"  Warning: Error loading model for naturalistic seed {seed}: {e}. Skipping.")

    # --- Task and Data Collection Setup ---
    tasks = ['arrows', 'filled', 'irregular', 'lines', 'open', 'regular']
    vanilla_adaptation_deltas = []
    pb_meta_adaptation_deltas = []

    # --- Adaptation Loop ---
    for task in tqdm(tasks, desc="Adapting to tasks"):
        data_path = Path(args.data_base_dir) / f"{task}_support{args.support_size}_train.h5"
        if not data_path.exists():
            print(f"  Warning: Data for task '{task}' not found at {data_path}. Skipping.")
            continue
        
        print(f"\n--- Adapting to {task} ---")
        # Adapt Vanilla Model
        loader = DataLoader(SimpleHDF5Dataset(data_path, transform=T.Compose([T.ToPILImage(), T.Resize((128, 128)), T.ToTensor()])), batch_size=args.adaptation_batch_size, shuffle=True)
        adapted_vanilla = adapt_model(vanilla_model, loader, device, args.lr, args.steps)
        vanilla_post_weights = flatten_weights(adapted_vanilla)
        vanilla_adaptation_deltas.append(vanilla_post_weights - vanilla_pre_weights)

        # Adapt PB-Meta-Trained Model
        adapted_pb_meta = adapt_model(pb_meta_model, loader, device, args.lr, args.steps)
        pb_meta_post_weights = flatten_weights(adapted_pb_meta)
        pb_meta_adaptation_deltas.append(pb_meta_post_weights - pb_meta_pre_weights)

    # --- PCA on Adaptation Vectors ---
    print("\n--- Performing PCA on All Change Vectors ---")
    # Combine all three types of "change" vectors to be visualized from a common origin
    all_deltas = vanilla_adaptation_deltas + pb_meta_adaptation_deltas + naturalistic_deltas

    max_len = max(len(d) for d in all_deltas)
    padded_deltas = np.vstack([np.pad(d, (0, max_len - len(d)), 'constant') for d in all_deltas])
    
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(padded_deltas)

    # --- Plotting ---
    print("\n--- Generating Plot ---")
    fig, ax = plt.subplots(figsize=(14, 12))
    
    num_vanilla_deltas = len(vanilla_adaptation_deltas)
    num_pb_meta_deltas = len(pb_meta_adaptation_deltas)
    
    vanilla_pcs = pcs[:num_vanilla_deltas]
    pb_meta_pcs = pcs[num_vanilla_deltas : num_vanilla_deltas + num_pb_meta_deltas]
    naturalistic_pcs = pcs[num_vanilla_deltas + num_pb_meta_deltas:]
    
    # Plot spokes from origin
    origin = np.array([0, 0])

    # Plot Vanilla Adaptation
    for i in range(vanilla_pcs.shape[0]):
        ax.arrow(origin[0], origin[1], vanilla_pcs[i, 0], vanilla_pcs[i, 1], 
                 color='royalblue', ls='--', lw=1.5, head_width=0.2, label='Vanilla Adaptation' if i == 0 else "")

    # Plot PB-Meta Adaptation
    for i in range(pb_meta_pcs.shape[0]):
        ax.arrow(origin[0], origin[1], pb_meta_pcs[i, 0], pb_meta_pcs[i, 1], 
                 color='darkorange', ls='-', lw=2, head_width=0.2, label='PB-Meta Adaptation' if i == 0 else "")

    # Plot Naturalistic Meta-Training vectors
    for i in range(naturalistic_pcs.shape[0]):
        ax.arrow(origin[0], origin[1], naturalistic_pcs[i, 0], naturalistic_pcs[i, 1], 
                 color='forestgreen', ls=':', lw=2.5, head_width=0.2, label='Naturalistic Meta-Training' if i == 0 else "")

    ax.scatter(origin[0], origin[1], c='black', s=150, zorder=5, marker='o', label='Shared Start Point')
    
    ax.set_title('PCA of Adaptation & Training Vectors from a Common Origin', fontsize=18)
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=14)
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    
    plot_path = output_dir / 'multi_task_adaptation_delta_pca.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize multi-task adaptation trajectories from a single starting point.")
    
    # --- Paths and Seeds ---
    parser.add_argument('--vanilla_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/single_task/results/pb_single_task', help='Base directory for initial Vanilla-PB models.')
    parser.add_argument('--meta_models_dir', type=str, default='/scratch/gpfs/mg7411/exp1_(untested)conv6lr_runs_20250127_110352', help='Directory for trained 6-layer Meta-PB models.')
    parser.add_argument('--naturalistic_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_meta', help='Base directory for meta-models trained on naturalistic data.')
    parser.add_argument('--data_base_dir', type=str, default='/scratch/gpfs/mg7411/data/pb/pb', help='Base directory containing PB task HDF5 files.')
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/multi_task_pca', help='Directory to save the output plot.')
    parser.add_argument('--vanilla_seed', type=int, default=0, help='Seed for the single vanilla model to use.')
    parser.add_argument('--meta_seed', type=int, default=1, help='Seed for the single meta model to use (e.g., 0, 1, or 2).')
    parser.add_argument('--naturalistic_seeds', type=int, nargs='+', default=[42, 123, 555, 789, 999], help='List of seeds for the naturalistic meta-trained models.')

    # --- Hyperparameters ---
    parser.add_argument('--lr', type=float, default=0.001, help='Unified learning rate for adaptation.')
    parser.add_argument('--steps', type=int, default=5, help='Number of adaptation steps.')
    parser.add_argument('--adaptation_batch_size', type=int, default=64, help='Batch size for adaptation.')
    parser.add_argument('--support_size', type=int, default=6, help='Support size (e.g., 6 for arrows_support6_train.h5) to use for tasks.')

    args = parser.parse_args()
    main(args) 