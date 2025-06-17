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
    from scripts.temp_model import PB_Conv6 as MetaModel
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
    meta_path = Path(args.meta_models_dir) / f"model_seed_{args.meta_seed}_pretesting.pt"

    vanilla_model = VanillaModel()
    vanilla_model.load_state_dict(torch.load(vanilla_path, map_location='cpu'))
    
    meta_model = MetaModel()
    checkpoint = torch.load(meta_path, map_location='cpu')
    meta_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    vanilla_pre_weights = flatten_weights(vanilla_model)
    meta_pre_weights = flatten_weights(meta_model)

    # --- Task and Data Collection Setup ---
    tasks = ['arrows', 'filled', 'irregular', 'lines', 'open', 'regular']
    all_weights = [vanilla_pre_weights, meta_pre_weights]
    vanilla_post_weights_list = []
    meta_post_weights_list = []

    # --- Adaptation Loop ---
    for task in tqdm(tasks, desc="Adapting to tasks"):
        data_path = Path(args.data_base_dir) / f"{task}_support{args.support_size}_train.h5"
        if not data_path.exists():
            print(f"  Warning: Data for task '{task}' not found at {data_path}. Skipping.")
            continue
        
        print(f"\n--- Adapting to {task} ---")
        # Adapt Vanilla
        loader_vanilla = DataLoader(SimpleHDF5Dataset(data_path, transform=T.Compose([T.ToPILImage(), T.Resize((128, 128)), T.ToTensor()])), batch_size=args.adaptation_batch_size, shuffle=True)
        adapted_vanilla = adapt_model(vanilla_model, loader_vanilla, device, args.lr, args.steps)
        vanilla_post_weights_list.append(flatten_weights(adapted_vanilla))

        # Adapt Meta
        loader_meta = DataLoader(SimpleHDF5Dataset(data_path, transform=T.Compose([T.ToPILImage(), T.Resize((35, 35)), T.ToTensor()])), batch_size=args.adaptation_batch_size, shuffle=True)
        adapted_meta = adapt_model(meta_model, loader_meta, device, args.lr, args.steps)
        meta_post_weights_list.append(flatten_weights(adapted_meta))

    all_weights.extend(vanilla_post_weights_list)
    all_weights.extend(meta_post_weights_list)

    # --- PCA and Plotting ---
    print("\n--- Performing PCA and Plotting ---")
    max_len = max(len(w) for w in all_weights)
    padded_weights = np.vstack([np.pad(w, (0, max_len - len(w)), 'constant') for w in all_weights])
    
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(padded_weights)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    vanilla_pre_pc = pcs[0]
    meta_pre_pc = pcs[1]
    
    num_tasks_run = len(vanilla_post_weights_list)
    vanilla_post_pcs = pcs[2 : 2 + num_tasks_run]
    meta_post_pcs = pcs[2 + num_tasks_run :]

    # Plot Vanilla
    ax.scatter(vanilla_pre_pc[0], vanilla_pre_pc[1], c='royalblue', s=300, alpha=0.9, marker='o', label='Vanilla Start')
    for i, task_name in enumerate(tasks):
        ax.scatter(vanilla_post_pcs[i, 0], vanilla_post_pcs[i, 1], c='royalblue', s=100, marker='X')
        ax.arrow(vanilla_pre_pc[0], vanilla_pre_pc[1], vanilla_post_pcs[i, 0] - vanilla_pre_pc[0], vanilla_post_pcs[i, 1] - vanilla_pre_pc[1], color='royalblue', ls='--', lw=1.5, head_width=0.1)

    # Plot Meta
    ax.scatter(meta_pre_pc[0], meta_pre_pc[1], c='darkorange', s=300, alpha=0.9, marker='o', label='Meta Start')
    for i, task_name in enumerate(tasks):
        ax.scatter(meta_post_pcs[i, 0], meta_post_pcs[i, 1], c='darkorange', s=100, marker='X')
        ax.arrow(meta_pre_pc[0], meta_pre_pc[1], meta_post_pcs[i, 0] - meta_pre_pc[0], meta_post_pcs[i, 1] - meta_pre_pc[1], color='darkorange', ls='--', lw=1.5, head_width=0.1)

    ax.set_title('Multi-Task Adaptation Trajectories (Shared PCA Space)', fontsize=18)
    ax.set_xlabel('Principal Component 1', fontsize=14)
    ax.set_ylabel('Principal Component 2', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    plot_path = output_dir / 'multi_task_adaptation_pca.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize multi-task adaptation trajectories from a single starting point.")
    
    # --- Paths and Seeds ---
    parser.add_argument('--vanilla_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/single_task/results/pb_single_task', help='Base directory for initial Vanilla-PB models.')
    parser.add_argument('--meta_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6', help='Directory for trained Meta-PB models.')
    parser.add_argument('--data_base_dir', type=str, default='/scratch/gpfs/mg7411/data/pb/pb', help='Base directory containing PB task HDF5 files.')
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/multi_task_pca', help='Directory to save the output plot.')
    parser.add_argument('--vanilla_seed', type=int, default=0, help='Seed for the single vanilla model to use.')
    parser.add_argument('--meta_seed', type=int, default=3, help='Seed for the single meta model to use.')

    # --- Hyperparameters ---
    parser.add_argument('--lr', type=float, default=0.001, help='Unified learning rate for adaptation.')
    parser.add_argument('--steps', type=int, default=5, help='Number of adaptation steps.')
    parser.add_argument('--adaptation_batch_size', type=int, default=64, help='Batch size for adaptation.')
    parser.add_argument('--support_size', type=int, default=6, help='Support size (e.g., 6 for arrows_support6_train.h5) to use for tasks.')

    args = parser.parse_args()
    main(args) 