import argparse
import os
import copy
import torch
import torch.nn as nn
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset

from meta_baseline.models.conv6lr import SameDifferentCNN

# The SameDifferentDataset class is no longer needed for static PCA, but we keep it for future use.
class SameDifferentDataset(Dataset):
    """
    Dataset for loading PB same-different task data.
    """
    def __init__(self, data_dir, tasks, split, support_sizes=[10]):
        self.data_dir = data_dir
        self.tasks = tasks
        self.split = split
        self.support_sizes = support_sizes
        self.episode_files = []
        for task in tasks:
            for support_size in support_sizes:
                file_path = os.path.join(data_dir, f'{task}_support{support_size}_{split}.h5')
                if os.path.exists(file_path):
                    self.episode_files.append({'file_path': file_path, 'task': task, 'support_size': support_size})
        if not self.episode_files: raise ValueError(f"No valid files found for tasks {tasks} in {data_dir}")
        self.total_episodes, self.file_episode_counts = 0, []
        for file_info in self.episode_files:
            with h5py.File(file_info['file_path'], 'r') as f:
                num_episodes = f['support_images'].shape[0]
                self.file_episode_counts.append(num_episodes)
                self.total_episodes += num_episodes

    def __len__(self):
        return self.total_episodes

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.file_episode_counts[file_idx]:
            idx -= self.file_episode_counts[file_idx]
            file_idx += 1
        file_info = self.episode_files[file_idx]
        with h5py.File(file_info['file_path'], 'r') as f:
            support_images = torch.from_numpy(f['support_images'][idx]).float() / 255.0
            support_labels = torch.from_numpy(f['support_labels'][idx]).long()
        support_images = support_images.permute(0, 3, 1, 2)
        return {'support_images': support_images, 'support_labels': support_labels, 'task': file_info['task']}

def load_model(path, device):
    """Loads a model checkpoint."""
    model = SameDifferentCNN().to(device)
    try:
        # Try loading the whole checkpoint
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Check for different possible state_dict keys
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # The meta-trained model has 'module.' prefix from DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)

    except (RuntimeError, KeyError, TypeError) as e:
        print(f"Failed to load checkpoint directly, trying to load just state_dict. Error: {e}")
        # If above fails, assume it's just the state dict
        state_dict = torch.load(path, map_location=device, weights_only=False)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
    print(f"Loaded model from {os.path.basename(path)} | Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    return model

def get_model_weights(model):
    """Flattens and returns model weights as a numpy array."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

# The adapt_model function is not used in this simplified script.
# def adapt_model(model, data_loader, criterion, device, adaptation_steps=10, lr=0.01):
#    ...

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Define Model Paths ---
    meta_seeds = [111, 222, 333]
    vanilla_seeds = [123, 555, 42, 999, 789]
    
    meta_paths = [f"/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_meta/conv6lr/seed_{s}/conv6lr/seed_{s}/conv6lr_best.pth" for s in meta_seeds]
    vanilla_paths = [f"/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/conv6lr/seed_{s}/best_model.pt" for s in vanilla_seeds]

    # --- Load Models and Collect Weights ---
    print("\n--- Loading All Models for Static PCA ---")
    all_weights = []
    model_types = [] # 'meta' or 'vanilla'

    for path in tqdm(meta_paths, desc="Loading meta models"):
        try:
            model = load_model(path, device)
            all_weights.append(get_model_weights(model))
            model_types.append('meta')
        except Exception as e:
            print(f"Could not load meta model {path}. Error: {e}")
            
    for path in tqdm(vanilla_paths, desc="Loading vanilla models"):
        try:
            model = load_model(path, device)
            all_weights.append(get_model_weights(model))
            model_types.append('vanilla')
        except Exception as e:
            print(f"Could not load vanilla model {path}. Error: {e}")

    if not all_weights:
        print("No models were loaded successfully. Exiting.")
        return

    # --- Perform PCA ---
    print("\n--- Performing PCA on Initial Weights ---")
    all_weights = np.vstack(all_weights)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_weights)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # --- Plotting ---
    print("\n--- Plotting Results ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'meta': 'blue', 'vanilla': 'red'}
    labels = {'meta': 'Meta-Trained', 'vanilla': 'Vanilla-Trained'}

    # Collect points for legend
    plotted_labels = set()

    for i, pc in enumerate(principal_components):
        model_type = model_types[i]
        label = labels[model_type] if model_type not in plotted_labels else ""
        ax.scatter(pc[0], pc[1], color=colors[model_type], s=100, alpha=0.8, label=label)
        plotted_labels.add(model_type)

    ax.set_title(f'PCA of Initial Conv6 Weights (Naturalistic Models)', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(fontsize=12)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    save_path = os.path.join(args.save_dir, 'naturalistic_static_pca.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize static PCA of model weights.')
    parser.add_argument('--save_dir', type=str, default='./visualizations/domain_adaptation_pca/naturalistic_static', help='Directory to save the plot.')
    args = parser.parse_args()
    main(args) 