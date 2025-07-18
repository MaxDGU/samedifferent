import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import sys
import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import copy
import h5py

# --- Environment Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6lrCNN

# --- Configuration ---
PB_SEEDS = [789, 999, 42, 555, 123]
NAT_SEEDS = [999, 555, 111, 222, 333]
MODEL_CLASS = Conv6lrCNN

# Paths - using Path objects for robustness
BASE_PATH = Path('/scratch/gpfs/mg7411/samedifferent/') # Assumes running on Della
PB_PATH_TEMPLATE = BASE_PATH / 'results/pb_retrained_conv6lr/conv6/seed_{seed}/best_model.pt'
NAT_PATH_TEMPLATE = BASE_PATH / 'logs_naturalistic_meta/conv6lr/seed_{seed}/conv6lr/seed_{seed}/conv6lr_best.pth'
NAT_DATA_H5_PATH = BASE_PATH / 'data/naturalistic/test.h5' # <--- UPDATED Data Path

OUTPUT_DIR = Path('visualizations/domain_adaptation_pca')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Adaptation Hyperparameters
ADAPTATION_STEPS = 10 # <--- UPDATED k
ADAPTATION_LR = 0.01

# --- Data Loader for Adaptation (H5 Version) ---
class H5EpisodicSupportSetDataset(Dataset):
    """
    Loads support sets from an H5 file containing episodic data.
    """
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Input is already (128, 128, 3) numpy array
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with h5py.File(self.h5_path, 'r') as f:
            # Get a list of all episodes, e.g., ['episode_000001', 'episode_000002']
            self.episode_keys = sorted(list(f.keys()))

    def __len__(self):
        return len(self.episode_keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            episode_group = f[self.episode_keys[idx]]
            
            # Load support set images and labels
            support_images = np.array(episode_group['support_images'])
            support_labels = np.array(episode_group['support_labels'])
            
            # Apply transformations to all images
            transformed_images = torch.stack([self.transform(img) for img in support_images])
            
            return transformed_images, torch.tensor(support_labels, dtype=torch.long)

# --- Core Functions ---
def load_and_flatten_weights(model_path):
    if not model_path.exists():
        print(f"Warning: File not found: {model_path}")
        return None
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model = MODEL_CLASS()
        model.load_state_dict(state_dict)
        return torch.cat([p.detach().clone().flatten() for p in model.parameters()]).numpy()
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

def adapt_model_and_flatten(model_path, support_images, support_labels):
    """
    Loads a model, adapts it on a given support set, and returns a list
    of flattened weight vectors for each adaptation step.
    """
    if not model_path.exists(): return None
    
    device = torch.device("cpu") # Force CPU usage

    try:
        # Load the model state directly into the model we will adapt
        adapted_model = MODEL_CLASS().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        adapted_model.load_state_dict(state_dict)
        adapted_model.train()
    except Exception as e:
        print(f"Error loading model {model_path} for adaptation: {e}")
        return None
    
    # --- Store trajectory of weights ---
    weight_trajectory = [torch.cat([p.detach().clone().flatten() for p in adapted_model.parameters()]).numpy().copy()]
    
    optimizer = optim.SGD(adapted_model.parameters(), lr=ADAPTATION_LR)
    loss_fn = nn.CrossEntropyLoss()
    
    # Move the single support set to device
    support_images, support_labels = support_images.to(device), support_labels.to(device)

    for i in range(ADAPTATION_STEPS):
        optimizer.zero_grad()
        logits = adapted_model(support_images)
        loss = loss_fn(logits, support_labels)
        loss.backward()
        optimizer.step()
        
        # Store a TRUE COPY of the weights after each step
        weight_trajectory.append(torch.cat([p.detach().clone().flatten() for p in adapted_model.parameters()]).numpy().copy())
            
    return weight_trajectory

# --- Main Execution ---
def main():
    print("--- PCA of Conv6 Domain Adaptation ---")
    
    # Prepare adaptation data from the H5 file
    if not NAT_DATA_H5_PATH.exists():
        print(f"Error: Adaptation data not found at {NAT_DATA_H5_PATH}")
        return
        
    support_dataset = H5EpisodicSupportSetDataset(NAT_DATA_H5_PATH)
    if len(support_dataset) == 0:
        print(f"Error: No episodes found in {NAT_DATA_H5_PATH}")
        return
        
    # Use the support set from the very first episode for all adaptations
    first_episode_support_images, first_episode_support_labels = support_dataset[0]
    print(f"Using support set from episode '{support_dataset.episode_keys[0]}' for adaptation ({len(first_episode_support_labels)} samples).")

    weights_collection = {'PB_pre': [], 'Naturalistic': [], 'PB_post': []}
    
    # 1. Load Naturalistic meta-trained models
    for seed in NAT_SEEDS:
        path = Path(str(NAT_PATH_TEMPLATE).format(seed=seed))
        w = load_and_flatten_weights(path)
        if w is not None: weights_collection['Naturalistic'].append(w)

    # 2. Load PB meta-trained models and get adaptation trajectories
    for seed in PB_SEEDS:
        path = Path(str(PB_PATH_TEMPLATE).format(seed=seed))
        
        trajectory = adapt_model_and_flatten(path, first_episode_support_images, first_episode_support_labels)
        
        if trajectory:
            weights_collection['PB_pre'].append(trajectory[0])   # Pre-adaptation state
            weights_collection['PB_post'].append(trajectory[-1]) # Post-adaptation state
        
    # 3. Perform PCA on all collected weights
    all_weights = sum(weights_collection.values(), [])
    
    if len(all_weights) < 2:
        print("Not enough weights loaded to perform PCA. Exiting.")
        return
        
    weights_matrix = np.array(all_weights)
    pca_clusters = PCA(n_components=2)
    transformed_weights = pca_clusters.fit_transform(weights_matrix)
    
    # 4. Plotting (Plot 1: Absolute Positions)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a mapping from flat index to label and original data
    plot_data = {}
    start_idx = 0
    for label, w_list in weights_collection.items():
        end_idx = start_idx + len(w_list)
        plot_data[label] = transformed_weights[start_idx:end_idx]
        start_idx = end_idx

    # Plot points
    ax.scatter(plot_data['Naturalistic'][:, 0], plot_data['Naturalistic'][:, 1], 
               c='red', marker='s', s=150, label='Meta-Trained on Naturalistic', edgecolors='k', zorder=3)
    ax.scatter(plot_data['PB_pre'][:, 0], plot_data['PB_pre'][:, 1],
               c='blue', marker='o', s=150, label='Meta-Trained on PB (Pre-Adaptation)', edgecolors='k', zorder=3)
    ax.scatter(plot_data['PB_post'][:, 0], plot_data['PB_post'][:, 1],
               c='cyan', marker='X', s=150, label='Meta-Trained on PB (Post-Adaptation)', edgecolors='k', zorder=3)
               
    # Draw arrows from pre- to post-adaptation
    for i in range(len(plot_data['PB_pre'])):
        start_point = plot_data['PB_pre'][i]
        end_point = plot_data['PB_post'][i]
        ax.arrow(start_point[0], start_point[1], 
                 end_point[0] - start_point[0], end_point[1] - start_point[1],
                 head_width=0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), # Scale arrow head
                 fc='gray', ec='gray', length_includes_head=True, zorder=2, alpha=0.7)

    ax.set_title('PCA of Absolute Model Weights (Clusters)', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(fontsize=12)
    
    output_path_clusters = OUTPUT_DIR / 'pca_conv6_absolute_positions.png'
    plt.savefig(output_path_clusters, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Successfully saved cluster plot to {output_path_clusters}")
    
    # --- PLOT 2: PCA of Adaptation Vectors ---
    
    # 5. Calculate and perform PCA on the adaptation vectors themselves
    pb_pre_weights = np.array(weights_collection['PB_pre'])
    pb_post_weights = np.array(weights_collection['PB_post'])
    adaptation_vectors = pb_post_weights - pb_pre_weights
    
    pca_vectors = PCA(n_components=2)
    transformed_vectors = pca_vectors.fit_transform(adaptation_vectors)
    explained_variance_vectors = pca_vectors.explained_variance_ratio_

    # 6. Plotting the adaptation vectors (Quiver Plot)
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(len(transformed_vectors)):
        ax2.quiver(0, 0,
                   transformed_vectors[i, 0],
                   transformed_vectors[i, 1],
                   angles='xy', scale_units='xy', scale=1,
                   color=colors[i % len(colors)],
                   label=f'Seed {PB_SEEDS[i]}')

    ax2.scatter(0, 0, color='k', marker='+', s=100, label='Origin (No Change)', zorder=5)
    ax2.set_title('PCA of Adaptation Vectors (Direction of Change)', fontsize=16)
    ax2.set_xlabel(f'Principal Component 1 ({explained_variance_vectors[0]*100:.1f}%)', fontsize=12)
    ax2.set_ylabel(f'Principal Component 2 ({explained_variance_vectors[1]*100:.1f}%)', fontsize=12)
    ax2.legend()
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)
    
    output_path_vectors = OUTPUT_DIR / 'pca_conv6_adaptation_vectors.png'
    plt.savefig(output_path_vectors, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Successfully saved vector plot to {output_path_vectors}")

if __name__ == '__main__':
    main()
