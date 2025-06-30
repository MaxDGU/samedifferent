import argparse
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

from meta_baseline.models.conv6lr import SameDifferentCNN
from data.meta_data_generator_h5 import MetaDatasetGenerator

def load_model(path, device):
    """Loads a model checkpoint."""
    model = SameDifferentCNN(num_classes=2).to(device)
    try:
        # Try loading the whole checkpoint
        checkpoint = torch.load(path, map_location=device)
        
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

    except (RuntimeError, KeyError) as e:
        print(f"Failed to load checkpoint directly, trying to load just state_dict. Error: {e}")
        # If above fails, assume it's just the state dict
        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        
    print(f"Loaded model from {path} | Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    return model

def get_model_weights(model):
    """Flattens and returns model weights as a numpy array."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def adapt_model(model, data_loader, criterion, device, adaptation_steps=10, lr=0.01):
    """Adapt a model on a few batches of data."""
    weights_trajectory = [get_model_weights(model)]
    
    # Use a copy for adaptation to not change the original model
    adapt_model = copy.deepcopy(model)
    optimizer = optim.SGD(adapt_model.parameters(), lr=lr)
    
    adapt_model.train()
    
    data_iter = iter(data_loader)
    for step in tqdm(range(adaptation_steps), desc="Adapting model"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
            
        support_images = batch['support'][0].to(device)
        support_labels = batch['support_labels'][0].to(device)
        
        optimizer.zero_grad()
        outputs = adapt_model(support_images)
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            outputs = outputs[:, 1] - outputs[:, 0]
        else:
            outputs = outputs.squeeze()

        loss = criterion(outputs, support_labels.float())
        loss.backward()
        optimizer.step()
        
        weights_trajectory.append(get_model_weights(adapt_model))
        
    return np.array(weights_trajectory)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models ---
    print("--- Loading Models ---")
    meta_model = load_model(args.meta_model_path, device)
    vanilla_model = load_model(args.vanilla_model_path, device)

    # Check for architecture mismatch
    meta_params = sum(p.numel() for p in meta_model.parameters())
    vanilla_params = sum(p.numel() for p in vanilla_model.parameters())
    if meta_params != vanilla_params:
        print(f"ERROR: Model parameter count mismatch!")
        print(f"Meta model params: {meta_params}")
        print(f"Vanilla model params: {vanilla_params}")
        return

    # --- Load Data for Adaptation ---
    print("\n--- Loading Adaptation Data (PB) ---")
    # We use MetaPBDataset as it provides convenient support/query splits
    pb_dataset = MetaDatasetGenerator(
        data_dir=args.data_dir,
        task='all', 
        split='train',
        support_size=10,
        query_size=10
    )
    pb_loader = torch.utils.data.DataLoader(pb_dataset, batch_size=1, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()

    # --- Adapt Models ---
    print("\n--- Adapting Meta-Trained Model ---")
    meta_weights = adapt_model(meta_model, pb_loader, criterion, device, args.adaptation_steps, args.adaptation_lr)
    
    print("\n--- Adapting Vanilla-Trained Model ---")
    vanilla_weights = adapt_model(vanilla_model, pb_loader, criterion, device, args.adaptation_steps, args.adaptation_lr)

    # --- Perform PCA ---
    print("\n--- Performing PCA ---")
    all_weights = np.vstack([meta_weights, vanilla_weights])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(all_weights)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # Split back into meta and vanilla trajectories
    meta_pcs = principal_components[:len(meta_weights)]
    vanilla_pcs = principal_components[len(meta_weights):]

    # --- Plotting ---
    print("\n--- Plotting Results ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot meta trajectory
    ax.plot(meta_pcs[:, 0], meta_pcs[:, 1], '-o', label='Meta-Trained', color='b', markersize=8, markeredgecolor='k')
    ax.plot(meta_pcs[0, 0], meta_pcs[0, 1], 's', color='b', markersize=12, markeredgecolor='k', label='Meta Start')

    # Plot vanilla trajectory
    ax.plot(vanilla_pcs[:, 0], vanilla_pcs[:, 1], '-o', label='Vanilla-Trained', color='r', markersize=8, markeredgecolor='k')
    ax.plot(vanilla_pcs[0, 0], vanilla_pcs[0, 1], 's', color='r', markersize=12, markeredgecolor='k', label='Vanilla Start')

    ax.set_title(f'PCA of Conv6 Weights during Adaptation (Naturalistic -> PB)', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(fontsize=10)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    save_path = os.path.join(args.save_dir, f'naturalistic_to_pb_pca_meta_{args.meta_seed}_vanilla_{args.vanilla_seed}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize cross-domain adaptation via PCA.')
    parser.add_argument('--meta_model_path', type=str, required=True, help='Path to the meta-trained model.')
    parser.add_argument('--vanilla_model_path', type=str, required=True, help='Path to the vanilla-trained model.')
    parser.add_argument('--meta_seed', type=int, required=True, help='Seed of the meta-trained model for file naming.')
    parser.add_argument('--vanilla_seed', type=int, required=True, help='Seed of the vanilla-trained model for file naming.')
    parser.add_argument('--data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb', help='Directory for the adaptation dataset (PB).')
    parser.add_argument('--save_dir', type=str, default='./visualizations/domain_adaptation_pca', help='Directory to save the plot.')
    parser.add_argument('--adaptation_steps', type=int, default=15, help='Number of adaptation steps.')
    parser.add_argument('--adaptation_lr', type=float, default=0.01, help='Learning rate for adaptation.')
    args = parser.parse_args()
    main(args) 