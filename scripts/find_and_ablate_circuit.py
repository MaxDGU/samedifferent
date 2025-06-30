import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import json
import argparse
import sys
import gc
import glob
import h5py
from torch.utils.data import Dataset

# Add the root directory to the path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_baseline.models.conv6lr import SameDifferentCNN
from meta_baseline.models.utils_meta import collate_episodes

class SameDifferentDataset(Dataset):
    def __init__(self, data_dir, tasks, split, support_sizes=[4, 6, 8, 10]):
        self.data_dir = data_dir
        self.tasks = tasks
        self.split = split
        self.support_sizes = support_sizes
        
        self.episode_files = []
        for task in tasks:
            for support_size in support_sizes:
                file_path = os.path.join(data_dir, f'{task}_support{support_size}_{split}.h5')
                if os.path.exists(file_path):
                    with h5py.File(file_path, 'r') as f:
                        num_episodes = f['support_images'].shape[0]
                        self.episode_files.append({
                            'file_path': file_path,
                            'task': task,
                            'support_size': support_size,
                            'num_episodes': num_episodes
                        })
        
        if not self.episode_files:
            raise ValueError(f"No valid files found for tasks {tasks} in {data_dir}")
        
        self.total_episodes = sum(f['num_episodes'] for f in self.episode_files)
        print(f"\nDataset initialization for {split} split:")
        print(f"Found {len(self.episode_files)} valid files, {self.total_episodes} total episodes.")

    def __len__(self):
        return self.total_episodes
    
    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.episode_files[file_idx]['num_episodes']:
            idx -= self.episode_files[file_idx]['num_episodes']
            file_idx += 1
        
        file_info = self.episode_files[file_idx]
        
        with h5py.File(file_info['file_path'], 'r') as f:
            support_images = torch.from_numpy(f['support_images'][idx]).float()
            support_labels = torch.from_numpy(f['support_labels'][idx]).long()
            query_images = torch.from_numpy(f['query_images'][idx]).float()
            query_labels = torch.from_numpy(f['query_labels'][idx]).long()
        
        support_images = (support_images.permute(0, 3, 1, 2) / 127.5) - 1.0
        query_images = (query_images.permute(0, 3, 1, 2) / 127.5) - 1.0
        
        return {
            'support_images': support_images,
            'support_labels': support_labels.squeeze(),
            'query_images': query_images,
            'query_labels': query_labels.squeeze(),
            'task': file_info['task'],
        }

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def accuracy(predictions, targets):
    """Computes binary accuracy."""
    predictions = (torch.sigmoid(predictions[:, 1]) > 0.5).long()
    return (predictions == targets).float().mean().item()

def train_one_epoch(maml, loader, optimizer, device, adaptation_steps):
    """Trains the model for one epoch."""
    maml.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        meta_train_loss = 0.0
        meta_train_acc = 0.0

        for episode in batch:
            learner = maml.clone()
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)

            for _ in range(adaptation_steps):
                preds = learner(support_images)
                loss = F.cross_entropy(preds, support_labels.long())
                learner.adapt(loss, allow_unused=True)

            query_preds = learner(query_images)
            query_loss = F.cross_entropy(query_preds, query_labels.long())
            
            meta_train_loss += query_loss
            meta_train_acc += accuracy(query_preds, query_labels.long())

        meta_train_loss /= len(batch)
        meta_train_acc /= len(batch)
        meta_train_loss.backward()
        optimizer.step()
        
        total_loss += meta_train_loss.item()
        total_acc += meta_train_acc
        
    return total_loss / len(loader), total_acc / len(loader)

def validate(maml, loader, device, adaptation_steps, channel_to_ablate=None):
    """Validates the model, with an option to ablate a channel."""
    maml.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    # Ablation hook
    hook = None
    if channel_to_ablate is not None:
        def ablate_hook(module, input, output):
            # Clone the output to avoid modifying it in-place
            output_clone = output.clone()
            # Zero out the specific channel
            output_clone[:, channel_to_ablate] = 0
            return output_clone
        
        # Find the target layer and register the hook
        target_layer = maml.module.features[15] # conv6
        hook = target_layer.register_forward_hook(ablate_hook)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            meta_val_loss = 0.0
            meta_val_acc = 0.0
            for episode in batch:
                learner = maml.clone()
                support_images = episode['support_images'].to(device)
                support_labels = episode['support_labels'].to(device)
                query_images = episode['query_images'].to(device)
                query_labels = episode['query_labels'].to(device)

                for _ in range(adaptation_steps):
                    preds = learner(support_images)
                    loss = F.cross_entropy(preds, support_labels.long())
                    learner.adapt(loss, allow_unused=True)

                query_preds = learner(query_images)
                query_loss = F.cross_entropy(query_preds, query_labels.long())
                
                meta_val_loss += query_loss.item()
                meta_val_acc += accuracy(query_preds, query_labels.long())

            total_loss += meta_val_loss / len(batch)
            total_acc += meta_val_acc / len(batch)

    if hook:
        hook.remove()
            
    return total_loss / len(loader), total_acc / len(loader)

def train(args):
    """Main function to train the baseline model."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Data Loading ---
    PB_TASKS = [
        'regular', 'lines', 'open', 'wider_line', 'scrambled',
        'random_color', 'arrows', 'irregular', 'filled', 'original'
    ]

    train_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'train', support_sizes=args.support_size)
    val_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'val', support_sizes=args.val_support_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_episodes)

    # --- Model & MAML Setup ---
    model = SameDifferentCNN().to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False, allow_unused=True)
    optimizer = torch.optim.Adam(maml.parameters(), lr=args.meta_lr)
    
    best_val_acc = 0.0
    output_dir = os.path.join(args.output_dir, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStarting baseline training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(maml, train_loader, optimizer, device, args.adaptation_steps)
        val_loss, val_acc = validate(maml, val_loader, device, args.adaptation_steps)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model.")
            torch.save(model.state_dict(), os.path.join(output_dir, 'baseline_model.pt'))

    print("\nBaseline training finished.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

def find_circuit(args):
    """Identifies the most critical channel by ablating one by one."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = os.path.join(args.output_dir, f'seed_{args.seed}', 'baseline_model.pt')
    if not os.path.exists(model_path):
        print(f"Error: Baseline model not found at {model_path}.")
        print("Please run in 'train' mode first.")
        return

    # --- Model & MAML Setup ---
    model = SameDifferentCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False, allow_unused=True)

    # --- Data Loading for Validation ---
    PB_TASKS = [
        'regular', 'lines', 'open', 'wider_line', 'scrambled',
        'random_color', 'arrows', 'irregular', 'filled', 'original'
    ]
    val_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'val', support_sizes=args.val_support_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_episodes)

    # --- Base Performance ---
    print("Calculating baseline performance without ablation...")
    base_loss, base_acc = validate(maml, val_loader, device, args.adaptation_steps)
    print(f"Baseline Validation Accuracy: {base_acc:.4f}")

    # --- Ablation Analysis ---
    num_channels_to_test = model.features[15].out_channels # conv6
    results = []

    print(f"\nStarting ablation analysis on {num_channels_to_test} channels...")
    for i in tqdm(range(num_channels_to_test), desc="Ablating Channels"):
        _, ablated_acc = validate(maml, val_loader, device, args.adaptation_steps, channel_to_ablate=i)
        performance_drop = base_acc - ablated_acc
        results.append({'channel': i, 'accuracy': ablated_acc, 'drop': performance_drop})

    # --- Report Results ---
    results.sort(key=lambda x: x['drop'], reverse=True)
    
    print("\n--- Ablation Results (Top 10) ---")
    for res in results[:10]:
        print(f"Channel {res['channel']}: Accuracy = {res['accuracy']:.4f}, Drop = {res['drop']:.4f}")

    critical_channel = results[0]['channel']
    print(f"\nMost critical channel identified: {critical_channel}")
    print("To train a model with this channel ablated, run with:")
    print(f"  --mode train_ablated --channel_to_ablate {critical_channel}")

def train_ablated(args):
    """Trains a model from scratch with a specific channel permanently ablated."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting ablated training on device: {device}")
    print(f"Permanently ablating channel: {args.channel_to_ablate}")

    # --- Data Loading ---
    PB_TASKS = [
        'regular', 'lines', 'open', 'wider_line', 'scrambled',
        'random_color', 'arrows', 'irregular', 'filled', 'original'
    ]

    train_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'train', support_sizes=args.support_size)
    val_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'val', support_sizes=args.val_support_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_episodes)

    # --- Model & MAML Setup ---
    model = SameDifferentCNN().to(device)
    
    # --- Apply Permanent Ablation ---
    if args.channel_to_ablate is not None:
        # Get the target convolutional layer (conv6)
        target_layer = model.features[15]
        
        # Zero out the weights and bias for the specified channel
        with torch.no_grad():
            target_layer.weight[args.channel_to_ablate].zero_()
            if target_layer.bias is not None:
                target_layer.bias[args.channel_to_ablate].zero_()
        
        # Prevent the optimizer from updating these weights
        target_layer.weight.requires_grad = False
        if target_layer.bias is not None:
            target_layer.bias.requires_grad = False
            
        print(f"Channel {args.channel_to_ablate} in conv6 has been permanently ablated.")

    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False, allow_unused=True)
    optimizer = torch.optim.Adam(maml.parameters(), lr=args.meta_lr)
    
    best_val_acc = 0.0
    output_dir = os.path.join(args.output_dir, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nStarting ablated model training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(maml, train_loader, optimizer, device, args.adaptation_steps)
        val_loss, val_acc = validate(maml, val_loader, device, args.adaptation_steps)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy for ablated model: {best_val_acc:.4f}. Saving model.")
            model_name = f'ablated_channel_{args.channel_to_ablate}_model.pt'
            torch.save(model.state_dict(), os.path.join(output_dir, model_name))

    print("\nAblated training finished.")
    print(f"Best validation accuracy for ablated model: {best_val_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Find and ablate circuits in a meta-trained model.")
    
    # General arguments
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'find_circuit', 'train_ablated'], help="Execution mode.")
    parser.add_argument('--pb_data_dir', type=str, default='data/meta_h5/pb', help='Path to PB data.')
    parser.add_argument('--output_dir', type=str, default='results/circuit_ablation', help='Directory to save results.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Meta-batch size.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--support_size', type=int, nargs='+', default=[4, 6, 8, 10], help='List of support sizes for training.')
    parser.add_argument('--val_support_size', type=int, nargs='+', default=[10], help='List of support sizes for validation.')
    parser.add_argument('--adaptation_steps', type=int, default=5, help='Number of adaptation steps.')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate.')
    parser.add_argument('--fast_lr', type=float, default=0.05, help='Adaptation learning rate.')

    # Ablation arguments
    parser.add_argument('--channel_to_ablate', type=int, default=None, help='The channel index to ablate during training.')

    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'find_circuit':
        find_circuit(args)
    elif args.mode == 'train_ablated':
        train_ablated(args) 