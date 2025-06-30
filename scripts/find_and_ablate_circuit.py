import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import learn2learn as l2l
from tqdm import tqdm
import os
import h5py
import numpy as np
import argparse
import json
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_baseline.models.conv6lr import SameDifferentCNN

def custom_collate(batch):
    """A simple collate function that returns the batch as a list of dicts."""
    return batch

class SameDifferentDataset(Dataset):
    """
    Custom dataset for the same-different task.
    Loads data from H5 files for a set of specified tasks.
    """
    def __init__(self, data_dir, tasks, split, support_sizes):
        self.data_dir = data_dir
        self.tasks = tasks
        self.split = split
        self.support_sizes = support_sizes
        self.file_paths = self._get_file_paths()
        self.episodes = self._load_episodes()
        print(f"Dataset initialization for {split} split:")
        print(f"Found {len(self.file_paths)} valid files, {len(self.episodes)} total episodes.")

    def _get_file_paths(self):
        file_paths = []
        for task in self.tasks:
            for size in self.support_sizes:
                fname = f"{task}_support{size}_{self.split}.h5"
                path = os.path.join(self.data_dir, fname)
                if os.path.exists(path):
                    file_paths.append(path)
        return file_paths

    def _load_episodes(self):
        episodes = []
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                num_episodes = f['images'].shape[0]
                support_size = f['images'].shape[1]
                for i in range(num_episodes):
                    episodes.append({
                        'file_path': path, 
                        'episode_index': i,
                        'support_size': support_size
                    })
        return episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode_info = self.episodes[idx]
        with h5py.File(episode_info['file_path'], 'r') as f:
            images = torch.from_numpy(f['images'][episode_info['episode_index']]).float()
            labels = torch.from_numpy(f['labels'][episode_info['episode_index']]).long()
            
            support_size = images.size(0) // 2
            support_images = images[:support_size]
            query_images = images[support_size:]
            support_labels = labels[:support_size]
            query_labels = labels[support_size:]

        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels,
            'support_size': episode_info['support_size']
        }

# --- Training and Validation Functions ---
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

            # Adapt the model
            for step in range(adaptation_steps):
                preds = learner(support_images)
                loss = F.cross_entropy(preds, support_labels)
                learner.adapt(loss, allow_unused=True)

            # Evaluate the adapted model on query set
            query_preds = learner(query_images)
            query_loss = F.cross_entropy(query_preds, query_labels)
            query_acc = (query_preds.argmax(dim=1) == query_labels).float().mean()
            
            meta_train_loss += query_loss
            meta_train_acc += query_acc.item()
        
        # Average the loss and acc over the batch and backprop
        meta_train_loss /= len(batch)
        meta_train_acc /= len(batch)
        meta_train_loss.backward()
        optimizer.step()

        total_loss += meta_train_loss.item()
        total_acc += meta_train_acc
            
    return total_loss / len(loader), total_acc / len(loader)

def validate(maml, loader, device, adaptation_steps, ablated_channel=None):
    """Validates the model."""
    maml.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    for batch in tqdm(loader, desc="Validating"):
        meta_val_loss = 0.0
        meta_val_acc = 0.0

        for episode in batch:
            learner = maml.clone()
            
            # If ablating, set the hook
            if ablated_channel is not None:
                # This hook zeros out the specified channel
                def make_ablation_hook(channel_to_ablate):
                    def hook(module, input, output):
                        output[:, channel_to_ablate] = 0
                        return output
                    return hook
                # Register the hook on the final conv layer
                hook = learner.model.layer4[1].register_forward_hook(make_ablation_hook(ablated_channel))

            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)

            # Adapt the model
            for step in range(adaptation_steps):
                preds = learner(support_images)
                loss = F.cross_entropy(preds, support_labels)
                learner.adapt(loss, allow_unused=True)

            # Evaluate the adapted model
            with torch.no_grad():
                query_preds = learner(query_images)
                query_loss = F.cross_entropy(query_preds, query_labels)
                query_acc = (query_preds.argmax(dim=1) == query_labels).float().mean()
            
            meta_val_loss += query_loss.item()
            meta_val_acc += query_acc.item()

            # Remove hook
            if ablated_channel is not None:
                hook.remove()

        total_loss += meta_val_loss / len(batch)
        total_acc += meta_val_acc / len(batch)
            
    return total_loss / len(loader), total_acc / len(loader)


def train(args):
    """Main function to train the baseline model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    PB_TASKS = [
        "trignometric", "spatial", "wider_line", "scrambled", "open",
        "filled", "rounded", "simple", "arrows", "lines"
    ]
    train_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'train', support_sizes=args.support_size)
    val_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'val', support_sizes=args.val_support_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)

    # --- Model & MAML Setup ---
    model = SameDifferentCNN(num_classes=2)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False, allow_unused=True)
    optimizer = torch.optim.Adam(maml.parameters(), lr=args.meta_lr)

    # --- Training Loop ---
    print("Starting baseline training...")
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(maml, train_loader, optimizer, device, args.adaptation_steps)
        val_loss, val_acc = validate(maml, val_loader, device, args.adaptation_steps)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  New best validation accuracy! Saving model...")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'baseline_model.pt'))
    print("Baseline training finished.")

def find_circuit(args):
    """Identifies the most critical channel by ablating one by one."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Model ---
    model = SameDifferentCNN(num_classes=2)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'baseline_model.pt')))
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False, allow_unused=True)

    # --- Data Loading ---
    PB_TASKS = [
        "trignometric", "spatial", "wider_line", "scrambled", "open",
        "filled", "rounded", "simple", "arrows", "lines"
    ]
    val_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'val', support_sizes=args.val_support_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)

    # --- Base Performance ---
    print("Evaluating base model performance...")
    base_val_loss, base_val_acc = validate(maml, val_loader, device, args.adaptation_steps)
    print(f"Base model validation accuracy: {base_val_acc:.4f}")

    # --- Ablation Analysis ---
    num_channels = model.layer4[1].out_channels
    results = []
    print(f"\nStarting ablation analysis on {num_channels} channels...")
    for channel in tqdm(range(num_channels), desc="Ablating Channels"):
        val_loss, val_acc = validate(maml, val_loader, device, args.adaptation_steps, ablated_channel=channel)
        performance_drop = base_val_acc - val_acc
        results.append({'channel': channel, 'accuracy': val_acc, 'drop': performance_drop})
        
    # --- Find and Save Critical Channel ---
    critical_channel = max(results, key=lambda x: x['drop'])
    print("\nAblation results:")
    for res in sorted(results, key=lambda x: x['drop'], reverse=True)[:10]:
         print(f"  Channel {res['channel']}: Acc = {res['accuracy']:.4f}, Drop = {res['drop']:.4f}")

    print(f"\nMost critical channel is {critical_channel['channel']} with a performance drop of {critical_channel['drop']:.4f}")
    
    # Save the critical channel info
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'critical_channel.json'), 'w') as f:
        json.dump(critical_channel, f, indent=4)
    print("Critical channel information saved.")

def train_ablated(args):
    """Trains a model from scratch with a specific channel permanently ablated."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Critical Channel ---
    with open(os.path.join(args.output_dir, 'critical_channel.json'), 'r') as f:
        critical_channel_info = json.load(f)
    critical_channel = critical_channel_info['channel']
    print(f"Training ablated model with channel {critical_channel} permanently zeroed out.")

    # --- Data Loading ---
    PB_TASKS = [
        "trignometric", "spatial", "wider_line", "scrambled", "open",
        "filled", "rounded", "simple", "arrows", "lines"
    ]
    train_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'train', support_sizes=args.support_size)
    val_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'val', support_sizes=args.val_support_size)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)

    # --- Model & MAML Setup ---
    model = SameDifferentCNN(num_classes=2, ablated_channel=critical_channel) # Pass ablated channel to model
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False, allow_unused=True)
    optimizer = torch.optim.Adam(maml.parameters(), lr=args.meta_lr)

    # --- Training Loop ---
    print("Starting ablated model training...")
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(maml, train_loader, optimizer, device, args.adaptation_steps)
        val_loss, val_acc = validate(maml, val_loader, device, args.adaptation_steps)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  New best validation accuracy! Saving ablated model...")
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'ablated_model.pt'))
    print("Ablated model training finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and analyze circuits in a MAML model.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'find_circuit', 'train_ablated'],
                        help="The mode to run the script in.")
    
    # --- Directory and Path Arguments ---
    parser.add_argument('--pb_data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb',
                        help="Path to the meta-h5/pb data directory.")
    parser.add_argument('--output_dir', type=str, default='circuit_analysis_results',
                        help="Directory to save models and results.")

    # --- Training Arguments ---
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Meta-batch size.")
    parser.add_argument('--support_size', type=int, nargs='+', default=[2, 4, 6, 8, 10], 
                        help="List of support set sizes for training.")
    parser.add_argument('--val_support_size', type=int, nargs='+', default=[2, 4, 6, 8, 10], 
                        help="List of support set sizes for validation.")
    parser.add_argument('--meta_lr', type=float, default=0.001, help="Meta-learning rate for the outer loop.")
    parser.add_argument('--fast_lr', type=float, default=0.01, help="Learning rate for the inner loop adaptation.")
    parser.add_argument('--adaptation_steps', type=int, default=5, help="Number of adaptation steps in the inner loop.")

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'find_circuit':
        find_circuit(args)
    elif args.mode == 'train_ablated':
        # We need to modify the model definition to accept an ablated channel
        # This part requires modifying SameDifferentCNN in conv6lr.py
        # For now, let's just run the training
        # But we need to actually implement the ablation in the model itself
        print("Warning: The 'train_ablated' mode requires the SameDifferentCNN model to be modified to accept an 'ablated_channel' argument and zero out the weights of that channel. This is not yet implemented.")
        # Before calling train_ablated(args), we would need to ensure the model architecture supports it.
        # As a placeholder:
        # train_ablated(args)
        pass # Placeholder until model is modified.
