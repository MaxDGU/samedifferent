import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import glob
import json
import argparse
import sys

# Add the root directory to the path to allow imports from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the dataset and collation function from the existing script
from holdout_experiment.conv6lr import SameDifferentDataset, collate_episodes

# We need a conv6 model that does *not* have per-layer learning rates.
# I will define a simplified one here to avoid touching the original files.
class SimpleConv6(nn.Module):
    def __init__(self):
        super(SimpleConv6, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Placeholder to calculate flattened size
        self._to_linear = None
        self._initialize_size()

        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def _initialize_size(self):
        if self._to_linear is None:
            x = torch.randn(1, 3, 128, 128)
            x = self.features(x)
            self._to_linear = x.reshape(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

def accuracy(predictions, targets):
    """Computes binary accuracy."""
    predictions = (torch.sigmoid(predictions[:, 1]) > 0.5).long()
    return (predictions == targets).float().mean().item()

def train(args):
    """Main training loop."""
    print(f"Starting training with args: {args}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cpu") # Forcing CPU for local training
    print(f"Using device: {device}")

    # --- Data Loading ---
    all_tasks = [os.path.basename(p).replace('_support10_train.h5', '') 
                 for p in glob.glob(os.path.join(args.pb_data_dir, '*_train.h5'))]
    train_tasks = [t for t in all_tasks if t != 'regular']
    
    train_dataset = SameDifferentDataset(
        data_dir=args.pb_data_dir,
        tasks=train_tasks,
        split='train',
        support_sizes=[10]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.meta_batch_size,
        shuffle=True,
        collate_fn=collate_episodes,
        num_workers=2
    )

    # --- Model & MAML Setup ---
    model = SimpleConv6().to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False)
    optimizer = torch.optim.Adam(maml.parameters(), lr=args.meta_lr)
    
    print("\nStarting training loop...")
    for iteration in range(args.num_iterations):
        optimizer.zero_grad()
        meta_train_loss = 0.0
        meta_train_acc = 0.0
        
        # Sample a batch of episodes
        batch = next(iter(train_loader))

        for episode in batch:
            learner = maml.clone()
            
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)

            # Adapt the model
            for _ in range(args.adaptation_steps):
                preds = learner(support_images)
                loss = F.cross_entropy(preds, support_labels.long())
                learner.adapt(loss, allow_unused=True)

            # Evaluate on query set
            query_preds = learner(query_images)
            query_loss = F.cross_entropy(query_preds, query_labels.long())
            
            meta_train_loss += query_loss
            meta_train_acc += accuracy(query_preds, query_labels.long())

        # Average losses and accuracies and update main model
        meta_train_loss /= len(batch)
        meta_train_acc /= len(batch)
        meta_train_loss.backward()
        optimizer.step()
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}/{args.num_iterations} | Loss: {meta_train_loss.item():.3f} | Acc: {meta_train_acc:.3f}")

    print("\nTraining finished.")
    # --- Save the model ---
    model_path = os.path.join(args.output_dir, 'local_model.pt')
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ablation training script for local development.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output_dir', type=str, default='results/local_ablations', help='Directory to save results.')
    parser.add_argument('--pb_data_dir', type=str, default='data/meta_h5/pb', help='Path to PB data.')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='Meta learning rate.')
    parser.add_argument('--fast_lr', type=float, default=0.05, help='Adaptation learning rate.')
    parser.add_argument('--meta_batch_size', type=int, default=8, help='Number of episodes per meta-batch.')
    parser.add_argument('--adaptation_steps', type=int, default=5, help='Number of adaptation steps.')
    parser.add_argument('--num_iterations', type=int, default=100, help='Total number of meta-updates to run.')
    
    args = parser.parse_args()
    train(args) 