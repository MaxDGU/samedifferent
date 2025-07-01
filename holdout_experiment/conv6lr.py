import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import math
import random
import glob
import json
import gc
import sys
import argparse

class SameDifferentDataset(Dataset):
    def __init__(self, data_dir, tasks, split, support_sizes=[4, 6, 8, 10]):
        """
        Dataset for loading same-different task data.
        
        Args:
            data_dir (str): Path to data directory
                - For PB data: 'data/pb/pb'
                - For SVRT data: 'data/svrt_fixed'
            tasks (list): List of tasks to load
                - For PB: ['regular', 'lines', 'open', etc.]
                - For SVRT: ['1', '7', '5', etc.]
            split (str): One of ['train', 'val', 'test']
            support_sizes (list): List of support set sizes to use
        """
        self.data_dir = data_dir
        self.tasks = tasks
        self.split = split
        self.support_sizes = support_sizes
        
        # Create a list of all possible episode files
        self.episode_files = []
        for task in tasks:
            for support_size in support_sizes:
                # Handle both PB and SVRT data paths
                if 'svrt_fixed' in data_dir:
                    # SVRT test data path
                    file_path = os.path.join(data_dir, 
                                           f'results_problem_{task}',
                                           f'support{support_size}_{split}.h5')
                else:
                    # PB training/validation data path
                    file_path = os.path.join(data_dir,
                                           f'{task}_support{support_size}_{split}.h5')
                
                if os.path.exists(file_path):
                    self.episode_files.append({
                        'file_path': file_path,
                        'task': task,
                        'support_size': support_size
                    })
        
        if not self.episode_files:
            raise ValueError(f"No valid files found for tasks {tasks} in {data_dir}")
        
        # Calculate total number of episodes
        self.total_episodes = 0
        self.file_episode_counts = []
        for file_info in self.episode_files:
            with h5py.File(file_info['file_path'], 'r') as f:
                num_episodes = f['support_images'].shape[0]
                self.file_episode_counts.append(num_episodes)
                self.total_episodes += num_episodes
        
        # Track episodes per task for balanced sampling
        self.task_indices = {task: [] for task in tasks}
        total_idx = 0
        for i, file_info in enumerate(self.episode_files):
            task = file_info['task']
            num_episodes = self.file_episode_counts[i]
            self.task_indices[task].extend(
                range(total_idx, total_idx + num_episodes))
            total_idx += num_episodes
        
        # Debug prints
        print(f"\nDataset initialization for {split} split:")
        print(f"Found files: {[f['file_path'] for f in self.episode_files]}")
        print(f"Task indices counts: {[(task, len(indices)) for task, indices in self.task_indices.items()]}")
    
    def __len__(self):
        return self.total_episodes
    
    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = 0
        running_count = 0
        while running_count + self.file_episode_counts[file_idx] <= idx:
            running_count += self.file_episode_counts[file_idx]
            file_idx += 1
        
        # Calculate the episode index within the file
        episode_idx = idx - running_count
        
        # Load the episode from the appropriate file
        file_info = self.episode_files[file_idx]
        with h5py.File(file_info['file_path'], 'r') as f:
            episode = {
                'support_images': torch.FloatTensor(f['support_images'][episode_idx]),
                'support_labels': torch.FloatTensor(f['support_labels'][episode_idx]),
                'query_images': torch.FloatTensor(f['query_images'][episode_idx]),
                'query_labels': torch.FloatTensor(f['query_labels'][episode_idx]),
                'task': file_info['task'],
                'support_size': file_info['support_size']
            }
        
        # Convert to NCHW format and normalize
        return {
            'support_images': ((episode['support_images'].permute(0, 3, 1, 2) / 127.5) - 1.0),
            'support_labels': episode['support_labels'],
            'query_images': ((episode['query_images'].permute(0, 3, 1, 2) / 127.5) - 1.0),
            'query_labels': episode['query_labels'],
            'task': episode['task'],
            'support_size': episode['support_size']
        }
    
    def get_balanced_batch(self, batch_size):
        """Get a batch with equal representation from each task"""
        episodes = []
        
        # Filter out tasks with no episodes
        available_tasks = [task for task in self.tasks if self.task_indices[task]]
        
        if not available_tasks:
            raise ValueError(f"No episodes available for any tasks in {self.split} split")
        
        tasks_per_batch = max(1, batch_size // len(available_tasks))
        
        # First, get equal episodes from each available task
        for task in available_tasks:
            # Make sure we don't request more episodes than available
            available_episodes = len(self.task_indices[task])
            n_episodes = min(tasks_per_batch, available_episodes)
            if n_episodes > 0:
                task_episodes = random.sample(self.task_indices[task], n_episodes)
                episodes.extend([self[idx] for idx in task_episodes])
        
        # If we still need more episodes to reach batch_size, sample randomly
        while len(episodes) < batch_size:
            task = random.choice(available_tasks)
            idx = random.choice(self.task_indices[task])
            episodes.append(self[idx])
        
        return episodes

class SameDifferentCNN(nn.Module):
    def __init__(self):
        super(SameDifferentCNN, self).__init__()
        
        # 6-layer CNN with increasing filter counts
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256, track_running_stats=False)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512, track_running_stats=False)
        
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024, track_running_stats=False)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(0.3)
        
        self._to_linear = None
        self._initialize_size()
        
        # FC layers with decreasing sizes
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.LayerNorm(512),
            nn.LayerNorm(256)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(3)
        ])
        
        self.classifier = nn.Linear(256, 2)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Learnable per-layer learning rates
        self.lr_conv = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.01) for _ in range(6)
        ])
        self.lr_fc = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.01) for _ in range(3)
        ])
        self.lr_classifier = nn.Parameter(torch.ones(1) * 0.01)
        
        self._initialize_weights()
    
    def _initialize_size(self):
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear) and m != self.classifier:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.01)
        
        # Initialize classifier with smaller weights for less confident predictions
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout2d(x)
        
        x = x.reshape(x.size(0), -1)
        
        for fc, ln, dropout in zip(self.fc_layers, self.layer_norms, self.dropouts):
            x = dropout(F.relu(ln(fc(x))))
        
        x = self.classifier(x)
        return F.softmax(x / self.temperature.abs(), dim=1)
    
    def get_layer_lrs(self):
        """Return a dictionary mapping parameters to their learning rates"""
        lrs = {}
        
        for i, (conv, bn) in enumerate(zip(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6],
            [self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6]
        )):
            lrs.update({name: self.lr_conv[i].abs() for name, _ in conv.named_parameters()})
            lrs.update({name: self.lr_conv[i].abs() for name, _ in bn.named_parameters()})
        
        for i, (fc, ln) in enumerate(zip(self.fc_layers, self.layer_norms)):
            lrs.update({name: self.lr_fc[i].abs() for name, _ in fc.named_parameters()})
            lrs.update({name: self.lr_fc[i].abs() for name, _ in ln.named_parameters()})
        
        lrs.update({name: self.lr_classifier.abs() for name, _ in self.classifier.named_parameters()})
        
        return lrs

def accuracy(predictions, targets):
    """Calculate binary classification accuracy"""
    with torch.no_grad():
        predictions = F.softmax(predictions, dim=1)
        predictions = (predictions[:, 1] > 0.5).float()
        return (predictions == targets).float().mean()

def load_model(model_path, model, optimizer=None):
    """Load model and optimizer state from checkpoint"""
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

class EarlyStopping:
    """Early stopping with patience and minimum improvement threshold"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val = None
        self.should_stop = False
    
    def __call__(self, val_acc):
        if self.best_val is None:
            self.best_val = val_acc
        elif val_acc < self.best_val + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_val = val_acc   
            self.counter = 0

def validate(maml, val_dataset, device, meta_batch_size=8, num_adaptation_steps=5, max_episodes=200):
    """Validation with learned per-layer learning rates"""
    maml.module.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    num_tasks = len(val_dataset.tasks)
    episodes_per_task = max_episodes // num_tasks
    num_batches = max_episodes // meta_batch_size
    
    task_metrics = {task: {'acc': [], 'loss': []} for task in val_dataset.tasks}
    pbar = tqdm(range(num_batches), desc="Validating")
    
    for _ in pbar:
        batch_loss = 0.0
        batch_acc = 0.0
        
        episodes = val_dataset.get_balanced_batch(meta_batch_size)
        
        for episode in episodes:
            task = episode['task']
            learner = maml.clone()
            
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            layer_lrs = learner.module.get_layer_lrs()
            for _ in range(num_adaptation_steps):
                support_preds = learner(support_images)
                support_loss = F.binary_cross_entropy_with_logits(
                    support_preds[:, 1], support_labels.squeeze(1).float())
                
                grads = torch.autograd.grad(support_loss, learner.parameters(),
                                          create_graph=True,
                                          allow_unused=True)
                
                for (name, param), grad in zip(learner.named_parameters(), grads):
                    if grad is not None:
                        lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                        param.data = param.data - lr.abs() * grad
            
            with torch.no_grad():
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                query_acc = accuracy(query_preds, query_labels)
            
            batch_loss += query_loss.item()
            batch_acc += query_acc.item()
            
            task_metrics[task]['acc'].append(query_acc.item())
            task_metrics[task]['loss'].append(query_loss.item())
        
        batch_loss /= len(episodes)
        batch_acc /= len(episodes)
        val_loss += batch_loss
        val_acc += batch_acc
        
        task_accs = {task: np.mean(metrics['acc']) if metrics['acc'] else 0.0
                    for task, metrics in task_metrics.items()}
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{batch_acc:.4f}',
            'task_accs': {t: f'{acc:.2f}' for t, acc in task_accs.items()}
        })
    
    print("\nValidation Results by Task:")
    for task in task_metrics:
        task_acc = np.mean(task_metrics[task]['acc']) if task_metrics[task]['acc'] else 0.0
        task_loss = np.mean(task_metrics[task]['loss']) if task_metrics[task]['loss'] else 0.0
        print(f"{task}: Acc = {task_acc:.4f}, Loss = {task_loss:.4f}")
    
    return val_loss / num_batches, val_acc / num_batches

def collate_episodes(batch):
    """Collate function that preserves episodes as a list"""
    return batch

def train_epoch(maml, train_loader, optimizer, device, adaptation_steps, scaler):
    """Train using MAML with learned per-layer learning rates"""
    maml.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch_loss = 0
        batch_acc = 0
        optimizer.zero_grad()
        
        for episode in batch:
            try:
                learner = maml.clone()
                
                support_images = episode['support_images'].to(device, non_blocking=True)
                support_labels = episode['support_labels'].to(device, non_blocking=True)
                query_images = episode['query_images'].to(device, non_blocking=True)
                query_labels = episode['query_labels'].to(device, non_blocking=True)
                
                for _ in range(adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.float())
                    learner.adapt(support_loss)
                
                with torch.cuda.amp.autocast():
                    query_preds = learner(query_images)
                    query_loss = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.float())
                    query_acc = accuracy(query_preds, query_labels)
                
                scaled_loss = scaler.scale(query_loss)
                scaled_loss.backward()
                
                batch_loss += query_loss.item()
                batch_acc += query_acc.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: GPU OOM error in batch. Trying to recover...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    continue
                else:
                    raise e
        
        batch_loss /= len(batch)
        batch_acc /= len(batch)
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += batch_loss
        total_acc += batch_acc
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{batch_acc:.4f}'
        })
    
    return total_loss / num_batches, total_acc / num_batches

def main(seed=None, output_dir=None, pb_data_dir='data/pb/pb'):
    
    if seed is None:
        seed = random.randint(1, 10000)
    
    output_dir = output_dir or f"holdout_experiment/results_seed_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Configuration ---
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using seed: {seed}, device: {device}")
    
    # Training parameters
    meta_lr = 0.001
    fast_lr = 0.05
    meta_batch_size = 16
    adaptation_steps = 15
    num_iterations = 2000
    
    # --- 2. Data Loading ---
    # Define tasks: all PB tasks except for 'regular'
    all_tasks = [os.path.basename(p).replace('_support10_train.h5', '') 
                 for p in glob.glob(os.path.join(pb_data_dir, '*_train.h5'))]
    
    train_tasks = [t for t in all_tasks if t != 'regular']
    val_task = ['regular']
    
    print(f"\nTraining on tasks: {train_tasks}")
    print(f"Validating on holdout task: {val_task}\n")
    
    train_dataset = SameDifferentDataset(
        data_dir=pb_data_dir,
        tasks=train_tasks,
        split='train',
        support_sizes=[4, 6, 8, 10]
    )
    
    val_dataset = SameDifferentDataset(
        data_dir=pb_data_dir,
        tasks=val_task,
        split='val',
        support_sizes=[10]
    )
    
    # --- 3. Model & Optimizer ---
    model = SameDifferentCNN().to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False, allow_unused=True)
    
    optimizer = torch.optim.Adam(maml.parameters(), lr=meta_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # --- 4. Training Loop ---
    best_val_acc = 0.0
    early_stopper = EarlyStopping(patience=10, min_delta=0.005)
    
    for epoch in range(3): # Train for 3 epochs
        print(f"\n--- Epoch {epoch+1}/{num_iterations} ---")
        
        # Create a new loader for each epoch to ensure random sampling
        train_loader = DataLoader(train_dataset, 
                                  batch_size=meta_batch_size, 
                                  shuffle=True, 
                                  collate_fn=collate_episodes,
                                  num_workers=4,
                                  pin_memory=True)
        
        train_loss, train_acc = train_epoch(
            maml,
            train_loader,
            optimizer,
            device,
            adaptation_steps,
            scaler
        )
        
        val_acc, _ = validate(
            maml,
            val_dataset,
            device,
            meta_batch_size=meta_batch_size,
            num_adaptation_steps=adaptation_steps
        )
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': maml.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, model_path)
            print(f"Saved new best model to {model_path}")
            
        if early_stopper(val_acc):
            print("Early stopping triggered.")
            break
            
    print("\nTraining finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MAML on holdout PB tasks.")
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results.')
    parser.add_argument('--pb_data_dir', type=str, default='data/meta_h5/pb', help='Path to problem-book data.')
    
    args = parser.parse_args()
    main(seed=args.seed, output_dir=args.output_dir, pb_data_dir=args.pb_data_dir)


