import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from learn2learn.data import MetaDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import math
import random

class SameDifferentDataset(Dataset):
    def __init__(self, data_dir, tasks, split, support_sizes=[4, 6, 8, 10]):
        self.data_dir = data_dir
        self.tasks = tasks
        self.split = split
        self.support_sizes = support_sizes
        
        # Create a list of all possible episode files
        self.episode_files = []
        for task in tasks:
            for support_size in support_sizes:
                file_path = os.path.join(data_dir, f'{task}_support{support_size}_{split}.h5')
                if os.path.exists(file_path):
                    self.episode_files.append({
                        'file_path': file_path,
                        'task': task,
                        'support_size': support_size
                    })
        
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
        for file_info in self.episode_files:
            task = file_info['task']
            num_episodes = self.file_episode_counts[len(self.file_episode_counts)-1]
            self.task_indices[task].extend(
                range(total_idx, total_idx + num_episodes))
            total_idx += num_episodes
    
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
        tasks_per_batch = max(1, batch_size // len(self.tasks))  # Ensure at least 1 episode per task
        
        for task in self.tasks:
            # Make sure we don't request more episodes than available
            available_episodes = len(self.task_indices[task])
            n_episodes = min(tasks_per_batch, available_episodes)
            if n_episodes > 0:
                task_episodes = random.sample(self.task_indices[task], n_episodes)
                episodes.extend([self[idx] for idx in task_episodes])
        
        # If we don't have enough episodes, sample randomly to fill the batch
        while len(episodes) < batch_size:
            task = random.choice(self.tasks)
            idx = random.choice(self.task_indices[task])
            episodes.append(self[idx])
        
        return episodes

def debug_gradients(model, name=""):
    print(f"\nGradient check for {name}:")
    for name, param in model.named_parameters():
        print(f"{name}:")
        print(f"- requires_grad: {param.requires_grad}")
        print(f"- has grad_fn: {param.grad_fn is not None}")
        print(f"- grad: {param.grad is not None}")
        if hasattr(param, 'grad_fn'):
            print(f"- grad_fn type: {type(param.grad_fn)}")

class SameDifferentCNN(nn.Module):
    def __init__(self):
        # 2 conv layers:
        # 3 → 6 → 12
        # MaxPool3d(3,2) reduces spatial dimensions after each conv
        super(SameDifferentCNN, self).__init__()
        
        # Convolutional layers with batch norm
        self.conv1 = nn.Conv2d(3, 6, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(6, 12, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12, track_running_stats=False)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._initialize_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 2)  # Changed to 2D output
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_size(self):
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def accuracy(predictions, targets):
    with torch.no_grad():
        # Convert 2D logits to binary prediction
        predictions = F.softmax(predictions, dim=1)
        predictions = (predictions[:, 1] > 0.5).float()
        # Convert binary targets to match prediction format
        targets = targets.squeeze(1)
        return (predictions == targets).float().mean()

def load_model(model_path, model, optimizer=None):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint

def test_gradient_flow():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SameDifferentCNN().to(device)
    
    # Get a sample input using create_datasets
    train_dataset, _, _ = create_datasets('data/meta_h5', ['regular'])
    episode = train_dataset[0]
    
    # Move data to device
    images = episode['support_images'].to(device)
    labels = episode['support_labels'].unsqueeze(1).to(device)
    
    # Zero gradients
    model.zero_grad()
    
    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels.squeeze(1).long())
    
    # Backward pass
    loss.backward()
    
    # Check gradients layer by layer
    has_gradients = False
    print("\nChecking gradients:")
    
    # Check conv layers
    for i in range(1, 3):
        conv = getattr(model, f'conv{i}')
        if conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0:
            print(f"Conv{i} has gradients")
            has_gradients = True
    
    # Check fc layers
    for i in range(1, 4):
        fc = getattr(model, f'fc{i}')
        if fc.weight.grad is not None and fc.weight.grad.abs().sum() > 0:
            print(f"FC{i} has gradients")
            has_gradients = True
    
    if not has_gradients:
        print("No gradients detected in any layer")
    
    return has_gradients

def create_datasets(data_dir, all_tasks, support_sizes=[4, 6, 8, 10]):
    """Create train, validation, and test datasets."""
    train_dataset = SameDifferentDataset(data_dir, all_tasks, 'train', support_sizes)
    val_dataset = SameDifferentDataset(data_dir, all_tasks, 'val', support_sizes)
    test_dataset = SameDifferentDataset(data_dir, all_tasks, 'test', support_sizes)
    
    print(f"\nDataset Split Info:")
    print(f"Training episodes: {len(train_dataset)}")
    print(f"Validation episodes: {len(val_dataset)}")
    print(f"Test episodes: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def validate(maml, val_dataset, device, meta_batch_size=8, num_adaptation_steps=5, max_episodes=200):
    """Improved validation with consistent episode sampling and larger batch size"""
    maml.module.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    # Use more episodes for more stable metrics
    total_episodes = min(len(val_dataset), max_episodes)
    
    # Pre-select episodes for consistency
    episode_indices = np.random.choice(len(val_dataset), total_episodes, replace=False)
    
    # Track per-task performance
    task_metrics = {task: {'acc': [], 'loss': []} for task in val_dataset.tasks}
    
    num_batches = max(1, total_episodes // meta_batch_size)
    pbar = tqdm(range(num_batches), desc="Validating")
    
    for batch_idx in pbar:
        batch_loss = 0.0
        batch_acc = 0.0
        
        # Get indices for this batch
        batch_start = batch_idx * meta_batch_size
        batch_end = min(batch_start + meta_batch_size, total_episodes)
        batch_indices = episode_indices[batch_start:batch_end]
        
        for idx in batch_indices:
            episode = val_dataset[idx]
            task = episode['task']
            learner = maml.clone()
            
            # Rest of validation logic...
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            # Adapt using support set
            for _ in range(num_adaptation_steps):
                support_preds = learner(support_images)
                support_loss = F.cross_entropy(support_preds, support_labels.squeeze(1).long())
                learner.adapt(support_loss)
            
            # Evaluate on query set
            with torch.no_grad():
                query_preds = learner(query_images)
                query_loss = F.cross_entropy(query_preds, query_labels.squeeze(1).long())
                query_acc = accuracy(query_preds, query_labels)
            
            batch_loss += query_loss.item()
            batch_acc += query_acc.item()
            
            # Track per-task performance
            task_metrics[task]['acc'].append(query_acc.item())
            task_metrics[task]['loss'].append(query_loss.item())
        
        valid_tasks = len(batch_indices)
        if valid_tasks > 0:
            batch_loss /= valid_tasks
            batch_acc /= valid_tasks
            val_loss += batch_loss
            val_acc += batch_acc
            
            # Show per-task accuracies in progress bar
            task_accs = {task: np.mean(metrics['acc']) 
                        for task, metrics in task_metrics.items()}
            
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'acc': f'{batch_acc:.4f}',
                'task_accs': {t: f'{acc:.2f}' for t, acc in task_accs.items()}
            })
    
    # Print detailed validation results
    print("\nValidation Results by Task:")
    for task in task_metrics:
        task_acc = np.mean(task_metrics[task]['acc'])
        task_loss = np.mean(task_metrics[task]['loss'])
        print(f"{task}: Acc = {task_acc:.4f}, Loss = {task_loss:.4f}")
    
    return val_loss / num_batches, val_acc / num_batches

class MetricTracker:
    def __init__(self, alpha=0.95):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

def train_epoch(maml, train_dataset, optimizer, scheduler, device, meta_batch_size, max_batches_per_epoch, num_adaptation_steps=5):
    """Run one training epoch."""
    maml.module.train()
    total_loss = 0.0
    total_acc = 0.0
    
    num_batches = min(max_batches_per_epoch, len(train_dataset) // meta_batch_size)
    pbar = tqdm(range(num_batches), desc='Training')
    
    # Add task-specific metrics
    task_metrics = {task: {'acc': [], 'loss': []} for task in train_dataset.tasks}
    
    for _ in pbar:
        optimizer.zero_grad()
        batch_losses = []  # Store tensor losses
        batch_acc = 0.0
        
        # Get balanced batch
        episodes = train_dataset.get_balanced_batch(meta_batch_size)
        
        if not episodes:  # Check if we got any episodes
            print("Warning: No episodes in batch, skipping...")
            continue
            
        for episode in episodes:
            task = episode['task']
            learner = maml.clone()
            
            try:
                # Move data to device
                support_images = episode['support_images'].to(device)
                support_labels = episode['support_labels'].unsqueeze(1).to(device)
                query_images = episode['query_images'].to(device)
                query_labels = episode['query_labels'].unsqueeze(1).to(device)
                
                # Inner loop adaptation
                for _ in range(num_adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.cross_entropy(support_preds, support_labels.squeeze(1).long())
                    learner.adapt(support_loss)
                
                # Evaluate on query set
                query_preds = learner(query_images)
                query_loss = F.cross_entropy(query_preds, query_labels.squeeze(1).long())
                query_acc = accuracy(query_preds, query_labels)
                
                # Store tensor loss
                batch_losses.append(query_loss)
                batch_acc += query_acc.item()
                
                # Track per-task metrics
                task_metrics[task]['acc'].append(query_acc.item())
                task_metrics[task]['loss'].append(query_loss.item())
                
            except Exception as e:
                print(f"Error processing episode for task {task}: {str(e)}")
                continue
        
        if not batch_losses:  # Check if we have any losses to process
            print("Warning: No valid losses in batch, skipping...")
            continue
            
        # Average losses and backward
        meta_batch_loss = torch.stack(batch_losses).mean()
        meta_batch_loss.backward()
        
        # Add gradient clipping before optimizer step
        max_grad_norm = 1.5
        torch.nn.utils.clip_grad_norm_(maml.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        # Convert to float for logging only after backward()
        batch_loss_value = meta_batch_loss.item()
        
        # Update progress bar with more info
        mean_task_accs = {task: np.mean(metrics['acc'][-10:]) 
                         for task, metrics in task_metrics.items()
                         if metrics['acc']}  # Only compute mean if we have values
        
        pbar.set_postfix({
            'loss': f'{batch_loss_value:.4f}',
            'acc': f'{(batch_acc/len(episodes)):.4f}',  # Use actual number of episodes
            'task_accs': {t: f'{acc:.2f}' for t, acc in mean_task_accs.items()}
        })
        
        total_loss += batch_loss_value
        total_acc += batch_acc / len(episodes)
    
    return total_loss / num_batches, total_acc / num_batches

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_batch_size = 8
    num_epochs = 50
    max_batches_per_epoch = 20
    train_adaptation_steps = 5
    val_adaptation_steps = 15
    test_adaptation_steps = [15, 20, 25]
    max_val_episodes = 200
    
    # Create datasets with merged tasks
    all_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
                 'random_color', 'arrows', 'irregular', 'filled', 'original']
    
    train_dataset, val_dataset, test_dataset = create_datasets(
        'data/meta_h5', all_tasks)
    
    # Create model and MAML
    model = SameDifferentCNN()
    model.to(device)
    
    maml = l2l.algorithms.MAML(model, lr=0.0005,  # Reduced from 0.001
                              first_order=True,
                              allow_unused=True, allow_nograd=True)
    
    opt = torch.optim.Adam(maml.parameters(), lr=0.00005)  # Reduced from 0.0001
    max_grad_norm = 0.5  # Add gradient clipping threshold
    
    # Create scheduler
    num_training_steps = num_epochs * (len(train_dataset) // meta_batch_size)
    num_warmup_steps = num_training_steps // 10
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps)
    
    # Training loop with validation
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(maml, train_dataset, opt, scheduler, 
                                          device, meta_batch_size, max_batches_per_epoch)
        
        # Validation
        val_loss, val_acc = validate(maml, val_dataset, device,
                                   num_adaptation_steps=val_adaptation_steps,
                                   max_episodes=max_val_episodes)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}')
        print(f'Val Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_acc': best_val_acc,
            }, 'best_model.pt')
    
    # Final testing with multiple adaptation steps
    print("\nFinal Testing...")
    for n_steps in test_adaptation_steps:
        test_loss, test_acc = validate(maml, test_dataset, device, 
                                     num_adaptation_steps=n_steps)
        print(f'Test Results ({n_steps} adaptation steps):')
        print(f'Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}')

if __name__ == '__main__':
    if test_gradient_flow():
        main()
    else:
        print("Gradient flow test failed. Please fix before running full training.") 