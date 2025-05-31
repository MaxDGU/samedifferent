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
from torchvision.models import resnet50, ResNet50_Weights

def create_datasets(data_dir, train_tasks, test_task='original', support_sizes=[4, 6, 8, 10]):
    """Create train, validation, and test datasets with held-out task."""
    # Test dataset only contains the held-out task
    test_dataset = SameDifferentDataset(data_dir, [test_task], 'test', support_sizes)
    
    # Train and val datasets contain all other tasks
    train_dataset = SameDifferentDataset(data_dir, train_tasks, 'train', support_sizes)
    val_dataset = SameDifferentDataset(data_dir, train_tasks, 'val', support_sizes)
    
    print(f"\nDataset Split Info:")
    print(f"Training episodes: {len(train_dataset)} ({len(train_tasks)} tasks)")
    print(f"Validation episodes: {len(val_dataset)} ({len(train_tasks)} tasks)")
    print(f"Test episodes: {len(test_dataset)} (held-out task: {test_task})")
    
    return train_dataset, val_dataset, test_dataset

class SameDifferentResNet50(nn.Module):
    def __init__(self):
        super(SameDifferentResNet50, self).__init__()
        # Load pretrained ResNet-50 (ImageNet weights)
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove the original FC layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove final FC layer
        
        # New classifier with 1024 hidden units
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        
        # Set batch norm to not track running stats for meta-learning
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
        
        # Initialize the new layers
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize learned learning rates for each layer group
        # ResNet layer groups
        self.lr_conv1 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_bn1 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_layer1 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_layer2 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_layer3 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_layer4 = nn.Parameter(torch.ones(1) * 0.01)
        
        # Classifier learning rates
        self.lr_classifier = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.01) for _ in range(2)  # 2 linear layers
        ])
    
    def get_layer_lrs(self):
        """Return a dictionary mapping parameters to their learning rates"""
        lrs = {}
        
        # ResNet main layers
        for name, param in self.resnet.conv1.named_parameters():
            lrs[f'resnet.conv1.{name}'] = self.lr_conv1.abs()
        
        for name, param in self.resnet.bn1.named_parameters():
            lrs[f'resnet.bn1.{name}'] = self.lr_bn1.abs()
        
        for i, layer in enumerate([self.resnet.layer1, self.resnet.layer2, 
                                 self.resnet.layer3, self.resnet.layer4]):
            lr = getattr(self, f'lr_layer{i+1}').abs()
            for name, param in layer.named_parameters():
                lrs[f'resnet.layer{i+1}.{name}'] = lr
        
        # Classifier layers - only get learning rates for linear layers
        linear_idx = 0
        for name, module in self.classifier.named_children():
            if isinstance(module, nn.Linear):
                for param_name, param in module.named_parameters():
                    lrs[f'classifier.{name}.{param_name}'] = self.lr_classifier[linear_idx].abs()
                linear_idx += 1
        
        return lrs
    
    def forward(self, x):
        # ResNet feature extraction (up to global average pooling)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Global average pooling
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        return self.classifier(x)

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
        
        if not self.episode_files:
            print(f"Warning: No files found for tasks {tasks} in split {split}")
            print(f"Looked in directory: {data_dir}")
            print(f"Tried pattern: task_support_N_split.h5 (N in {support_sizes})")
        
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
        
        # Convert to NCHW format and normalize using ImageNet stats
        support_images = episode['support_images'].permute(0, 3, 1, 2)
        query_images = episode['query_images'].permute(0, 3, 1, 2)
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        support_images = (support_images / 255.0 - mean) / std
        query_images = (query_images / 255.0 - mean) / std
        
        return {
            'support_images': support_images,
            'support_labels': episode['support_labels'],
            'query_images': query_images,
            'query_labels': episode['query_labels'],
            'task': episode['task'],
            'support_size': episode['support_size']
        }
    
    def get_balanced_batch(self, batch_size):
        """Get a batch with equal representation from each task"""
        episodes = []
        tasks_per_batch = max(1, batch_size // len(self.tasks))
        
        for task in self.tasks:
            available_episodes = len(self.task_indices[task])
            n_episodes = min(tasks_per_batch, available_episodes)
            if n_episodes > 0:
                task_episodes = random.sample(self.task_indices[task], n_episodes)
                episodes.extend([self[idx] for idx in task_episodes])
        
        while len(episodes) < batch_size:
            task = random.choice(self.tasks)
            idx = random.choice(self.task_indices[task])
            episodes.append(self[idx])
        
        return episodes

def accuracy(predictions, targets):
    with torch.no_grad():
        predictions = F.softmax(predictions, dim=1)
        predictions = (predictions[:, 1] > 0.5).float()
        targets = targets.squeeze(1)
        return (predictions == targets).float().mean()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

class EarlyStopping:
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
    """Improved validation with learned per-layer learning rates"""
    maml.module.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    # Calculate number of batches to ensure all tasks are represented
    num_tasks = len(val_dataset.tasks)
    episodes_per_task = max_episodes // num_tasks
    num_batches = max_episodes // meta_batch_size
    
    task_metrics = {task: {'acc': [], 'loss': []} for task in val_dataset.tasks}
    pbar = tqdm(range(num_batches), desc="Validating")
    
    for _ in pbar:
        batch_loss = 0.0
        batch_acc = 0.0
        
        # Get balanced batch like in training
        episodes = val_dataset.get_balanced_batch(meta_batch_size)
        
        for episode in episodes:
            task = episode['task']
            learner = maml.clone()
            
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            # Inner loop adaptation
            layer_lrs = learner.module.get_layer_lrs()
            for _ in range(num_adaptation_steps):
                support_preds = learner(support_images)
                # Use BCE loss for support set
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
                # Use BCE loss for query set
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
        
        # Calculate running task accuracies
        task_accs = {task: np.mean(metrics['acc']) if metrics['acc'] else 0.0
                    for task, metrics in task_metrics.items()}
        
        # Show progress with task-specific accuracies
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
    """Custom collate function to handle episodes"""
    return batch

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()  # Only use mixed precision with CUDA
    torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
    meta_batch_size = 32
    num_epochs = 100
    adaptation_steps = 5
    gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches
    
    print(f"\nUsing device: {device}")
    if use_amp:
        print("Using mixed precision training")
    
    # Split tasks
    all_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
                 'random_color', 'arrows', 'irregular', 'filled', 'original']
    test_task = 'original'
    train_tasks = [task for task in all_tasks if task != test_task]
    
    # Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets(
        'data/meta_h5', train_tasks, test_task)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=meta_batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,  # No multiprocessing on CPU
        pin_memory=device.type == 'cuda',
        collate_fn=collate_episodes,  # Use separate collate function
        prefetch_factor=2 if device.type == 'cuda' else None  # Only prefetch on GPU
    )
    
    # Create model and move to GPU
    model = SameDifferentResNet50()
    model = model.to(device)
    
    maml = l2l.algorithms.MAML(model, 
                              lr=None,  # Using learned per-layer learning rates
                              first_order=True,
                              allow_unused=True, 
                              allow_nograd=True)
    
    opt = torch.optim.AdamW(maml.parameters(), 
                           lr=0.001,
                           weight_decay=0.01,
                           amsgrad=True)
    
    # Learning rate scheduler with warmup
    num_training_steps = num_epochs * (len(train_loader) // gradient_accumulation_steps)
    num_warmup_steps = num_training_steps // 10  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=10)
    
    # Initialize scaler only if using CUDA
    scaler = torch.amp.GradScaler() if use_amp else None
    
    # Training loop with validation
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Train epoch
        train_loss, train_acc = train_epoch(
            maml=maml,
            train_loader=train_loader,
            optimizer=opt,
            scheduler=scheduler,
            device=device,
            meta_batch_size=meta_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            adaptation_steps=adaptation_steps,
            scaler=scaler,
            use_amp=use_amp
        )
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}')
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss, val_acc = validate(maml, val_dataset, device)
            print(f'Val Loss = {val_loss:.4f}, Acc = {val_acc:.4f}')
            
            # Early stopping check
            early_stopping(val_acc)
            if early_stopping.should_stop:
                print("Early stopping triggered!")
                break
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': best_val_acc,
                }, 'best_resnet50_model.pt')
        else:
            print(f'Skipping validation.')
    
    # Final evaluation on held-out test task
    print("\nEvaluating on held-out test task...")
    model.eval()
    test_loss, test_acc = validate(maml, test_dataset, device)
    print(f'\nFinal Test Results on Held-out Task ({test_task}):')
    print(f'Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}')

def train_epoch(maml, train_loader, optimizer, scheduler, device, meta_batch_size, 
             gradient_accumulation_steps, adaptation_steps, scaler, use_amp):
    """Run one training epoch with optional mixed precision."""
    maml.module.train()
    running_loss = 0.0
    running_acc = 0.0
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
               desc='Training')
    
    for batch_idx, episodes in pbar:
        batch_losses = []
        batch_acc = 0.0
        
        for episode in episodes:
            learner = maml.clone()
            
            # Move data to GPU
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            # Inner loop adaptation with optional mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    layer_lrs = learner.module.get_layer_lrs()
                    for _ in range(adaptation_steps):
                        support_preds = learner(support_images)
                        # Use BCE loss for support set
                        support_loss = F.binary_cross_entropy_with_logits(
                            support_preds[:, 1], support_labels.squeeze(1).float())
                        
                        grads = torch.autograd.grad(support_loss, learner.parameters(),
                                                  create_graph=True, allow_unused=True)
                        
                        for (name, param), grad in zip(learner.named_parameters(), grads):
                            if grad is not None:
                                lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                                param.data = param.data - lr.abs() * grad
                    
                    # Compute query loss
                    query_preds = learner(query_images)
                    # Use BCE loss for query set
                    query_loss = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.squeeze(1).float())
                    query_acc = accuracy(query_preds, query_labels)
            else:
                layer_lrs = learner.module.get_layer_lrs()
                for _ in range(adaptation_steps):
                    support_preds = learner(support_images)
                    # Use BCE loss for support set
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.squeeze(1).float())
                    
                    grads = torch.autograd.grad(support_loss, learner.parameters(),
                                              create_graph=True, allow_unused=True)
                    
                    for (name, param), grad in zip(learner.named_parameters(), grads):
                        if grad is not None:
                            lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                            param.data = param.data - lr.abs() * grad
                
                # Compute query loss
                query_preds = learner(query_images)
                # Use BCE loss for query set
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                query_acc = accuracy(query_preds, query_labels)
            
            # Scale loss and accumulate gradients
            scaled_loss = query_loss / (meta_batch_size * gradient_accumulation_steps)
            
            if use_amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            batch_losses.append(scaled_loss.item())
            batch_acc += query_acc.item()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=5.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        running_loss += sum(batch_losses)
        running_acc += batch_acc / len(episodes)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{running_acc/(batch_idx+1):.4f}'
        })
    
    num_batches = len(train_loader)
    return running_loss / num_batches, running_acc / num_batches

if __name__ == '__main__':
    main() 