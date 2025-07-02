"""
Shared utilities for baseline models.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import glob
from PIL import Image
from torchvision import transforms
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py


ARCHITECTURES = ['conv2', 'conv4', 'conv6']
SEEDS = [47, 48, 49, 50, 51] 
PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 'random_color', 
            'arrows', 'irregular', 'filled', 'original'] 

class SameDifferentDataset(Dataset):
    """Dataset for PB tasks with balanced task representation."""
    def __init__(self, data_dir, task_names, split='train', support_sizes=[4, 6, 8, 10], query_size=None, transform=None):
        self.data_dir = data_dir
        self.task_names = task_names if isinstance(task_names, list) else [task_names]
        self.split = split
        self.support_sizes = support_sizes
        self.query_size = query_size
        self.transform = transform
        
        self.task_data = {task: {'episodes': []} for task in self.task_names}
        
        for task_name in self.task_names:
            for support_size in support_sizes:
                filename = f"{task_name}_support{support_size}_{split}.h5"
                filepath = os.path.join(data_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}")
                    continue
                
                print(f"Loading {filename}")
                with h5py.File(filepath, 'r') as f:
                    num_episodes = f['support_images'].shape[0]
                    for episode_idx in range(num_episodes):
                        self.task_data[task_name]['episodes'].append({
                            'support_images': f['support_images'][episode_idx],
                            'support_labels': f['support_labels'][episode_idx],
                            'query_images': f['query_images'][episode_idx],
                            'query_labels': f['query_labels'][episode_idx],
                        })
            
            print(f"Loaded {len(self.task_data[task_name]['episodes'])} episodes for task {task_name}")

        self.episodes_per_task = {task: len(data['episodes']) for task, data in self.task_data.items()}
        self.total_episodes = sum(self.episodes_per_task.values())
    
    def __len__(self):
        return self.total_episodes
    
    def __getitem__(self, idx):
        # Determine which task and episode this index corresponds to
        task_name = None
        for task, count in self.episodes_per_task.items():
            if idx < count:
                task_name = task
                episode_idx = idx
                break
            idx -= count
        
        episode = self.task_data[task_name]['episodes'][episode_idx]
        
        support_images = torch.FloatTensor(episode['support_images'].transpose(0, 3, 1, 2)) / 255.0
        support_labels = torch.tensor(episode['support_labels'], dtype=torch.long)
        query_images = torch.FloatTensor(episode['query_images'].transpose(0, 3, 1, 2)) / 255.0
        query_labels = torch.tensor(episode['query_labels'], dtype=torch.long)
        
        return {
            'support_images': support_images, 
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels,
            'task': task_name
        }

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Training"):
        data, labels = batch['image'].to(device), batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            outputs = outputs[:, 1] - outputs[:, 0]
        else:
            outputs = outputs.squeeze()
        
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            data, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(data)
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 1] - outputs[:, 0]
            else:
                outputs = outputs.squeeze()

            loss = criterion(outputs, labels.float())
            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='checkpoint.pt', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.delta = delta
    
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss 