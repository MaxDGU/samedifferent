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
    def __init__(self, data_dir, task_names, split='train', support_sizes=[4, 6, 8, 10], transform=None):
        self.data_dir = data_dir
        self.task_names = task_names if isinstance(task_names, list) else [task_names]
        self.split = split
        self.support_sizes = support_sizes
        self.transform = transform
        
        self.task_data = {task: {'images': [], 'labels': []} for task in self.task_names}
        #tasks are defined in a list from 
        for task_name in self.task_names:
            #support sizes go from 4 to 10 in steps of 2
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
                        support_images = f['support_images'][episode_idx]
                        support_labels = f['support_labels'][episode_idx]
                        query_images = f['query_images'][episode_idx]
                        query_labels = f['query_labels'][episode_idx]
                        
                        all_images = np.concatenate([support_images, query_images])
                        all_labels = np.concatenate([support_labels, query_labels])
                        
                        self.task_data[task_name]['images'].extend(all_images)
                        self.task_data[task_name]['labels'].extend(all_labels)
            
            self.task_data[task_name]['images'] = np.array(self.task_data[task_name]['images'])
            self.task_data[task_name]['labels'] = np.array(self.task_data[task_name]['labels'])
            
            print(f"Loaded {len(self.task_data[task_name]['images'])} images for task {task_name}")
            print(f"Label distribution: {np.bincount(self.task_data[task_name]['labels'].astype(int))}")
        
        self.total_size = sum(len(data['images']) for data in self.task_data.values())
        self.samples_per_task = min(len(data['images']) for data in self.task_data.values())
        
        self.task_indices = {
            task: np.arange(len(data['images'])) 
            for task, data in self.task_data.items()
        }
        
        for indices in self.task_indices.values():
            np.random.shuffle(indices)
        
        self.current_pos = {task: 0 for task in self.task_names}
    
    def __len__(self):
        #all tasks should have equal representation 
        return self.samples_per_task * len(self.task_names)
    
    def __getitem__(self, idx):
        #lookup for task index 
        task_idx = idx % len(self.task_names)
        task_name = self.task_names[task_idx]
        
        pos = self.current_pos[task_name]
        if pos >= len(self.task_indices[task_name]):
            #reshuffle indices if we've gone through all samples
            np.random.shuffle(self.task_indices[task_name])
            self.current_pos[task_name] = 0
            pos = 0
        
        actual_idx = self.task_indices[task_name][pos]
        self.current_pos[task_name] += 1
        
        image = self.task_data[task_name]['images'][actual_idx]
        label = self.task_data[task_name]['labels'][actual_idx]
        
        # Convert to float and normalize to [-1, 1]
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 127.5 - 1.0
        label = torch.tensor(int(label))
        
        return {'image': image, 'label': label, 'task': task_name}

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, labels in tqdm(loader, desc="Training"):
        data, labels = data.to(device), labels.to(device)
        
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
        for data, labels in tqdm(loader, desc="Validation"):
            data, labels = data.to(device), labels.to(device)
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
    def __init__(self, patience=10, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
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