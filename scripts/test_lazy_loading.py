#!/usr/bin/env python3
"""
Test the lazy loading VanillaPBDataset implementation
"""

import os
import sys
import h5py
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class VanillaPBDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that loads H5 data on-demand for vanilla SGD.
    Stores file paths and indices, loads actual data lazily in __getitem__.
    """
    def __init__(self, tasks, split='train', data_dir='data/meta_h5/pb', support_sizes=[4, 6, 8, 10]):
        self.data_dir = data_dir
        self.split = split
        self.sample_info = []  # List of (filepath, episode_idx, sample_type, sample_idx_in_episode)
        
        print(f"Indexing vanilla dataset for {split} split...")
        total_samples = 0
        
        for task in tasks:
            for support_size in support_sizes:
                filename = f"{task}_support{support_size}_{split}.h5"
                filepath = os.path.join(data_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}")
                    continue
                
                print(f"Indexing {filename}...")
                with h5py.File(filepath, 'r') as f:
                    num_episodes = f['support_images'].shape[0]
                    support_size_actual = f['support_images'].shape[1]
                    query_size_actual = f['query_images'].shape[1]
                    
                    # Index support samples
                    for episode_idx in range(num_episodes):
                        for sample_idx in range(support_size_actual):
                            self.sample_info.append((filepath, episode_idx, 'support', sample_idx))
                            total_samples += 1
                    
                    # Index query samples  
                    for episode_idx in range(num_episodes):
                        for sample_idx in range(query_size_actual):
                            self.sample_info.append((filepath, episode_idx, 'query', sample_idx))
                            total_samples += 1
        
        print(f"Indexed {total_samples} individual samples for vanilla SGD ({split} split)")
    
    def __len__(self):
        return len(self.sample_info)
    
    def __getitem__(self, idx):
        if idx >= len(self.sample_info):
            raise IndexError("Index out of range")
        
        filepath, episode_idx, sample_type, sample_idx = self.sample_info[idx]
        
        # Load data on-demand
        with h5py.File(filepath, 'r') as f:
            if sample_type == 'support':
                image = f['support_images'][episode_idx, sample_idx]  # Shape: (H, W, C)
                label = f['support_labels'][episode_idx, sample_idx]  # Scalar
            else:  # query
                image = f['query_images'][episode_idx, sample_idx]  # Shape: (H, W, C)
                label = f['query_labels'][episode_idx, sample_idx]  # Scalar
        
        # Convert to tensor and normalize
        # Convert from HWC to CHW format and normalize to [0, 1]
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        label = torch.tensor(int(label), dtype=torch.long)
        
        return image, label

def test_lazy_loading():
    print("Testing lazy loading VanillaPBDataset...")
    
    # Test dataset initialization
    data_dir = "/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb"
    if not os.path.exists(data_dir):
        data_dir = "data/meta_h5/pb"  # Local fallback
    
    # Create dataset with just one task and one support size for testing
    tasks = ['filled']
    support_sizes = [10]
    
    dataset = VanillaPBDataset(tasks, split='val', data_dir=data_dir, support_sizes=support_sizes)
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No data found! Check data paths.")
        return
    
    # Test getting individual samples
    print("\nTesting individual sample access:")
    for i in range(min(5, len(dataset))):
        try:
            image, label = dataset[i]
            print(f"Sample {i}: image shape {image.shape}, label {label}, label type {type(label)}")
        except Exception as e:
            print(f"Error accessing sample {i}: {e}")
    
    # Test dataloader
    print("\nTesting DataLoader:")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    try:
        batch = next(iter(dataloader))
        images, labels = batch
        print(f"Batch: images shape {images.shape}, labels shape {labels.shape}")
        print(f"Images dtype: {images.dtype}, Labels dtype: {labels.dtype}")
        print(f"Image value range: {images.min():.3f} to {images.max():.3f}")
    except Exception as e:
        print(f"Error with DataLoader: {e}")

if __name__ == '__main__':
    test_lazy_loading() 