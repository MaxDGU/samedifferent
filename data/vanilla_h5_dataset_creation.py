"""
Vanilla H5 Dataset Creation for PB Tasks

This module provides dataset classes for loading individual image-label pairs
from H5 files for vanilla SGD training (as opposed to episodic meta-learning).
"""

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class PB_dataset_h5(Dataset):
    """
    Dataset for loading individual image-label pairs from PB task H5 files.
    This is used for vanilla SGD training, not meta-learning.
    """
    
    def __init__(self, task, split='train', data_dir='data/vanilla_h5', support_sizes=[4, 6, 8, 10]):
        """
        Initialize the dataset.
        
        Args:
            task (str): PB task name (e.g., 'regular', 'lines', etc.)
            split (str): Data split ('train', 'val', 'test')
            data_dir (str): Directory containing H5 files
            support_sizes (list): Support sizes to load from
        """
        self.task = task
        self.split = split
        self.data_dir = data_dir
        self.support_sizes = support_sizes
        
        self.images = []
        self.labels = []
        
        self._load_data()
    
    def _load_data(self):
        """Load all images and labels from H5 files."""
        for support_size in self.support_sizes:
            filename = f"{self.task}_support{support_size}_{self.split}.h5"
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue
            
            print(f"Loading {filename} for vanilla dataset...")
            with h5py.File(filepath, 'r') as f:
                # Load support images and labels
                if 'support_images' in f and 'support_labels' in f:
                    support_images = f['support_images'][:]  # Shape: (episodes, support_size, H, W, C)
                    support_labels = f['support_labels'][:]  # Shape: (episodes, support_size)
                    
                    # Flatten episodes and support examples into individual samples
                    num_episodes, support_size, H, W, C = support_images.shape
                    support_images = support_images.reshape(-1, H, W, C)  # (episodes * support_size, H, W, C)
                    support_labels = support_labels.reshape(-1)  # (episodes * support_size,)
                    
                    self.images.append(support_images)
                    self.labels.append(support_labels)
                
                # Load query images and labels
                if 'query_images' in f and 'query_labels' in f:
                    query_images = f['query_images'][:]  # Shape: (episodes, query_size, H, W, C)
                    query_labels = f['query_labels'][:]  # Shape: (episodes, query_size)
                    
                    # Flatten episodes and query examples into individual samples
                    num_episodes, query_size, H, W, C = query_images.shape
                    query_images = query_images.reshape(-1, H, W, C)  # (episodes * query_size, H, W, C)
                    query_labels = query_labels.reshape(-1)  # (episodes * query_size,)
                    
                    self.images.append(query_images)
                    self.labels.append(query_labels)
        
        if self.images:
            # Concatenate all images and labels
            self.images = np.concatenate(self.images, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            
            print(f"Loaded {len(self.images)} individual samples for task {self.task}, split {self.split}")
        else:
            print(f"No data loaded for task {self.task}, split {self.split}")
            self.images = np.array([])
            self.labels = np.array([])
    
    def __len__(self):
        """Return the number of individual samples."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single image-label pair.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an int
        """
        if len(self.images) == 0:
            raise IndexError("No data available")
        
        # Get image and label
        image = self.images[idx]  # Shape: (H, W, C)
        label = self.labels[idx]  # Scalar
        
        # Convert to tensor and normalize
        # Convert from HWC to CHW format and normalize to [0, 1]
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        label = torch.tensor(int(label), dtype=torch.long)
        
        return image, label


def test_dataset():
    """Test function to verify the dataset works correctly."""
    print("Testing PB_dataset_h5...")
    
    # Test with a single task
    try:
        dataset = PB_dataset_h5(task='regular', split='train', data_dir='data/meta_h5/pb')
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            image, label = dataset[0]
            print(f"Sample image shape: {image.shape}")
            print(f"Sample label: {label}")
            print(f"Label type: {type(label)}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    test_dataset() 