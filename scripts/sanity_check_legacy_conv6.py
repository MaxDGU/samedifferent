import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import random

# Ensure the project root is in the Python path to allow for correct module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now, import the legacy model
from baselines.models.conv6 import SameDifferentCNN as LegacyConv6

# --- Configuration ---
DATA_DIR = 'data/meta_h5/pb'
# Use a subset of tasks for a quick sanity check
TASKS = ['regular', 'lines', 'open']
SUPPORT_SIZE = 10
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SEED = 42

class SimplePBDataset(Dataset):
    """
    A simple dataset to load images and labels from the PB H5 files,
    ignoring the episodic structure. We'll mix support and query sets
    for a standard supervised training setup.
    """
    def __init__(self, data_dir, tasks, split, support_size):
        self.images = []
        self.labels = []

        for task in tasks:
            file_path = os.path.join(data_dir, f'{task}_support{support_size}_{split}.h5')
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
            
            with h5py.File(file_path, 'r') as f:
                # Combine support and query sets for more data
                support_images = f['support_images'][:]
                support_labels = f['support_labels'][:]
                query_images = f['query_images'][:]
                query_labels = f['query_labels'][:]
                
                # Reshape from (episodes, num_images, H, W, C) to (total_images, H, W, C)
                num_support_eps, num_support_imgs = support_images.shape[0], support_images.shape[1]
                num_query_eps, num_query_imgs = query_images.shape[0], query_images.shape[1]

                all_images = np.vstack([
                    support_images.reshape(-1, 128, 128, 3),
                    query_images.reshape(-1, 128, 128, 3)
                ])
                
                all_labels = np.hstack([
                    support_labels.flatten(),
                    query_labels.flatten()
                ])

                self.images.append(all_images)
                self.labels.append(all_labels)

        self.images = np.vstack(self.images)
        self.labels = np.concatenate(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to torch tensor, normalize, and change to CHW format
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)
        label_tensor = torch.tensor(int(label), dtype=torch.long)
        
        return image_tensor, label_tensor

def main():
    """Main function for the sanity check."""
    print("--- Sanity Check for Legacy Conv6 Model ---")
    
    # Set seed for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Loading training data...")
    train_dataset = SimplePBDataset(DATA_DIR, TASKS, 'train', SUPPORT_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("Loading validation data...")
    val_dataset = SimplePBDataset(DATA_DIR, TASKS, 'val', SUPPORT_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize Model, Loss, and Optimizer
    model = LegacyConv6().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})

        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
            for inputs, labels in pbar_val:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Validation Accuracy after Epoch {epoch+1}: {accuracy:.2f}%")

    print("\n--- Sanity Check Complete ---")
    if accuracy > 55:  # A reasonable threshold for learning
        print("RESULT: SUCCESS! The model is learning.")
        print("This suggests the issue is with the MAML/learn2learn framework, not the model itself.")
    else:
        print("RESULT: FAILURE. The model is NOT learning.")
        print("This suggests a fundamental issue with the model architecture or data loading.")

if __name__ == '__main__':
    main()
