import os
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import argparse
from pathlib import Path # Use pathlib
# from conv2 import SameDifferentCNN as Conv2CNN # Adjusted import path
# from conv4 import SameDifferentCNN as Conv4CNN # Adjusted import path
# from conv6 import SameDifferentCNN as Conv6CNN # Adjusted import path

# New imports pointing to lr versions
# Ensure 'meta_baseline' is in PYTHONPATH or sys.path from where this script is run
# (e.g., by running from the project root or adding project root to PYTHONPATH)
from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
from baselines.models.conv6 import SameDifferentCNN as Conv6CNN

class PBDataset(Dataset):
    """Dataset for PB tasks with balanced task representation."""
    def __init__(self, data_dir, task_names, split='train', support_sizes=[4, 6, 8, 10], transform=None):
        self.data_dir = data_dir
        # Ensure task_names is always a list
        self.task_names = task_names if isinstance(task_names, list) else [task_names]
        self.split = split
        self.support_sizes = support_sizes
        self.transform = transform
        
        # Dictionary to store data per task
        self.task_data = {task: {'images': [], 'labels': []} for task in self.task_names}
        
        # Process each task separately
        total_loaded_images = 0
        for task_name in self.task_names:
            task_loaded_images = 0
            # Process each support size
            for support_size in support_sizes:
                filename = f"{task_name}_support{support_size}_{split}.h5"
                filepath = os.path.join(data_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}")
                    continue
                
                # print(f"Loading {filename}") # Less verbose logging
                try:
                    with h5py.File(filepath, 'r') as f:
                        # Check if keys exist before accessing
                        if 'support_images' not in f or 'support_labels' not in f or \
                           'query_images' not in f or 'query_labels' not in f:
                            print(f"Warning: Skipping {filepath} due to missing keys.")
                            continue

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
                            task_loaded_images += len(all_images)
                except Exception as e:
                     print(f"Error loading {filepath}: {e}")
                     continue # Skip problematic files
            
            if not self.task_data[task_name]['images']:
                print(f"Warning: No images loaded for task {task_name} in split {split}. Check HDF5 files.")
                # Remove task if no data was loaded to avoid errors later
                del self.task_data[task_name]
                self.task_names.remove(task_name)
                continue

            # Convert to numpy arrays
            self.task_data[task_name]['images'] = np.array(self.task_data[task_name]['images'])
            self.task_data[task_name]['labels'] = np.array(self.task_data[task_name]['labels'])
            
            print(f"Loaded {task_loaded_images} images for task {task_name} ({split})")
            # print(f"Label distribution: {np.bincount(self.task_data[task_name]['labels'].astype(int))}")
            total_loaded_images += task_loaded_images
            
        if not self.task_data:
             raise ValueError(f"No data loaded for any specified tasks in split {split}.")

        # Calculate total size and samples per task per batch
        self.total_size = total_loaded_images
        # Use min only if there are multiple tasks, otherwise use the size of the single task
        if len(self.task_names) > 1:
            self.samples_per_task = min(len(data['images']) for data in self.task_data.values() if data['images'].size > 0)
        elif self.task_data: # Single task case
             self.samples_per_task = len(next(iter(self.task_data.values()))['images'])
        else: # Should not happen due to earlier check, but for safety
             self.samples_per_task = 0

        print(f"Total images across loaded tasks ({split}): {self.total_size}")
        
        # Create indices for balanced sampling (or just indices for single task)
        self.task_indices = {
            task: np.arange(len(data['images'])) 
            for task, data in self.task_data.items()
        }
        
        # Shuffle indices for each task
        for indices in self.task_indices.values():
            np.random.shuffle(indices)
        
        # Keep track of current position in each task
        self.current_pos = {task: 0 for task in self.task_names}
    
    def __len__(self):
        # For single task, length is just the number of images in that task
        if len(self.task_names) == 1:
             return len(self.task_data[self.task_names[0]]['images'])
        # For multiple tasks, use the balanced approach
        elif self.samples_per_task > 0:
            return self.samples_per_task * len(self.task_names)
        else:
            return 0 # No data loaded
    
    def __getitem__(self, idx):
        if len(self.task_names) == 1:
            # Single task: directly use index after shuffling handled internally
            task_name = self.task_names[0]
            pos = self.current_pos[task_name]
            if pos >= len(self.task_indices[task_name]):
                 np.random.shuffle(self.task_indices[task_name])
                 self.current_pos[task_name] = 0
                 pos = 0
            actual_idx = self.task_indices[task_name][pos]
            self.current_pos[task_name] += 1
        else:
            # Balanced multi-task sampling
            task_idx = idx % len(self.task_names)
            task_name = self.task_names[task_idx]
            pos = self.current_pos[task_name]
            if pos >= len(self.task_indices[task_name]):
                np.random.shuffle(self.task_indices[task_name])
                self.current_pos[task_name] = 0
                pos = 0
            actual_idx = self.task_indices[task_name][pos]
            self.current_pos[task_name] += 1

        # Get the image and label
        try:
             image = self.task_data[task_name]['images'][actual_idx]
             label = self.task_data[task_name]['labels'][actual_idx]
        except IndexError:
             # This might happen if data loading failed silently earlier
             print(f"Error: IndexError accessing data for task {task_name} at index {actual_idx}. Dataset size: {len(self.task_data[task_name]['images'])}. Current pos: {pos}.")
             # Handle error appropriately, e.g., return dummy data or raise
             raise

        # Convert uint8 [0, 255] HWC to float32 [-1, 1] CHW
        # Ensure image is uint8 before conversion
        if image.dtype != np.uint8:
             print(f"Warning: Image dtype is {image.dtype}, expected uint8. Attempting conversion.")
             image = image.astype(np.uint8)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 127.5 - 1.0
        label_tensor = torch.tensor(int(label)) # Ensure label is int before tensor conversion

        return {'image': image_tensor, 'label': label_tensor, 'task': task_name}

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Setting deterministic can slow things down, consider if truly needed
        # torch.backends.cudnn.deterministic = True 
        # torch.backends.cudnn.benchmark = False

def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    """Train for one epoch with AMP.""" # Added scaler
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batches_processed = 0

    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for batch in progress_bar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device).float() # BCEWithLogitsLoss expects float labels
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')): # MODIFIED AMP context
             outputs = model(images)
             
             # Assuming model outputs raw logits for binary classification [N, 2] or [N, 1]
             # If model outputs [N, 2], use outputs[:, 1] or handle appropriately
             # If model outputs [N, 1], use outputs.squeeze(1)
             # Let's assume [N, 2] for now, consistent with original script? Needs verification.
             if outputs.shape[1] == 2:
                 output_logits = outputs[:, 1] 
             elif outputs.shape[1] == 1:
                 output_logits = outputs.squeeze(1)
             else:
                  raise ValueError(f"Unexpected output shape from model: {outputs.shape}")

             loss = criterion(output_logits, labels)
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        # Optional: Add gradient clipping here if needed, after scaler.unscale_
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        batches_processed += 1
        # Calculate accuracy from logits
        probs = torch.sigmoid(output_logits) # Get probabilities
        predicted = (probs >= 0.5).float()  # Threshold probabilities
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{running_loss/batches_processed:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader) # Average loss per batch
    avg_acc = correct / total
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        for batch in progress_bar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device).float() # BCEWithLogitsLoss expects float labels
            
            # No need for autocast during validation unless measuring performance
            outputs = model(images)

            # Consistent output handling with training
            if outputs.shape[1] == 2:
                output_logits = outputs[:, 1]
            elif outputs.shape[1] == 1:
                output_logits = outputs.squeeze(1)
            else:
                 raise ValueError(f"Unexpected output shape from model: {outputs.shape}")

            loss = criterion(output_logits, labels)
            
            running_loss += loss.item()
            # Calculate accuracy from logits
            probs = torch.sigmoid(output_logits) # Get probabilities
            predicted = (probs >= 0.5).float()  # Threshold probabilities
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = running_loss / len(val_loader) # Average loss per batch
    avg_acc = correct / total
    return avg_loss, avg_acc

def main():
    # Initialize device at the very beginning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Train a model on a single PB task.')
    # Add arguments (task, architecture, seed, epochs, batch_size, lr, data_dir, output_dir, patience, val_freq)
    parser.add_argument('--task', type=str, required=True, help='Name of the PB task to train on.')
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'], help='Model architecture.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb', help='Directory containing HDF5 task files.')
    parser.add_argument('--output_dir', type=str, default='results/single_task_test', help='Directory to save results and models.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--val_freq', type=int, default=5, help='Frequency (in epochs) to run validation.')

    args = parser.parse_args()

    # print(f"Using device: {device.type}") # This line can stay or be after args if preferred, device is now defined
    # Ensure output directory exists
    # Save under task_name/architecture/seed_X
    specific_output_dir = Path(args.output_dir) / args.task / args.architecture / f"seed_{args.seed}"
    specific_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {specific_output_dir}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load data
    print(f"\nLoading data for task: {args.task}")
    train_dataset = PBDataset(args.data_dir, args.task, split='train')
    val_dataset = PBDataset(args.data_dir, args.task, split='val')
    test_dataset = PBDataset(args.data_dir, args.task, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, criterion, optimizer
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # REMOVE/COMMENT THIS
    
    if args.architecture == 'conv2':
        model = Conv2CNN(num_classes=2).to(device) # PB is binary same/different
    elif args.architecture == 'conv4':
        model = Conv4CNN(num_classes=2).to(device)
    elif args.architecture == 'conv6':
        model = Conv6CNN(num_classes=2).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")
    print(f"Model {args.architecture} initialized.")

    # Save initial model weights
    initial_model_path = specific_output_dir / "initial_model.pth"
    torch.save(model.state_dict(), initial_model_path)
    print(f"Saved initial model weights to {initial_model_path}")
    
    criterion = nn.BCEWithLogitsLoss() # Suitable for binary classification with raw logits
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device_type=device.type, enabled=(device.type == 'cuda')) # MODIFIED AMP scaler

    best_val_acc = 0.0
    epochs_no_improve = 0
    metrics_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\nStarting Training...")
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        
        # Train with AMP
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        metrics_history['train_loss'].append(train_loss)
        metrics_history['train_acc'].append(train_acc)
        print(f"Epoch {epoch} [Train] Avg Loss: {train_loss:.4f}, Avg Acc: {train_acc:.4f}")
        
        # Validate periodically
        if epoch % args.val_freq == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['val_acc'].append(val_acc)
            print(f"Epoch {epoch} [Val]   Avg Loss: {val_loss:.4f}, Avg Acc: {val_acc:.4f}")

            # Checkpointing and Early Stopping
            if val_acc - best_val_acc > 0.005:
                best_val_acc = val_acc
                epochs_no_improve = 0
                save_path = specific_output_dir / "best_model.pth"
                torch.save(model.state_dict(), save_path)
                print(f"New best validation accuracy: {best_val_acc:.4f}. Saved model to {save_path}")
            else:
                epochs_no_improve += 1
                print(f"Patience: {epochs_no_improve}/{args.patience}")
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break
        # End validation block
    # End training loop

    print("\nTraining finished.")
    
    # Load best model for testing
    best_model_path = specific_output_dir / "best_model.pth"
    if best_model_path.exists():
        print(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: No best model found. Using model from last epoch for testing.")

    # Test
    print("\nStarting Testing...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # Save metrics
    metrics = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_history': metrics_history,
        'args': vars(args)
    }
    metrics_path = specific_output_dir / "metrics.json"
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

if __name__ == '__main__':
    main() 