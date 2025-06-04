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
        self.task_names = task_names if isinstance(task_names, list) else [task_names]
        self.split = split
        self.support_sizes = support_sizes
        self.transform = transform
        
        self.task_data = {task: {'images': [], 'labels': []} for task in self.task_names}
        
        total_loaded_images_across_all_tasks = 0
        active_tasks = [] # Keep track of tasks for which data was actually loaded

        for task_name in self.task_names:
            images_for_current_task = []
            labels_for_current_task = []
            task_has_data = False

            file_paths_to_try = []
            for support_size in self.support_sizes:
                # Add the primary file for the current split (e.g., _train.h5, _val.h5, _test.h5)
                primary_filename = f"{task_name}_support{support_size}_{self.split}.h5"
                file_paths_to_try.append(os.path.join(self.data_dir, primary_filename))

                # If we are initializing for the 'train' split, also add corresponding _val.h5 files for augmentation
                # REMOVED: No longer augment train split with val split data to prevent leakage
                # if self.split == 'train':
                #     val_augment_filename = f"{task_name}_support{support_size}_val.h5"
                #     file_paths_to_try.append(os.path.join(self.data_dir, val_augment_filename))
            
            # Remove duplicates that could arise if split='val' (primary and augment would be same)
            unique_file_paths = sorted(list(set(file_paths_to_try)))

            # if self.split == 'train': # Modified this logging
            #    print(f"Task {task_name} (train split): Will attempt to load from {len(unique_file_paths)} HDF5 sources (combining _{self.split}.h5 and potentially _val.h5 variants)...")
            print(f"Task {task_name} (split: {self.split}): Will attempt to load from {len(unique_file_paths)} HDF5 source(s): {unique_file_paths}")

            for filepath in unique_file_paths:
                if not os.path.exists(filepath):
                    if self.split == 'train' and '_val.h5' in os.path.basename(filepath) and f'_{self.split}.h5' not in os.path.basename(filepath):
                        print(f"Info: Optional validation data file for training augmentation not found: {filepath}")
                    # else: A missing primary file (_train.h5 for train, _val.h5 for val, etc.) is a more significant warning/error if it leads to no data for the task.
                    # The check `if not task_has_data:` later will handle if no data is loaded at all.
                    continue
                
                try:
                    with h5py.File(filepath, 'r') as f:
                        if not all(k in f for k in ['support_images', 'support_labels', 'query_images', 'query_labels']):
                            print(f"Warning: Skipping {filepath} due to missing one or more required HDF5 keys.")
                            continue

                        num_episodes = f['support_images'].shape[0]
                        for episode_idx in range(num_episodes):
                            s_imgs = f['support_images'][episode_idx]
                            s_lbls = f['support_labels'][episode_idx]
                            q_imgs = f['query_images'][episode_idx]
                            q_lbls = f['query_labels'][episode_idx]
                            
                            combined_imgs = np.concatenate([s_imgs, q_imgs])
                            combined_lbls = np.concatenate([s_lbls, q_lbls])
                            
                            images_for_current_task.extend(combined_imgs)
                            labels_for_current_task.extend(combined_lbls)
                            task_has_data = True # Mark that we've successfully loaded some data for this task
                except Exception as e:
                     print(f"Error loading data from {filepath}: {e}")
                     continue
            
            if task_has_data:
                self.task_data[task_name]['images'] = np.array(images_for_current_task)
                self.task_data[task_name]['labels'] = np.array(labels_for_current_task)
                current_task_image_count = len(images_for_current_task)
                total_loaded_images_across_all_tasks += current_task_image_count
                active_tasks.append(task_name)
                
                print(f"Loaded {current_task_image_count} images in total for task {task_name} (split: {self.split})")
                unique_labels_vals, counts = np.unique(self.task_data[task_name]['labels'].astype(int), return_counts=True)
                label_dist_str = ", ".join([f"Label {l}: {c}" for l, c in zip(unique_labels_vals, counts)])
                print(f"  Task {task_name} ({self.split}) Final Label Distribution: {label_dist_str}")
                if len(unique_labels_vals) == 0:
                    print(f"  ERROR: Task {task_name} ({self.split}) has NO labels loaded despite task_has_data being true. This is an issue.")
                elif len(unique_labels_vals) == 1:
                    print(f"  WARNING: Task {task_name} ({self.split}) has only one class label: {unique_labels_vals[0]}")
            else:
                print(f"Warning: No images loaded for task {task_name} (split: {self.split}) after attempting all sources. This task will be skipped.")
                del self.task_data[task_name] # Remove task if no data loaded

        self.task_names = active_tasks # Update task_names to only include tasks with data
        if not self.task_names:
             raise ValueError(f"No data loaded for ANY specified tasks in split {self.split} after attempting all sources. Check HDF5 files and paths.")

        self.total_size = total_loaded_images_across_all_tasks
        # Use min only if there are multiple tasks, otherwise use the size of the single task
        if len(self.task_names) > 1:
            self.samples_per_task = min(len(data['images']) for data in self.task_data.values() if data['images'].size > 0)
        elif self.task_data: # Single task case
             self.samples_per_task = len(next(iter(self.task_data.values()))['images'])
        else: # Should not happen due to earlier check, but for safety
             self.samples_per_task = 0

        print(f"Total images across loaded tasks ({self.split}): {self.total_size}")
        
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed for RNG (e.g., PyTorch, NumPy).')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb', help='Directory containing HDF5 task files.')
    parser.add_argument('--output_dir', type=str, default='results/single_task_test', help='Base directory to save results and models.')
    parser.add_argument('--output_seed_idx', type=int, required=True, help='Seed index (0-9) to use for naming the output subfolder (e.g., seed_0, seed_1).')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--val_freq', type=int, default=5, help='Frequency (in epochs) to run validation.')
    parser.add_argument('--improvement_threshold', type=float, default=0.005, help='Minimum improvement in val_acc to reset patience.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for Adam optimizer.')

    args = parser.parse_args()

    print(f"Using device: {device.type}")

    # Ensure output directory exists using output_seed_idx for the specific seed run
    specific_output_dir = Path(args.output_dir) / args.task / args.architecture / f"seed_{args.output_seed_idx}"
    specific_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving results to: {specific_output_dir}")

    # Set RNG seed (this is args.seed, the globally_unique_seed from the Slurm script)
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
    # Original device assignment here should be commented out/removed if it was ever here
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # ENSURE THIS IS GONE OR COMMENTED
    
    if args.architecture == 'conv2':
        model = Conv2CNN().to(device) # REMOVED num_classes
    elif args.architecture == 'conv4':
        model = Conv4CNN().to(device) # REMOVED num_classes
    elif args.architecture == 'conv6':
        model = Conv6CNN().to(device) # REMOVED num_classes
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")
    print(f"Model {args.architecture} initialized.")

    # Save initial model weights
    initial_model_path = specific_output_dir / "initial_model.pth"
    torch.save(model.state_dict(), initial_model_path)
    print(f"Saved initial model weights to {initial_model_path}")
    
    criterion = nn.BCEWithLogitsLoss() 
    # Use args.weight_decay in the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

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

            # Checkpointing and Early Stopping using args.improvement_threshold
            if val_acc - best_val_acc > args.improvement_threshold:
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