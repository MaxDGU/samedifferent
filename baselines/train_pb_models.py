import os
import json
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
import random
import argparse
from PIL import Image
import sys

# Add meta_baseline to path to import the new architectures that match meta models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from meta_baseline.models.conv2lr import SameDifferentCNN as Conv2CNN
from meta_baseline.models.conv4lr import SameDifferentCNN as Conv4CNN
from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6CNN
from meta_baseline.models.utils_meta import SameDifferentDataset, train_epoch, validate_epoch, EarlyStopping

def collate_vanilla(batch):
    """Collate function for vanilla training.
    
    Converts a list of episode dicts into a single batch of (images, labels).
    """
    images = []
    labels = []
    for episode in batch:
        # Combine support and query sets for vanilla training
        images.append(episode['support_images'])
        images.append(episode['query_images'])
        labels.append(episode['support_labels'])
        labels.append(episode['query_labels'])
    
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return images, labels

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_task', type=str, required=True, help='PB task to test on')
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=1)
    parser.add_argument('--improvement_threshold', type=float, default=0.01)
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb')
    parser.add_argument('--output_dir', type=str, default='results/pb_baselines')
    parser.add_argument('--dropout_rate_fc', type=float, default=0.5, help='Dropout rate for fully connected layers')
    args = parser.parse_args()
    
    # Define all PB tasks
    PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
                'random_color', 'arrows', 'irregular', 'filled', 'original']
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory - new structure reflecting training on all tasks
    model_dir = os.path.join(args.output_dir, 'all_tasks', args.architecture, f'test_{args.test_task}', f'seed_{args.seed}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Create datasets and dataloaders
    # Training on all tasks
    train_dataset = SameDifferentDataset(args.data_dir, PB_TASKS, split='train')
    # Validation and testing on specified task only
    val_dataset = SameDifferentDataset(args.data_dir, [args.test_task], split='val')
    test_dataset = SameDifferentDataset(args.data_dir, [args.test_task], split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_vanilla)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_vanilla)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_vanilla)
    
    # Initialize model
    if args.architecture == 'conv2':
        model = Conv2CNN(dropout_rate_fc=args.dropout_rate_fc)
    elif args.architecture == 'conv4':
        model = Conv4CNN(dropout_rate_fc=args.dropout_rate_fc)
    else:  # conv6
        model = Conv6CNN(dropout_rate_fc=args.dropout_rate_fc)
    
    model = model.to(device)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_acc = 0
    training_history = []
    early_stopping = EarlyStopping(patience=10, verbose=True, path=os.path.join(model_dir, 'checkpoint.pt'), delta=-0.01)
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Only validate every val_freq epochs
        if (epoch + 1) % args.val_freq == 0:
            # Validate
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            # Save history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            # Check if validation accuracy improved
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f'New best validation accuracy: {val_acc:.2f}%')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, os.path.join(model_dir, 'best_model.pt'))
            
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Early stopping check
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
        else:
            # Just save training metrics
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': None,
                'val_acc': None
            })
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    
    # Load best model and evaluate on test set
    if os.path.exists(os.path.join(model_dir, 'best_model.pt')):
        checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    else:
        print("No best model found, using final model state")
    
    print("\n--- Starting Final Evaluation on All Tasks ---")
    test_results = {}
    for task in PB_TASKS:
        print(f"\nTesting on task: {task}")
        # Each task needs its own dataset and dataloader for testing
        test_dataset = SameDifferentDataset(args.data_dir, [task], split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_vanilla)
        
        # We need to get the train_epoch and validate_epoch functions from the utils_meta file
        # as the previous utils.py file was deleted. We will use validate_epoch for testing.
        test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
        test_results[task] = {'test_loss': test_loss, 'test_acc': test_acc}
        print(f'Test Loss for {task}: {test_loss:.4f} | Test Acc for {task}: {test_acc:.2f}%')
    
    # Save metrics
    metrics = {
        'args': vars(args),
        'training_history': training_history,
        'best_val_acc': best_val_acc,
        'test_results': test_results,
        'total_epochs': epoch + 1
    }
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main() 