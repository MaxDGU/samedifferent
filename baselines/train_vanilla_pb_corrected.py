import os
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

# --- Path Setup ---
# Add project root to sys.path to allow for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from baselines.models.utils import SameDifferentDataset, train_epoch, validate, EarlyStopping
from meta_baseline.models.conv6lr import SameDifferentCNN # Use the modern conv6lr architecture

def main(args):
    """Main function to train the vanilla model."""
    # --- Setup ---
    if args.seed is not None:
        print(f"Setting seed to {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Output Directory ---
    # Construct a clear output path
    output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    # --- Data Loading ---
    # Using 'all_tasks' to train a generalist vanilla model on the PB dataset
    print("Loading PB dataset...")
    train_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        task_names='all_tasks', # Special keyword to load all PB tasks
        split='train'
    )
    val_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        task_names='all_tasks',
        split='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("Datasets and DataLoaders created.")

    # --- Model, Optimizer, Loss ---
    print("Initializing model...")
    model = SameDifferentCNN().to(device) # Using the conv6lr architecture

    # Count parameters to confirm architecture
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {type(model).__name__}")
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=10, min_delta=0.005) # Stricter patience
    
    best_val_acc = 0.0
    
    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            print(f"  New best validation accuracy: {val_acc*100:.2f}%. Saving model...")
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pt'))
        
        early_stopping(val_acc)
        if early_stopping.should_stop:
            print(f"Early stopping triggered after epoch {epoch+1}. Best val acc: {best_val_acc*100:.2f}%")
            break
            
    print("\nTraining finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a vanilla Conv6LR model on the PB dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the PB dataset HDF5 files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Parent directory to save model checkpoints.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    
    args = parser.parse_args()
    main(args) 