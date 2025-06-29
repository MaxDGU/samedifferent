import os
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import json
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from baselines.models.utils import SameDifferentDataset, train_epoch, validate, EarlyStopping, PB_TASKS
from meta_baseline.models.conv6lr import SameDifferentCNN

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    """Main function to train the vanilla model."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Output Directory ---
    model_dir = os.path.join(args.output_dir, 'all_tasks', 'conv6lr', f'test_{args.test_task}', f'seed_{args.seed}')
    os.makedirs(model_dir, exist_ok=True)
    print(f"Output will be saved to: {model_dir}")

    # --- Data Loading ---
    print(f"Training on all PB tasks, validating/testing on '{args.test_task}'")
    train_dataset = SameDifferentDataset(data_dir=args.data_dir, task_names=PB_TASKS, split='train')
    val_dataset = SameDifferentDataset(data_dir=args.data_dir, task_names=[args.test_task], split='val')
    test_dataset = SameDifferentDataset(data_dir=args.data_dir, task_names=[args.test_task], split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model, Optimizer, Loss ---
    model = SameDifferentCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__} | Trainable Params: {total_params / 1e6:.2f}M")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.improvement_threshold)
    
    best_val_acc = 0.0
    training_history = []
    
    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_acc = None, None
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            if val_acc > best_val_acc:
                print(f"  Epoch {epoch+1}: New best val acc: {val_acc*100:.2f}%. Saving model...")
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(model_dir, 'best_model.pt'))

            early_stopping(val_acc)
            if early_stopping.should_stop:
                print(f"Early stopping triggered after epoch {epoch+1}.")
                break
        
        training_history.append({
            'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        })
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

    # --- Final Evaluation ---
    print("\nTraining finished. Evaluating on test set...")
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', -1)+1}")
    else:
        print("No best model found. Using final model for testing.")

    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    # --- Save Metrics ---
    metrics = {
        'args': vars(args), 'training_history': training_history,
        'best_val_acc': best_val_acc, 'test_loss': test_loss, 'test_acc': test_acc
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a vanilla Conv6LR model on the PB dataset.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the PB dataset HDF5 files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Parent directory to save model checkpoints.')
    parser.add_argument('--test_task', type=str, required=True, help='Which PB task to use for validation and testing.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping.')
    parser.add_argument('--val_freq', type=int, default=10, help='Frequency (in epochs) to run validation.')
    parser.add_argument('--improvement_threshold', type=float, default=0.02, help='Min improvement for early stopping.')
    
    args = parser.parse_args()
    main(args) 