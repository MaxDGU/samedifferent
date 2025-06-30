import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm

from meta_baseline.models.conv6lr import SameDifferentCNN
from baselines.models.utils import train_epoch, validate_epoch, EarlyStopping
from data.vanilla_h5 import VanillaPBDataset

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create save directory
    save_dir = os.path.join(args.save_dir, f"test_{args.test_task}", f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output will be saved to: {save_dir}")

    # Setup Datasets
    all_tasks = ['original', 'filled', 'lines', 'arrows', 'scrambled', 'wider_line', 'open', 'regular', 'random_color', 'irregular']
    train_tasks = [t for t in all_tasks if t != args.test_task]
    print(f"Training on all PB tasks except '{args.test_task}', which is used for validation/testing.")

    train_datasets = [VanillaPBDataset(task=t, split='train', data_dir=args.data_dir) for t in train_tasks]
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = VanillaPBDataset(task=args.test_task, split='val', data_dir=args.data_dir)
    test_dataset = VanillaPBDataset(task=args.test_task, split='test', data_dir=args.data_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, Loss, Optimizer
    model = SameDifferentCNN().to(device)
    print(f"Model: {model.__class__.__name__} | Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopper = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(save_dir, 'best_model.pt'))

    # Training Loop
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Final Testing
    print("\nLoading best model for final testing...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")

    # Save results
    results = {'args': vars(args), 'metrics': metrics, 'test_loss': test_loss, 'test_acc': test_acc}
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Final results saved to {os.path.join(save_dir, 'results.json')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train vanilla model on PB tasks with corrected architecture.')
    parser.add_argument('--data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/vanilla_h5', help='Directory for the dataset')
    parser.add_argument('--save_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_stable/all_tasks/conv6lr', help='Parent directory to save results')
    parser.add_argument('--test_task', type=str, required=True, help='Task to hold out for validation and testing')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--epochs', type=int, default=100, help='Max number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    args = parser.parse_args()
    main(args) 