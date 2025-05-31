import os
import torch
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import learn2learn as l2l
import copy
import gc
import random
import torch.nn as nn

from conv2lr import SameDifferentCNN as Conv2CNN
from conv4lr import SameDifferentCNN as Conv4CNN
from conv6lr import SameDifferentCNN as Conv6CNN
from conv6lr import SameDifferentDataset, collate_episodes, EarlyStopping

# Define all PB tasks
PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(predictions, targets):
    """Calculate binary classification accuracy."""
    predicted_labels = (predictions[:, 1] > 0.0).float()
    return (predicted_labels == targets).float().mean()

def train_epoch(maml, train_loader, optimizer, device, adaptation_steps, scaler):
    """Single training epoch with improved monitoring"""
    maml.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    # Check model initialization
    with torch.no_grad():
        init_check = next(iter(train_loader))
        init_episode = init_check[0]
        init_preds = maml(init_episode['support_images'].to(device))
        if torch.allclose(init_preds, torch.zeros_like(init_preds)):
            print("\nWARNING: Model is initializing to all zeros. Reinitializing weights...")
            for m in maml.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.01)
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, episodes in enumerate(pbar):
        optimizer.zero_grad()
        batch_loss = 0
        batch_acc = 0
        
        # Process each episode with mixed precision
        with torch.amp.autocast(device_type='cuda'):
            for episode in episodes:
                learner = maml.clone()
                
                # Move data to GPU
                support_images = episode['support_images'].to(device, non_blocking=True)
                support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
                query_images = episode['query_images'].to(device, non_blocking=True)
                query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)
                
                # Inner loop adaptation with loss checking
                for step in range(adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.squeeze(1).float())
                    
                    # Check for abnormal loss values
                    if support_loss.item() > 10 and step == 0:
                        print(f"\nWARNING: High support loss: {support_loss.item():.4f}")
                        print("Support predictions:", torch.sigmoid(support_preds[:, 1]).detach().cpu().numpy())
                        print("Support labels:", support_labels.squeeze(1).cpu().numpy())
                    
                    # Modified gradient computation to handle unused parameters
                    grads = torch.autograd.grad(
                        support_loss,
                        [p for p in learner.parameters() if p.requires_grad],
                        create_graph=True,
                        allow_unused=True
                    )
                    
                    # Filter out None gradients and corresponding parameters
                    valid_grads_and_params = [(g, p) for g, p in zip(grads, [p for p in learner.parameters() if p.requires_grad]) if g is not None]
                    
                    if valid_grads_and_params:
                        # Gradient norm clipping for stability
                        grads, params = zip(*valid_grads_and_params)
                        grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
                        if grad_norm > 10:
                            scaling_factor = 10 / grad_norm
                            grads = [g * scaling_factor for g in grads]
                        
                        # Manual parameter update with inner learning rate
                        for param, grad in zip(params, grads):
                            param.data = param.data - learner.lr * grad
                
                # Query loss and accuracy
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                
                # Check for abnormal query loss
                if query_loss.item() > 10:
                    print(f"\nWARNING: High query loss: {query_loss.item():.4f}")
                    print("Query predictions:", torch.sigmoid(query_preds[:, 1]).detach().cpu().numpy())
                    print("Query labels:", query_labels.squeeze(1).cpu().numpy())
                
                query_acc = accuracy(query_preds, query_labels)
                
                batch_loss += query_loss
                batch_acc += query_acc.item()
        
        # Scale loss and backward pass
        scaled_loss = batch_loss / len(episodes)
        
        # Additional loss value check
        if scaled_loss.item() > 10:
            print(f"\nWARNING: High batch loss: {scaled_loss.item():.4f}")
        
        scaler.scale(scaled_loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in maml.parameters() if p.requires_grad], max_norm=10.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += scaled_loss.item()
        total_acc += batch_acc / len(episodes)
        n_batches += 1
        
        # Update progress bar with more information
        pbar.set_postfix({
            'loss': f'{total_loss/n_batches:.4f}',
            'acc': f'{total_acc/n_batches:.4f}',
            'batch_loss': f'{scaled_loss.item():.4f}'
        })
        
        # Early warning for unstable training
        if batch_idx == 0 and total_loss/n_batches > 10:
            print("\nWARNING: Initial loss is very high. Consider:")
            print("1. Reducing learning rates (current inner_lr={}, outer_lr={})".format(
                learner.lr, optimizer.param_groups[0]['lr']))
            print("2. Checking data normalization")
            print("3. Adjusting model initialization")
    
    return total_loss / n_batches, total_acc / n_batches

def validate(model, val_loader, device, adaptation_steps, inner_lr):
    """Validate using MAML."""
    model.eval()
    total_val_loss = 0
    total_val_acc = 0
    num_batches = 0

    for batch_idx, episodes in enumerate(val_loader):
        batch_loss = 0
        batch_acc = 0
        
        for episode in episodes:
            # Move data to GPU and handle dimensions
            support_images = episode['support_images'].to(device, non_blocking=True)
            support_labels = episode['support_labels'].unsqueeze(1).to(device, non_blocking=True)
            query_images = episode['query_images'].to(device, non_blocking=True)
            query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)

            # Clone the model for adaptation
            adapted_model = copy.deepcopy(model)
            adapted_model.train()  # Put in training mode for adaptation
            
            # Ensure parameters require gradients for adaptation
            for param in adapted_model.parameters():
                param.requires_grad_(True)

            # Support set adaptation
            for step in range(adaptation_steps):
                support_preds = adapted_model(support_images)
                support_loss = F.binary_cross_entropy_with_logits(
                    support_preds[:, 1],
                    support_labels.squeeze(1).float()
                )
                
                # Check for high loss in first step
                if step == 0 and support_loss.item() > 10:
                    print(f"\nWARNING: High validation support loss: {support_loss.item():.4f}")
                    print("Support predictions:", torch.sigmoid(support_preds[:, 1]).detach().cpu().numpy())
                    print("Support labels:", support_labels.squeeze(1).cpu().numpy())
                
                # Modified gradient computation to handle unused parameters
                grads = torch.autograd.grad(
                    support_loss,
                    [p for p in adapted_model.parameters() if p.requires_grad],
                    create_graph=False,  # No need for create_graph in validation
                    allow_unused=True
                )
                
                # Filter out None gradients and corresponding parameters
                valid_grads_and_params = [(g, p) for g, p in zip(grads, [p for p in adapted_model.parameters() if p.requires_grad]) if g is not None]
                
                if valid_grads_and_params:
                    # Gradient norm clipping for stability
                    grads, params = zip(*valid_grads_and_params)
                    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
                    if grad_norm > 10:
                        scaling_factor = 10 / grad_norm
                        grads = [g * scaling_factor for g in grads]
                    
                    # Manual parameter update with inner learning rate
                    for param, grad in zip(params, grads):
                        param.data = param.data - inner_lr * grad

            # Evaluate on query set
            adapted_model.eval()
            with torch.no_grad():
                query_preds = adapted_model(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1],
                    query_labels.squeeze(1).float()
                )
                
                # Check for high query loss
                if query_loss.item() > 10:
                    print(f"\nWARNING: High validation query loss: {query_loss.item():.4f}")
                    print("Query predictions:", torch.sigmoid(query_preds[:, 1]).detach().cpu().numpy())
                    print("Query labels:", query_labels.squeeze(1).cpu().numpy())
                
                query_acc = accuracy(query_preds, query_labels)
                batch_loss += query_loss.item()
                batch_acc += query_acc.item()
        
        # Average over episodes in the batch
        avg_batch_loss = batch_loss / len(episodes)
        avg_batch_acc = batch_acc / len(episodes)
        
        total_val_loss += avg_batch_loss
        total_val_acc += avg_batch_acc
        num_batches += 1

    avg_val_loss = total_val_loss / num_batches
    avg_val_acc = total_val_acc / num_batches

    return avg_val_loss, avg_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_task', type=str, required=True, help='PB task to test on')
    parser.add_argument('--architecture', type=str, default='conv2', choices=['conv2', 'conv4', 'conv6'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/test_baseline')
    parser.add_argument('--support_size', type=int, default=10, help='Number of support examples per class')
    parser.add_argument('--adaptation_steps', type=int, default=5, help='Number of adaptation steps during training')
    parser.add_argument('--test_adaptation_steps', type=int, default=15, help='Number of adaptation steps during testing')
    parser.add_argument('--inner_lr', type=float, default=0.05, help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Outer loop learning rate')
    args = parser.parse_args()
    
    # Set random seeds
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets - train and validate on all tasks
    print("\nCreating datasets...")
    print(f"Training on all tasks, will test on new examples from: {args.test_task}")
    
    train_dataset = SameDifferentDataset('data/pb/pb', PB_TASKS, 'train', support_sizes=[args.support_size])
    val_dataset = SameDifferentDataset('data/pb/pb', PB_TASKS, 'val', support_sizes=[args.support_size])
    # Test only on new examples from test split of specified task
    test_dataset = SameDifferentDataset('data/pb/pb', [args.test_task], 'test', support_sizes=[args.support_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_episodes)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_episodes)
    
    # Create model
    print(f"\nCreating {args.architecture} model")
    if args.architecture == 'conv2':
        model = Conv2CNN()
        # Special initialization for conv2
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Slightly positive bias
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)  # Scale to 1
                nn.init.constant_(m.bias, 0.1)    # Slightly positive bias
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)    # Slightly positive bias
    elif args.architecture == 'conv4':
        model = Conv4CNN()
        # Special initialization for conv4
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Slightly positive bias
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)  # Scale to 1
                nn.init.constant_(m.bias, 0.1)    # Slightly positive bias
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)    # Slightly positive bias
    else:
        model = Conv6CNN()
    
    model = model.to(device)
    print(f"Model created on {device}")
    
    # Create MAML model with first-order approximation
    maml = l2l.algorithms.MAML(model, lr=args.inner_lr, first_order=False, 
                              allow_unused=True, allow_nograd=True)
    
    # Adjust optimizer based on architecture
    if args.architecture in ['conv2', 'conv4']:
        # Use a smaller learning rate for shallower networks
        optimizer = torch.optim.Adam(maml.parameters(), lr=args.outer_lr * 0.5)
    else:
        optimizer = torch.optim.Adam(maml.parameters(), lr=args.outer_lr)
    
    # Initialize early stopping and AMP
    early_stopping = EarlyStopping(patience=10)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': None,
        'test_acc': None,
        'train_tasks': PB_TASKS,
        'test_task': args.test_task,
        'train_adaptation_steps': args.adaptation_steps,
        'test_adaptation_steps': args.test_adaptation_steps,
        'inner_lr': args.inner_lr,
        'outer_lr': args.outer_lr
    }
    
    print("\nStarting training...")
    best_val_acc = 0
    for epoch in range(100):  # Max 100 epochs
        print(f"\nEpoch {epoch+1}/100")
        
        # Train and validate
        train_loss, train_acc = train_epoch(
            maml, train_loader, optimizer, device,
            args.adaptation_steps, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, device,
            args.adaptation_steps, args.inner_lr
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'maml_state_dict': maml.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            early_stopping.counter = 0
        else:
            early_stopping.counter += 1
            if early_stopping.counter >= early_stopping.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    maml.load_state_dict(checkpoint['maml_state_dict'])
    
    # Test
    test_loss, test_acc = validate(
        model, test_loader, device,
        args.test_adaptation_steps, args.inner_lr
    )
    
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    
    # Save final results
    results = {
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_acc,
            'adaptation_steps': args.test_adaptation_steps
        },
        'best_val_metrics': {
            'loss': checkpoint['val_loss'],
            'accuracy': checkpoint['val_acc'],
            'epoch': checkpoint['epoch']
        },
        'args': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 