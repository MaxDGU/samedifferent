#!/bin/env python
import os
import sys
import torch

# Add project root to path to allow imports from meta_baseline
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch.nn.functional as F
import torch.nn as nn
# Import from the 'meta_baseline.models' package
from meta_baseline.models.conv2lr import SameDifferentCNN as Conv2CNN_lr
from meta_baseline.models.conv4lr import SameDifferentCNN as Conv4CNN_lr
from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6CNN_lr
from meta_baseline.models.utils_meta import SameDifferentDataset, accuracy, collate_episodes

import learn2learn as l2l
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import json
from datetime import datetime
import argparse
import copy
import gc
from pathlib import Path # For Path object operations

# Define all possible tasks
ALL_PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

ARCHITECTURES = {
    'conv2': Conv2CNN_lr,
    'conv4': Conv4CNN_lr,
    'conv6': Conv6CNN_lr
}

# Define support and query sizes for internal sampling
VARIABLE_SUPPORT_SIZES = [4, 6, 8, 10]
FIXED_QUERY_SIZE = 2


def train_epoch(maml, train_loader, optimizer, device, adaptation_steps, scaler, epoch_num, args):
    maml.train()
    total_meta_loss = 0
    total_meta_acc = 0
    num_meta_batches_processed = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch_num+1} Training')
    for meta_batch_idx, current_meta_batch_data in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)
        
        sum_query_losses_for_meta_batch = 0.0
        sum_query_accs_for_meta_batch = 0.0
        num_successful_tasks = 0
        
        actual_meta_batch_size = current_meta_batch_data['support_images'].size(0)
        if actual_meta_batch_size == 0:
            print(f"Warning: Meta-batch {meta_batch_idx} has 0 tasks. Skipping.")
            continue

        for task_idx in range(actual_meta_batch_size):
            learner = maml.clone()
            
            support_images = current_meta_batch_data['support_images'][task_idx].to(device, non_blocking=True)
            support_labels = current_meta_batch_data['support_labels'][task_idx].to(device, non_blocking=True)
            query_images = current_meta_batch_data['query_images'][task_idx].to(device, non_blocking=True)
            query_labels = current_meta_batch_data['query_labels'][task_idx].to(device, non_blocking=True)

            # Adaptation Phase
            task_adaptation_failed = False
            for adapt_step in range(adaptation_steps):
                with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.float()
                    )
                
                if torch.isnan(support_loss) or torch.isinf(support_loss):
                    print(f"CRITICAL: NaN/Inf support_loss detected. Arch: {args.architecture}, E{epoch_num} B{meta_batch_idx} T{task_idx} AS{adapt_step}. Loss: {support_loss.item()}. Skipping adapt for this step and task.")
                    task_adaptation_failed = True
                    break 
                
                learner.adapt(support_loss, allow_unused=True, allow_nograd=True)
            
            if task_adaptation_failed:
                continue

            # Evaluation Phase
            with torch.no_grad():
                query_preds = learner(query_images)
                query_loss_for_task = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.float()
                )
                query_acc_for_task = accuracy(query_preds, query_labels)

            if not torch.isnan(query_loss_for_task) and not torch.isinf(query_loss_for_task):
                sum_query_losses_for_meta_batch += query_loss_for_task
                sum_query_accs_for_meta_batch += query_acc_for_task.item()
                num_successful_tasks += 1
            else:
                print(f"Warning: NaN/Inf query_loss detected for task {task_idx} in meta-batch {meta_batch_idx}. Skipping its contribution.")

        # Average the meta-loss and meta-accuracy over the tasks in the meta-batch
        if num_successful_tasks > 0:
            meta_loss_for_batch = sum_query_losses_for_meta_batch / num_successful_tasks
            meta_acc_for_batch = sum_query_accs_for_meta_batch / num_successful_tasks
        else: 
            print(f"CRITICAL: All {actual_meta_batch_size} tasks failed in meta-batch {meta_batch_idx}. Skipping backward pass for this batch.")
            meta_loss_for_batch = None
            meta_acc_for_batch = 0.0

        # --- Outer Loop Backward Pass ---
        if meta_loss_for_batch is not None:
            if scaler is not None:
                scaler.scale(meta_loss_for_batch).backward()
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(maml.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                meta_loss_for_batch.backward()
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(maml.parameters(), args.grad_clip_norm)
                optimizer.step()
            
            total_meta_loss += meta_loss_for_batch.item()
            total_meta_acc += meta_acc_for_batch
            num_meta_batches_processed += 1
        
        pbar.set_postfix(meta_loss=meta_loss_for_batch.item() if meta_loss_for_batch is not None else 'N/A', meta_acc=meta_acc_for_batch if meta_acc_for_batch is not None else 'N/A')

    if num_meta_batches_processed == 0:
        print("Warning: No batches were successfully processed in this epoch.")
        return 0.0, 0.0
    
    avg_meta_loss = total_meta_loss / num_meta_batches_processed
    avg_meta_acc = total_meta_acc / num_meta_batches_processed
    return avg_meta_loss, avg_meta_acc


def validate_or_test(maml, dataloader, device, adaptation_steps, mode='Validating', epoch_num=None, test_task_name=None):
    maml.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_processed_batches = 0
    
    task_accuracies = {task: [] for task in ALL_PB_TASKS}
    task_losses = {task: [] for task in ALL_PB_TASKS}

    desc = f"Epoch {epoch_num+1} {mode}" if epoch_num is not None else mode
    pbar = tqdm(dataloader, desc=desc)
    
    for batch_idx, current_meta_batch_data in enumerate(pbar):
        support_images = current_meta_batch_data['support_images'].to(device, non_blocking=True)
        support_labels = current_meta_batch_data['support_labels'].to(device, non_blocking=True)
        query_images = current_meta_batch_data['query_images'].to(device, non_blocking=True)
        query_labels = current_meta_batch_data['query_labels'].to(device, non_blocking=True)
        tasks_in_batch = current_meta_batch_data['task']
        
        actual_meta_batch_size = support_images.size(0)

        for task_idx in range(actual_meta_batch_size):
            learner = maml.clone()
            
            # Adaptation
            for step in range(adaptation_steps):
                preds = learner(support_images[task_idx])
                loss = F.binary_cross_entropy_with_logits(preds[:, 1], support_labels[task_idx].float())
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    learner.adapt(loss, allow_unused=True, allow_nograd=True)

            # Evaluation
            with torch.no_grad():
                query_preds = learner(query_images[task_idx])
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels[task_idx].float()
                )
                query_acc = accuracy(query_preds, query_labels[task_idx])

            # Accumulate metrics for the batch
            if not (torch.isnan(query_loss) or torch.isinf(query_loss)):
                total_loss += query_loss.item()
                total_acc += query_acc.item()
                num_processed_batches += 1

                if mode == 'Testing':
                    task_name = tasks_in_batch[task_idx]
                    task_losses[task_name].append(query_loss.item())
                    task_accuracies[task_name].append(query_acc.item())

        pbar.set_postfix(loss=total_loss/num_processed_batches if num_processed_batches > 0 else 0, acc=total_acc/num_processed_batches if num_processed_batches > 0 else 0)

    if num_processed_batches == 0:
        print(f"Warning: No batches were successfully processed during {mode}.")
        return (0.0, 0.0) if mode != 'Testing' else (0.0, 0.0, {})

    avg_loss = total_loss / num_processed_batches
    avg_acc = total_acc / num_processed_batches
    
    if mode == 'Testing':
        final_task_metrics = {}
        for task in ALL_PB_TASKS:
            if task_losses[task]:
                final_task_metrics[task] = {
                    'loss': np.mean(task_losses[task]),
                    'accuracy': np.mean(task_accuracies[task])
                }
        return avg_loss, avg_acc, final_task_metrics
    
    return avg_loss, avg_acc


def set_seed(seed, device_type):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    use_cuda = not args.force_cpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(args.seed, device.type)

    # --- Directories and Files ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use Svar_Q2 to denote the fixed query size from the data
    exp_name = f"exp_all_tasks_fomaml_{args.architecture}_seed{args.seed}_Svar_Q2_{timestamp}"
    output_dir = Path(args.output_base_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Datasets ---
    train_dataset = SameDifferentDataset(args.data_dir, ALL_PB_TASKS, split='train', support_sizes=VARIABLE_SUPPORT_SIZES)
    val_dataset = SameDifferentDataset(args.data_dir, ALL_PB_TASKS, split='val', support_sizes=VARIABLE_SUPPORT_SIZES)

    # --- Dataloaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.meta_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.meta_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train loader size: {len(train_loader)} batches")
    print(f"Validation loader size: {len(val_loader)} batches")
    
    # --- Model, MAML, Optimizer ---
    model_constructor = ARCHITECTURES[args.architecture]
    model = model_constructor().to(device)

    # Add Kaiming Normal initialization (matches meta_baseline script)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0.01)

    # Wrap model with MAML
    maml = l2l.algorithms.MAML(
        model,
        lr=args.inner_lr,
        first_order=args.first_order,
        allow_unused=True, 
        allow_nograd=True
    )
    maml.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(maml.parameters(), lr=args.outer_lr, weight_decay=args.weight_decay)
    
    # Setup Automatic Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and use_cuda else None
    
    # --- Training Loop ---
    best_val_acc = 0
    patience_counter = 0
    metrics_history = []
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(maml, train_loader, optimizer, device, args.adaptation_steps_train, scaler, epoch, args)
        
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_acc = validate_or_test(maml, val_loader, device, args.adaptation_steps_val, mode='Validating', epoch_num=epoch)
            
            print(f"Epoch {epoch+1}/{args.epochs} Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Epoch {epoch+1}/{args.epochs}          Val Loss: {val_loss:.4f},   Val Acc: {val_acc:.4f}")

            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': val_loss, 'val_acc': val_acc
            })

            if val_acc > best_val_acc + args.improvement_threshold:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(maml.state_dict(), output_dir / 'best_model.pth')
                print(f"Epoch {epoch+1}: New best model saved with Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after epoch {epoch+1}. No improvement > {args.improvement_threshold} for {args.patience} validations.")
                    break
        else:
             metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss, 'train_acc': train_acc,
                'val_loss': None, 'val_acc': None
            })

    # --- Final Testing ---
    print("\n--- Final Testing on All Tasks ---")
    best_model_path = output_dir / 'best_model.pth'
    if best_model_path.exists():
        maml.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for final testing.")
    else:
        print("Warning: No best model found. Testing with the final model state.")

    test_dataset = SameDifferentDataset(args.data_dir, ALL_PB_TASKS, 'test', support_sizes=[args.support_size_test])
    test_loader = DataLoader(test_dataset, batch_size=args.meta_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    test_loss, test_acc, individual_task_results = validate_or_test(maml, test_loader, device, args.adaptation_steps_test, mode='Testing')
    
    print("\n--- Overall Test Performance ---")
    print(f"  Overall Test Loss: {test_loss:.4f}")
    print(f"  Overall Test Acc: {test_acc:.4f}")
    
    print("\n--- Per-Task Test Performance ---")
    for task, metrics in individual_task_results.items():
        print(f"  Task: {task:<12} | Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f}")

    # --- Save Final Results ---
    final_summary = {
        'args': vars(args),
        'best_val_acc': best_val_acc,
        'overall_test_loss': test_loss,
        'overall_test_acc': test_acc,
        'individual_task_test_results': individual_task_results,
        'full_metrics_history': metrics_history
    }
    with open(output_dir / 'final_summary_results.json', 'w') as f:
        json.dump(final_summary, f, indent=4)
        
    print(f"\nResults and metrics saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FOMAML training and testing on PB tasks.')
    
    # --- Path Arguments ---
    parser.add_argument('--data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb', help='Directory for the PB HDF5 files.')
    parser.add_argument('--output_base_dir', type=str, default='./results/fomaml_experiments', help='Base directory to save experiment outputs.')

    # --- Model and Run Arguments ---
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'], help='CNN architecture to use.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed.')
    parser.add_argument('--force_cpu', action='store_true', help='Force use of CPU even if CUDA is available.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for DataLoader.')

    # --- MAML specific ---
    parser.add_argument('--first_order', action='store_true', help='Use First-Order MAML (FOMAML). If not set, 2nd order MAML is used.')
    
    # --- Meta-learning parameters ---
    parser.add_argument('--meta_batch_size', type=int, default=8, help='Number of tasks per meta-batch.')
    parser.add_argument('--inner_lr', type=float, default=0.05, help='Learning rate for the inner loop adaptation.')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Learning rate for the outer loop.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the outer loop optimizer.')
    parser.add_argument('--support_size_test', type=int, default=10, help='Support size for the test set (if using a different one).')
    # This argument is logged but does not affect data loading, which is determined by the HDF5 file structure.
    parser.add_argument('--query_size_test', type=int, default=2, help='Query size for the test set. (LOGGING ONLY)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of meta-training epochs.')
    parser.add_argument('--adaptation_steps_train', type=int, default=5, help='Number of adaptation steps during the meta-training inner loop.')
    parser.add_argument('--adaptation_steps_val', type=int, default=10, help='Number of adaptation steps during validation.')
    parser.add_argument('--adaptation_steps_test', type=int, default=10, help='Number of adaptation steps during meta-testing.')
    parser.add_argument('--num_val_meta_batches', type=int, default=25, help='Number of meta-batches for a full validation pass.')
    parser.add_argument('--val_freq', type=int, default=1, help='How many epochs to wait between validations.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping before validation acc improvement.')
    parser.add_argument('--improvement_threshold', type=float, default=0.001, help='Minimum improvement in val_acc to reset patience.')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (AMP) for training if CUDA is available.')
    parser.add_argument('--grad_clip_norm', type=float, default=None, help='Max norm for gradient clipping. Default is None (no clipping).')

    main_args = parser.parse_args()
    
    exit_code = 0 
    try:
        main(main_args)
    except Exception as e:
        print(f"Error during MAML script execution: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    print(f"MAML script finished with exit code: {exit_code}")
    sys.exit(exit_code)