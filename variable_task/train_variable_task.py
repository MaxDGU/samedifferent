#!/bin/env python
import os
import torch
import torch.nn.functional as F
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
from torch.utils.data import Sampler # For custom sampler


ARCHITECTURES = {
    'conv2': Conv2CNN_lr,
    'conv4': Conv4CNN_lr,
    'conv6': Conv6CNN_lr
}

# Define support and query sizes for internal sampling
VARIABLE_SUPPORT_SIZES = [4, 6, 8, 10]
FIXED_QUERY_SIZE = 2

# Helper function for printing model/learner stats
ARGS_REF_FOR_PRINTING = None # Global to be set by main for easy access in helper

def print_debug_stats(tag, arch, epoch, meta_batch_idx, task_idx=None, adapt_step=None, learner=None, maml_model=None, support_preds=None, support_loss=None, query_preds=None, query_loss=None):
    if ARGS_REF_FOR_PRINTING is None or ARGS_REF_FOR_PRINTING.architecture not in ['conv4', 'conv6']:
        return
    if epoch > 0 or meta_batch_idx > 1 : # Limit prints to first epoch, first 2 meta-batches
        return

    prefix = f"DEBUG [{arch} E{epoch} B{meta_batch_idx}"
    if task_idx is not None:
        prefix += f" T{task_idx}"
    if adapt_step is not None:
        prefix += f" AS{adapt_step}"
    prefix += f"] {tag}:"

    print(prefix)
    if support_preds is not None:
        print(f"  Support Preds: min={support_preds.min().item():.3e}, max={support_preds.max().item():.3e}, mean_abs={support_preds.abs().mean().item():.3e}, has_nan={torch.isnan(support_preds).any().item()}")
    if support_loss is not None:
        print(f"  Support Loss: {support_loss.item() if isinstance(support_loss, torch.Tensor) else support_loss:.4e}, is_nan={torch.isnan(support_loss).any().item() if isinstance(support_loss, torch.Tensor) else 'N/A'}")
    if query_preds is not None:
        print(f"  Query Preds: min={query_preds.min().item():.3e}, max={query_preds.max().item():.3e}, mean_abs={query_preds.abs().mean().item():.3e}, has_nan={torch.isnan(query_preds).any().item()}")
    if query_loss is not None:
        print(f"  Query Loss: {query_loss.item() if isinstance(query_loss, torch.Tensor) else query_loss:.4e}, is_nan={torch.isnan(query_loss).any().item() if isinstance(query_loss, torch.Tensor) else 'N/A'}")

    model_to_inspect = learner if learner is not None else maml_model
    if model_to_inspect:
        try:
            if hasattr(model_to_inspect.module, 'conv1') and model_to_inspect.module.conv1.weight is not None:
                c1w_abs_mean = model_to_inspect.module.conv1.weight.data.abs().mean().item()
                c1g_abs_mean = 'NoGrad'
                if model_to_inspect.module.conv1.weight.grad is not None:
                    c1g_abs_mean = model_to_inspect.module.conv1.weight.grad.abs().mean().item()
                print(f"  Conv1: w_abs_mean={c1w_abs_mean:.3e}, g_abs_mean={c1g_abs_mean if isinstance(c1g_abs_mean, str) else f'{c1g_abs_mean:.3e}'}")

            if hasattr(model_to_inspect.module, 'classifier') and model_to_inspect.module.classifier.weight is not None:
                clw_abs_mean = model_to_inspect.module.classifier.weight.data.abs().mean().item()
                clg_abs_mean = 'NoGrad'
                if model_to_inspect.module.classifier.weight.grad is not None:
                    clg_abs_mean = model_to_inspect.module.classifier.weight.grad.abs().mean().item()
                print(f"  Classifier: w_abs_mean={clw_abs_mean:.3e}, g_abs_mean={clg_abs_mean if isinstance(clg_abs_mean, str) else f'{clg_abs_mean:.3e}'}")
            
            total_weight_sum = sum(p.data.abs().sum().item() for p in model_to_inspect.parameters() if p.requires_grad and p.data is not None)
            print(f"  Total W Abs Sum: {total_weight_sum:.3e}")
            if maml_model is None: # Grads are on the main maml model after outer backward, not on learner
                 total_grad_sum = sum(p.grad.abs().sum().item() for p in model_to_inspect.parameters() if p.grad is not None and p.requires_grad)
                 print(f"  Total G Abs Sum (Learner): {total_grad_sum:.3e}")


        except AttributeError as e:
            print(f"  Error accessing model attributes for stats: {e}")
    if maml_model and tag in ["PostOuterBackward", "PostOptimizerStep"]: # Grads are on main MAML model
        total_grad_sum_main = sum(p.grad.abs().sum().item() for p in maml_model.parameters() if p.grad is not None and p.requires_grad)
        print(f"  Total G Abs Sum (MAML model): {total_grad_sum_main:.3e}")


# --- Custom Sampler for Variable Support Sizes ---
class VariableSupportSampler(Sampler):
    def __init__(self, dataset, num_batches, meta_batch_size, available_support_sizes):
        self.dataset = dataset # SameDifferentDataset instance
        self.num_batches = num_batches # e.g., args.num_meta_batches_per_epoch
        self.meta_batch_size = meta_batch_size
        self.available_support_sizes = available_support_sizes

        # Pre-calculate indices for each support size for efficiency
        self.indices_by_support_size = {s: [] for s in self.available_support_sizes}
        global_idx_offset = 0
        if hasattr(self.dataset, 'episode_files') and hasattr(self.dataset, 'file_episode_counts'):
            for i, file_info in enumerate(self.dataset.episode_files):
                s_size = file_info['support_size']
                num_episodes_in_file = self.dataset.file_episode_counts[i]
                if s_size in self.indices_by_support_size:
                    self.indices_by_support_size[s_size].extend(
                        range(global_idx_offset, global_idx_offset + num_episodes_in_file)
                    )
                global_idx_offset += num_episodes_in_file
        else:
            # Fallback or error if dataset structure is not as expected
            # This might happen if SameDifferentDataset changes its internal structure
            print("Warning: VariableSupportSampler could not pre-cache indices due to unexpected dataset structure.")
            # The __iter__ method will then have to compute them on the fly (as it did before)

        # Check if any indices were loaded, otherwise, __iter__ will have issues
        if not any(self.indices_by_support_size.values()):
            # This check is important if the pre-caching above failed or found nothing.
            # If the dataset is genuinely empty for these support sizes, this will be caught here.
            # However, SameDifferentDataset itself raises an error if no HDF5 files are found at all.
            # This secondary check ensures the sampler is aware if its specific support_sizes have no episodes.
            found_any_episodes = False
            if hasattr(self.dataset, 'episode_files'): # Check again if iteration is needed
                for s_size_check in self.available_support_sizes:
                    if any(f_info['support_size'] == s_size_check for f_info in self.dataset.episode_files):
                        found_any_episodes = True
                        break
            if not found_any_episodes:
                 raise ValueError("VariableSupportSampler: No episodes found for ANY of the specified available_support_sizes. Check HDF5 files and dataset initialization.")


    def __iter__(self):
        for _ in range(self.num_batches):
            current_support_size = random.choice(self.available_support_sizes)
            
            candidate_indices = self.indices_by_support_size.get(current_support_size)

            # If pre-caching failed or that specific support size has no episodes from pre-caching
            if candidate_indices is None or not candidate_indices:
                # Fallback to on-the-fly computation if pre-caching didn't populate this key
                # or if it was empty. This makes it robust if __init__ pre-caching had issues.
                # print(f"Debug: Sampler falling back to on-the-fly index generation for S={current_support_size}")
                candidate_indices_fallback = []
                global_idx_offset = 0
                if hasattr(self.dataset, 'episode_files') and hasattr(self.dataset, 'file_episode_counts'):
                    for i, file_info in enumerate(self.dataset.episode_files):
                        num_episodes_in_file = self.dataset.file_episode_counts[i]
                        if file_info['support_size'] == current_support_size:
                            candidate_indices_fallback.extend(range(global_idx_offset, global_idx_offset + num_episodes_in_file))
                        global_idx_offset += num_episodes_in_file
                    candidate_indices = candidate_indices_fallback # Use the new list
                else:
                    # Should not happen if dataset is standard SameDifferentDataset
                    raise RuntimeError("VariableSupportSampler: Dataset does not have expected attributes for on-the-fly index generation.")
            
            if not candidate_indices:
                # This means even the fallback failed to find episodes for the chosen support_size.
                # Try to find a support size that *does* have episodes.
                # print(f"Warning: No episodes found for support size {current_support_size}. Resampling S.")
                attempts = 0
                original_choice = current_support_size
                while not candidate_indices and attempts < len(self.available_support_sizes) * 2:
                    current_support_size = random.choice(self.available_support_sizes)
                    candidate_indices = self.indices_by_support_size.get(current_support_size)
                    if candidate_indices is None or not candidate_indices: # Also check on-the-fly for the new choice
                        candidate_indices_fallback_retry = []
                        global_idx_offset_retry = 0
                        if hasattr(self.dataset, 'episode_files') and hasattr(self.dataset, 'file_episode_counts'):
                            for i_retry, file_info_retry in enumerate(self.dataset.episode_files):
                                num_episodes_in_file_retry = self.dataset.file_episode_counts[i_retry]
                                if file_info_retry['support_size'] == current_support_size:
                                    candidate_indices_fallback_retry.extend(range(global_idx_offset_retry, global_idx_offset_retry + num_episodes_in_file_retry))
                                global_idx_offset_retry += num_episodes_in_file_retry
                            candidate_indices = candidate_indices_fallback_retry
                    attempts += 1
                
                if not candidate_indices:
                    raise ValueError(f"VariableSupportSampler: Could not find episodes for any available support size after multiple attempts. Original S={original_choice}. Check HDF5 files.")

            batch_indices = random.sample(candidate_indices, self.meta_batch_size)
            yield batch_indices

    def __len__(self):
        # This is the number of meta-batches the DataLoader will produce per epoch
        # When using batch_sampler, __len__ should return the number of batches.
        return self.num_batches # <--- Correct: Number of meta-batches


def train_epoch(maml, train_loader, optimizer, device, adaptation_steps, scaler, epoch_num, args):
    maml.train()
    total_meta_loss = 0
    total_meta_acc = 0
    num_meta_batches_processed = 0
    
    current_arch = args.architecture
    if epoch_num == 0 and current_arch in ['conv4', 'conv6'] :
        print_debug_stats("TrainEpochStart", current_arch, epoch_num, 0, maml_model=maml)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch_num+1} Training')
    for meta_batch_idx, current_meta_batch_data in enumerate(pbar):
        optimizer.zero_grad(set_to_none=True)
        
        sum_query_losses_for_meta_batch = 0.0
        sum_query_accs_for_meta_batch = 0.0
        
        actual_meta_batch_size = current_meta_batch_data['support_images'].size(0)
        if actual_meta_batch_size == 0:
            print(f"Warning: Meta-batch {meta_batch_idx} has 0 tasks. Skipping.")
            continue

        for task_idx in range(actual_meta_batch_size):
            learner = maml.clone()
            print_debug_stats("LearnerCloned", current_arch, epoch_num, meta_batch_idx, task_idx=task_idx, learner=learner)
            
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
                print_debug_stats("AdaptStepPreLoss", current_arch, epoch_num, meta_batch_idx, task_idx=task_idx, adapt_step=adapt_step, support_preds=support_preds, support_loss=support_loss, learner=learner)
                
                if torch.isnan(support_loss) or torch.isinf(support_loss):
                    print(f"CRITICAL: NaN/Inf support_loss detected. Arch: {current_arch}, E{epoch_num} B{meta_batch_idx} T{task_idx} AS{adapt_step}. Loss: {support_loss.item()}. Skipping adapt for this step and task.")
                    task_adaptation_failed = True
                    break 
                
                learner.adapt(support_loss, allow_unused=True, allow_nograd=True)
                print_debug_stats("AdaptStepPostAdapt", current_arch, epoch_num, meta_batch_idx, task_idx=task_idx, adapt_step=adapt_step, learner=learner)
            
            if task_adaptation_failed:
                continue

            # Evaluation phase on query set
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.float()
                )
            
            print_debug_stats("EvalStep", current_arch, epoch_num, meta_batch_idx, task_idx=task_idx, query_preds=query_preds, query_loss=query_loss, learner=learner)

            if torch.isnan(query_loss) or torch.isinf(query_loss):
                 print(f"CRITICAL: NaN/Inf query_loss detected. Arch: {current_arch}, E{epoch_num} B{meta_batch_idx} T{task_idx}. Loss: {query_loss.item()}. Skipping task's contribution to meta-loss.")
                 continue

            sum_query_losses_for_meta_batch += query_loss
            with torch.no_grad():
                sum_query_accs_for_meta_batch += accuracy(query_preds, query_labels)

        # Meta-update after processing all tasks in the meta-batch
        avg_query_loss_for_meta_batch = sum_query_losses_for_meta_batch / actual_meta_batch_size
        avg_query_acc_for_meta_batch = sum_query_accs_for_meta_batch / actual_meta_batch_size
        
        print_debug_stats("PreOuterBackward", current_arch, epoch_num, meta_batch_idx, query_loss=avg_query_loss_for_meta_batch, maml_model=maml)

        if scaler is not None:
            scaler.scale(avg_query_loss_for_meta_batch).backward()
        else:
            avg_query_loss_for_meta_batch.backward()
        
        print_debug_stats("PostOuterBackward", current_arch, epoch_num, meta_batch_idx, maml_model=maml)

        # Gradient clipping
        if args.grad_clip is not None:
            if scaler is not None:
                scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(maml.parameters(), args.grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        print_debug_stats("PostOptimizerStep", current_arch, epoch_num, meta_batch_idx, maml_model=maml)

        total_meta_loss += avg_query_loss_for_meta_batch.item()
        total_meta_acc += avg_query_acc_for_meta_batch
        num_meta_batches_processed += 1

        pbar.set_postfix(meta_loss=total_meta_loss / num_meta_batches_processed, 
                            meta_acc=total_meta_acc / num_meta_batches_processed)

    if num_meta_batches_processed == 0:
        print("Warning: No meta-batches were successfully processed in this epoch.")
        return 0.0, 0.0

    avg_epoch_loss = total_meta_loss / num_meta_batches_processed
    avg_epoch_acc = total_meta_acc / num_meta_batches_processed
    return avg_epoch_loss, avg_epoch_acc


def validate_or_test(maml, dataloader, device, adaptation_steps, mode='Validating', epoch_num=None, test_task_name=None):
    maml.eval()
    total_loss = 0
    total_acc = 0
    num_tasks = 0
    
    desc = mode
    if epoch_num is not None:
        desc = f'Epoch {epoch_num+1} {mode}'
    if test_task_name:
        desc += f' on {test_task_name}'

    pbar = tqdm(dataloader, desc=desc)
    for task_data in pbar:
        learner = maml.clone()
        
        support_images = task_data['support_images'].squeeze(0).to(device)
        support_labels = task_data['support_labels'].squeeze(0).to(device)
        query_images = task_data['query_images'].squeeze(0).to(device)
        query_labels = task_data['query_labels'].squeeze(0).to(device)
        
        # Adaptation
        for _ in range(adaptation_steps):
            support_preds = learner(support_images)
            support_loss = F.binary_cross_entropy_with_logits(support_preds[:, 1], support_labels.float())
            if torch.isnan(support_loss) or torch.isinf(support_loss):
                print(f"Warning: NaN/Inf support_loss in {mode} for task. Skipping adaptation step.")
                break
            learner.adapt(support_loss, allow_unused=True, allow_nograd=True)

        # Evaluation
        with torch.no_grad():
            query_preds = learner(query_images)
            query_loss = F.binary_cross_entropy_with_logits(query_preds[:, 1], query_labels.float())

        if not (torch.isnan(query_loss) or torch.isinf(query_loss)):
            total_loss += query_loss.item()
            total_acc += accuracy(query_preds, query_labels)
            num_tasks += 1
            pbar.set_postfix(avg_loss=total_loss/num_tasks, avg_acc=total_acc/num_tasks)
        else:
            print(f"Warning: NaN/Inf query_loss in {mode} for task. Skipping task.")

    if num_tasks == 0:
        print(f"Warning: No tasks were successfully evaluated during {mode}.")
        return 0.0, 0.0
        
    avg_loss = total_loss / num_tasks
    avg_acc = total_acc / num_tasks
    return avg_loss, avg_acc

def set_seed(seed, device_type):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed_all(seed)

def main(args):
    global ARGS_REF_FOR_PRINTING
    ARGS_REF_FOR_PRINTING = args

    # Use 'cuda' if available, otherwise 'cpu'
    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device('cuda')
        print("Using CUDA.")
    else:
        device = torch.device('cpu')
        print("Using CPU.")

    set_seed(args.seed, device.type)
    
    # --- Create Output Directory ---
    num_tasks_str = f"{len(args.tasks)}tasks"
    run_name = f"arch_{args.architecture}_seed_{args.seed}_{num_tasks_str}"
    output_dir = Path(args.output_base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- Initialize Model and Meta-Learning Framework ---
    ModelClass = ARCHITECTURES[args.architecture]
    model = ModelClass()
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.inner_lr, first_order=args.first_order, allow_nograd=True)
    
    if device.type == 'cuda' and torch.cuda.device_count() > 1 and args.dataparallel:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        maml = torch.nn.DataParallel(maml)

    # --- Setup Optimizer, Scaler (for AMP) ---
    optimizer = torch.optim.AdamW(maml.parameters(), lr=args.outer_lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None
    if scaler: print("Using Automatic Mixed Precision (AMP).")

    # --- Setup Datasets ---
    data_path = Path(args.data_dir).resolve()
    
    print(f"Loading meta-training dataset with tasks: {args.tasks}")
    train_dataset = SameDifferentDataset(
        data_path,
        tasks=args.tasks,
        split='train',
        support_sizes=VARIABLE_SUPPORT_SIZES,
        query_size=FIXED_QUERY_SIZE
    )
    print(f"Loading meta-validation dataset with tasks: {args.tasks}")
    val_dataset = SameDifferentDataset(
        data_path,
        tasks=args.tasks,
        split='val',
        support_sizes=[10],
        query_size=FIXED_QUERY_SIZE
    )

    ALL_PB_TASKS = [
        'regular', 'lines', 'open', 'wider_line', 'scrambled',
        'random_color', 'arrows', 'irregular', 'filled', 'original'
    ]
    print(f"Loading meta-testing dataset with all PB tasks: {ALL_PB_TASKS}")
    test_datasets = {task: SameDifferentDataset(
                        data_path,
                        tasks=[task],
                        split='test',
                        support_sizes=[10],
                        query_size=FIXED_QUERY_SIZE,
                     ) for task in ALL_PB_TASKS}


    # --- Setup DataLoaders ---
    train_sampler = VariableSupportSampler(
        dataset=train_dataset,
        num_batches=args.num_meta_batches_per_epoch,
        meta_batch_size=args.meta_batch_size,
        available_support_sizes=VARIABLE_SUPPORT_SIZES
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_episodes,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_episodes)
    test_loaders = {task: DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_episodes)
                    for task, ds in test_datasets.items()}


    # --- Training Loop ---
    best_val_acc = -1.0
    best_model_state = None
    patience_counter = 0
    training_metrics = []

    print(f"Starting meta-training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(maml, train_loader, optimizer, device, args.adaptation_steps_train, scaler, epoch, args)
        
        val_loss, val_acc = -1.0, -1.0
        if (epoch + 1) % args.val_freq == 0:
            val_loss, val_acc = validate_or_test(maml, val_loader, device, args.adaptation_steps_test, 'Validating', epoch)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = copy.deepcopy(maml.module.state_dict() if isinstance(maml, torch.nn.DataParallel) else maml.state_dict())
                print(f"New best validation accuracy: {best_val_acc:.4f}. Saving model.")
            else:
                patience_counter += 1
                print(f"Validation accuracy did not improve. Patience: {patience_counter}/{args.patience}")

        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'validation_loss': val_loss if (epoch + 1) % args.val_freq == 0 else 'N/A',
            'validation_accuracy': val_acc if (epoch + 1) % args.val_freq == 0 else 'N/A',
            'best_validation_accuracy': best_val_acc,
            'patience_counter': patience_counter
        }
        training_metrics.append(metrics)

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break
        
        gc.collect()

    print("Meta-training finished.")

    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=4)
    print(f"Training metrics saved to {metrics_path}")

    if best_model_state:
        model_path = output_dir / 'best_model.pth'
        torch.save(best_model_state, model_path)
        print(f"Best model saved to {model_path}")
    else:
        print("Warning: No best model was saved (no validation improvement).")
        final_model_state = maml.module.state_dict() if isinstance(maml, torch.nn.DataParallel) else maml.state_dict()
        model_path = output_dir / 'final_model.pth'
        torch.save(final_model_state, model_path)
        print(f"Final model at last epoch saved to {model_path}")

    if best_model_state:
        print("Loading best model for meta-testing...")
        test_model_base = ModelClass(img_size=128, in_channels=3).to(device)
        test_model_base.load_state_dict(best_model_state)
        test_maml = l2l.algorithms.MAML(test_model_base, lr=args.inner_lr, first_order=args.first_order)
        if device.type == 'cuda' and torch.cuda.device_count() > 1 and args.dataparallel:
            test_maml = torch.nn.DataParallel(test_maml)
    else:
        print("No best model found. Using final model for meta-testing.")
        test_maml = maml

    test_results = {}
    print("Starting final meta-testing on all PB tasks...")
    for task_name, loader in test_loaders.items():
        test_loss, test_acc = validate_or_test(test_maml, loader, device, args.adaptation_steps_test, 'Testing', test_task_name=task_name)
        test_results[task_name] = {'loss': test_loss, 'accuracy': test_acc}
        print(f"  Test Task: {task_name} | Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    test_results_path = output_dir / 'meta_test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=4)
    print(f"Meta-test results saved to {test_results_path}")
    
    args_path = output_dir / 'args.json'
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {args_path}")

    print("Experiment run complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAML for Same-Different Task with Variable Tasks")
    
    # Paths and Setup
    parser.add_argument('--data_dir', type=str, required=True, help="Root directory of the PB HDF5 datasets.")
    parser.add_argument('--output_base_dir', type=str, required=True, help="Base directory to save experiment outputs.")
    parser.add_argument('--architecture', type=str, required=True, choices=ARCHITECTURES.keys(), help="Model architecture.")
    parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility.")
    parser.add_argument('--tasks', type=str, nargs='+', required=True, help="List of PB tasks to use for meta-training.")

    # MAML Hyperparameters
    parser.add_argument('--inner_lr', type=float, default=0.01, help="Inner loop learning rate.")
    parser.add_argument('--outer_lr', type=float, default=0.001, help="Outer loop learning rate.")
    parser.add_argument('--meta_batch_size', type=int, default=4, help="Number of tasks to sample per meta-update.")
    parser.add_argument('--adaptation_steps_train', type=int, default=5, help="Number of adaptation steps during training.")
    parser.add_argument('--adaptation_steps_test', type=int, default=15, help="Number of adaptation steps during testing.")
    parser.add_argument('--first_order', action='store_true', help="Use first-order approximation of MAML.")

    # Training Procedure
    parser.add_argument('--epochs', type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument('--num_meta_batches_per_epoch', type=int, default=100, help="Number of meta-batches per training epoch.")
    parser.add_argument('--patience', type=int, default=20, help="Patience for early stopping based on validation accuracy.")
    parser.add_argument('--val_freq', type=int, default=5, help="Frequency (in epochs) to run validation.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the AdamW optimizer.")
    parser.add_argument('--grad_clip', type=float, default=None, help="Gradient clipping norm value.")

    # System and Performance
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('--use_amp', action='store_true', help="Use Automatic Mixed Precision (AMP).")
    parser.add_argument('--force_cpu', action='store_true', help="Force use of CPU even if CUDA is available.")
    parser.add_argument('--dataparallel', action='store_true', help="Use DataParallel for multi-GPU training.")

    args = parser.parse_args()
    main(args)