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
from torch.utils.data import Sampler # For custom sampler

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
                # If adaptation failed for this task, we cannot evaluate it.
                # We'll add a very high loss and zero accuracy for this task to penalize,
                # or simply skip its contribution to the meta-loss.
                # For now, let's skip its contribution to avoid NaNs in meta-loss if we assigned a fixed large loss.
                # Effective meta_batch_size for averaging will be smaller if tasks fail.
                # This needs careful consideration: should a failed task contribute a max loss or be ignored?
                # Ignoring it might be too lenient. Let's add a large loss and 0 acc to make it count.
                print(f"Warning: Task {task_idx} adaptation failed. Assigning high loss and 0 acc.")
                query_loss_for_task = torch.tensor(100.0, device=device) # Arbitrary large loss
                acc = 0.0
            else: 
                # Evaluation Phase (Query Set)
                with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                    query_preds = learner(query_images)
                    query_loss_for_task = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.float()
                    )
                print_debug_stats("QueryPhase", current_arch, epoch_num, meta_batch_idx, task_idx=task_idx, query_preds=query_preds, query_loss=query_loss_for_task, learner=learner)

                if torch.isnan(query_loss_for_task) or torch.isinf(query_loss_for_task):
                    print(f"CRITICAL: NaN/Inf query_loss_for_task. Arch: {current_arch}, E{epoch_num} B{meta_batch_idx} T{task_idx}. Loss: {query_loss_for_task.item()}. Assigning high loss and 0 acc.")
                    query_loss_for_task = torch.tensor(100.0, device=device) # Arbitrary large loss
                    acc = 0.0
                else:
                    acc = accuracy(query_preds, query_labels)
            
            sum_query_losses_for_meta_batch += query_loss_for_task
            sum_query_accs_for_meta_batch += acc
            # End of per-task processing in meta-batch
        
        # Meta-update
        meta_loss_for_batch = sum_query_losses_for_meta_batch / actual_meta_batch_size
        meta_acc_for_batch = sum_query_accs_for_meta_batch / actual_meta_batch_size

        if torch.isnan(meta_loss_for_batch) or torch.isinf(meta_loss_for_batch):
            print(f"CRITICAL: NaN/Inf meta_loss_for_batch detected before backward. Arch: {current_arch}, E{epoch_num} B{meta_batch_idx}. Loss: {meta_loss_for_batch.item()}. Skipping batch.")
            optimizer.zero_grad(set_to_none=True) 
            del sum_query_losses_for_meta_batch, meta_loss_for_batch 
            gc.collect()
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            continue 

        print_debug_stats("PreOuterBackward", current_arch, epoch_num, meta_batch_idx, maml_model=maml, query_loss=meta_loss_for_batch)

        if scaler:
            scaler.scale(meta_loss_for_batch).backward()
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(maml.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            meta_loss_for_batch.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(maml.parameters(), args.grad_clip_norm)
            optimizer.step()
        
        print_debug_stats("PostOptimizerStep", current_arch, epoch_num, meta_batch_idx, maml_model=maml)

        total_meta_loss += meta_loss_for_batch.item()
        total_meta_acc += meta_acc_for_batch # meta_acc_for_batch is already a float
        num_meta_batches_processed += 1
        
        pbar.set_postfix(meta_loss=meta_loss_for_batch.item(), meta_acc=meta_acc_for_batch)

    if num_meta_batches_processed == 0: # Handle case where all batches were skipped
        print("Warning: No meta-batches processed in this epoch.")
        return 0.0, 0.0 
        
    avg_meta_loss = total_meta_loss / num_meta_batches_processed
    avg_meta_acc = total_meta_acc / num_meta_batches_processed
    return avg_meta_loss, avg_meta_acc

def validate_or_test(maml, dataloader, device, adaptation_steps, mode='Validating', epoch_num=None, test_task_name=None):
    maml.eval()
    total_meta_loss = 0
    total_meta_acc = 0
    num_meta_batches_processed = 0

    desc_str = mode
    if epoch_num is not None:
        desc_str = f'Epoch {epoch_num+1} {mode}'
    if test_task_name:
        desc_str = f'{mode} on {test_task_name}'
        
    pbar = tqdm(dataloader, desc=desc_str)
    for current_meta_batch_data in pbar:
        sum_query_losses_for_meta_batch = 0.0
        sum_query_accs_for_meta_batch = 0.0
        
        actual_meta_batch_size = current_meta_batch_data['support_images'].size(0)
        if actual_meta_batch_size == 0:
            continue

        for task_idx in range(actual_meta_batch_size):
            learner = maml.clone()
            
            # For testing, we often want batch norm to be in training mode during adaptation
            # and eval mode during query evaluation on the test task.
            # For validation during meta-training, maml itself is eval, so BN uses running stats.
            if mode == 'Testing':
                learner.train() # Enable train mode for adaptation to update BN stats for this task.
            # else, learner inherits maml.training status (which is False/eval)

            support_images = current_meta_batch_data['support_images'][task_idx].to(device, non_blocking=True)
            support_labels = current_meta_batch_data['support_labels'][task_idx].to(device, non_blocking=True)
            query_images = current_meta_batch_data['query_images'][task_idx].to(device, non_blocking=True)
            query_labels = current_meta_batch_data['query_labels'][task_idx].to(device, non_blocking=True)
            
            # Inner loop: adaptation
            for _ in range(adaptation_steps):
                support_preds = learner(support_images)
                support_loss = F.binary_cross_entropy_with_logits(
                    support_preds[:, 1], support_labels.float()
                )
                if torch.isnan(support_loss) or torch.isinf(support_loss):
                    print(f"Warning: NaN/Inf support_loss during {mode} adaptation. Skipping adapt for this step.")
                    break
                learner.adapt(support_loss, allow_unused=True, allow_nograd=True) # allow_unused=True
            else: # Continue if adaptation completed
                if mode == 'Testing':
                    learner.eval() # Switch to eval mode for query evaluation using adapted BN stats.
                
                with torch.no_grad():
                    query_preds = learner(query_images)
                    query_loss_for_task = F.binary_cross_entropy_with_logits(
                        query_preds[:, 1], query_labels.float()
                    )
                    query_acc_for_task = accuracy(query_preds, query_labels.long())
                
                if not (torch.isnan(query_loss_for_task) or torch.isinf(query_loss_for_task)):
                    sum_query_losses_for_meta_batch += query_loss_for_task.item()
                    sum_query_accs_for_meta_batch += query_acc_for_task.item() if isinstance(query_acc_for_task, torch.Tensor) else query_acc_for_task
        
        if actual_meta_batch_size > 0 and num_meta_batches_processed < float('inf'): # check ensures at least one task processed
            avg_query_loss_for_meta_batch = sum_query_losses_for_meta_batch / actual_meta_batch_size
            avg_query_acc_for_meta_batch = sum_query_accs_for_meta_batch / actual_meta_batch_size
            
            total_meta_loss += avg_query_loss_for_meta_batch
            total_meta_acc += avg_query_acc_for_meta_batch
            num_meta_batches_processed += 1
        
        pbar.set_postfix({
            'loss': f'{total_meta_loss / num_meta_batches_processed:.4f}' if num_meta_batches_processed > 0 else 'N/A',
            'acc': f'{total_meta_acc / num_meta_batches_processed:.4f}' if num_meta_batches_processed > 0 else 'N/A'
        })

    if num_meta_batches_processed == 0:
        print(f"Warning: {mode} phase completed with no meta-batches processed successfully.")
        return 0.0, 0.0
    return total_meta_loss / num_meta_batches_processed, total_meta_acc / num_meta_batches_processed

def set_seed(seed, device_type):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        # For speed, if input sizes are consistent (common in MAML episodes)
        torch.backends.cudnn.benchmark = True
        # For full reproducibility, set deterministic = True. This might impact performance.
        # Choose one or the other based on priority.
        torch.backends.cudnn.deterministic = False 
    else: # For CPU or other devices
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True # Keep deterministic for non-CUDA for consistency if desired


def main(args):
    global ARGS_REF_FOR_PRINTING
    ARGS_REF_FOR_PRINTING = args # Set for helper function

    # Determine device
    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")

    set_seed(args.seed, device.type)

    # --- Output Directory Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Experiment name reflects variable S and fixed Q (implicitly Q=2)
    exp_name = f"exp_all_tasks_fomaml_{args.architecture}_seed{args.seed}_Svar_Q{FIXED_QUERY_SIZE}_{timestamp}"
    output_dir = Path(args.output_base_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        # Add VARIABLE_SUPPORT_SIZES and FIXED_QUERY_SIZE to saved args for clarity
        args_to_save = vars(copy.deepcopy(args))
        args_to_save['variable_support_sizes_used'] = VARIABLE_SUPPORT_SIZES
        args_to_save['fixed_query_size_used'] = FIXED_QUERY_SIZE
        json.dump(args_to_save, f, indent=4)

    # --- Model Selection ---
    model_constructor = ARCHITECTURES.get(args.architecture)
    if model_constructor is None:
        raise ValueError(f"Unsupported architecture: {args.architecture}")
    
    # model_constructor constructor (e.g., Conv2CNN_lr) does not take track_running_stats directly.
    # This property is set on BatchNorm2d layers within the model definition.
    # The args.track_running_stats_maml argument was intended to guide MAML behaviour,
    # but l2l.MAML handles BN differently based on model.train()/eval() modes.
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

    # Save initial model weights before wrapping with MAML
    torch.save(model.state_dict(), output_dir / 'initial_model.pth')
    print(f"Saved initial model weights to {output_dir / 'initial_model.pth'}")

    effective_inner_lr = args.inner_lr # Use the inner_lr passed from run_all_tasks_pb.py directly
    # # Adjust inner_lr for deeper architectures if prone to instability - REMOVED THIS BLOCK
    # if args.architecture in ['conv4', 'conv6']:
    #     effective_inner_lr = 0.001  # Reduced inner LR for conv4/conv6
    #     print(f"INFO: Using reduced inner_lr for {args.architecture}: {effective_inner_lr} (original CLI arg was {args.inner_lr})")

    maml = l2l.algorithms.MAML(
        model,
        lr=effective_inner_lr,
        first_order=args.first_order,
        allow_unused=True, 
        allow_nograd=True
    )
    maml.to(device)
    optimizer = torch.optim.AdamW(maml.parameters(), lr=args.outer_lr, weight_decay=args.weight_decay)
    
    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler() if use_cuda and args.use_amp else None
    if scaler: print("Using Automatic Mixed Precision (AMP).")

    # --- Datasets and DataLoaders ---
    train_task_names = ALL_PB_TASKS
    val_task_names = ALL_PB_TASKS

    print(f"Meta-training on tasks: {train_task_names} with variable support sizes: {VARIABLE_SUPPORT_SIZES}, query size: {FIXED_QUERY_SIZE}")
    print(f"Meta-validating on tasks: {val_task_names} with variable support sizes: {VARIABLE_SUPPORT_SIZES}, query size: {FIXED_QUERY_SIZE}")

    # Instantiate dataset with ALL desired support sizes
    # SameDifferentDataset constructor takes `support_sizes` (plural)
    train_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=train_task_names,
        split='train',
        support_sizes=VARIABLE_SUPPORT_SIZES, # Pass the list of S sizes
    )
    val_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=val_task_names,
        split='val',
        support_sizes=VARIABLE_SUPPORT_SIZES, # Pass the list of S sizes
    )

    # Create custom samplers
    train_sampler = VariableSupportSampler(
        dataset=train_dataset,
        num_batches=args.num_meta_batches_per_epoch, # This is number of meta-batches
        meta_batch_size=args.meta_batch_size,
        available_support_sizes=VARIABLE_SUPPORT_SIZES
    )
    val_sampler = VariableSupportSampler(
        dataset=val_dataset,
        num_batches=args.num_val_meta_batches, # This is number of meta-batches
        meta_batch_size=args.meta_batch_size,
        available_support_sizes=VARIABLE_SUPPORT_SIZES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler, # Use batch_sampler instead of batch_size and shuffle
        num_workers=args.num_workers,
        pin_memory=True if use_cuda else False,
        collate_fn=collate_episodes
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler, # Use batch_sampler
        num_workers=args.num_workers,
        pin_memory=True if use_cuda else False,
        collate_fn=collate_episodes
    )

    # --- Training Loop ---
    best_val_acc = -1.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    metrics_history = []

    for epoch in range(args.epochs):
        epoch_metrics = {'epoch': epoch + 1}
        
        train_loss, train_acc = train_epoch(maml, train_loader, optimizer, device, args.adaptation_steps_train, scaler, epoch, args)
        epoch_metrics['train_loss'] = train_loss
        epoch_metrics['train_acc'] = train_acc
        print(f"Epoch {epoch+1}/{args.epochs} Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        if (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1:
            val_loss, val_acc = validate_or_test(maml, val_loader, device, args.adaptation_steps_val, mode='Validating', epoch_num=epoch)
            epoch_metrics['val_loss'] = val_loss
            epoch_metrics['val_acc'] = val_acc
            print(f"Epoch {epoch+1}/{args.epochs}          Val Loss: {val_loss:.4f},   Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc + args.improvement_threshold:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model based on validation accuracy
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(), # Save underlying model's state
                    'maml_state_dict': maml.state_dict(),   # Save MAML wrapper's state
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'args': vars(args)
                }
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"Epoch {epoch+1}: New best model saved with Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
        
        metrics_history.append(epoch_metrics)
        with open(output_dir / 'metrics_history.json', 'w') as f_metrics:
            json.dump(metrics_history, f_metrics, indent=4)

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch+1} due to no improvement in val_acc for {args.patience} validation cycles.")
            break
        
        # Clear CUDA cache
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    print("Meta-training finished.")

    # --- Testing Phase ---
    print("Loading best model for testing...")
    try:
        checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
        # Load model state into the original model, then potentially re-wrap with MAML if needed for test_model
        # Or, ensure maml object (if used by test_model) is loaded with its state
        model.load_state_dict(checkpoint['model_state_dict'])
        maml.load_state_dict(checkpoint['maml_state_dict']) # Ensure MAML specific state is also loaded
        print(f"Best model from epoch {checkpoint.get('epoch', 'N/A')} loaded with Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
    except FileNotFoundError:
        print("ERROR: Best model checkpoint ('best_model.pth') not found. Testing will use the last state of the model.")
        # Potentially save the last model state here if desired, or just proceed.
        torch.save({
            'epoch': epoch + 1 if 'epoch' in locals() else args.epochs, # last completed epoch or total epochs
            'model_state_dict': model.state_dict(),
            'maml_state_dict': maml.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args)
        }, output_dir / 'last_model_state_before_test.pth')


    test_results = {}
    avg_test_acc_all_tasks = []
    avg_test_loss_all_tasks = []

    print("Starting testing on all individual PB tasks...")
    for test_task_name in ALL_PB_TASKS:
        print(f"--- Testing on task: {test_task_name} ---")
        test_task_dataset = SameDifferentDataset(
            data_dir=args.data_dir,
            tasks=[test_task_name],
            split='test',
            support_sizes=[args.support_size_test], # Uses args.support_size_test
            # query_size handled by HDF5 content based on args.query_size_test if files are named accordingly
            # or if dataset internally uses a fixed Q or args.query_size_test to select episodes
            # For now, assume test HDF5 files exist for the given S_test (Q_test implicit)
        )
        test_task_loader = DataLoader(
            test_task_dataset,
            batch_size=args.meta_batch_size, # For testing, can make this larger if desired for more test episodes
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if use_cuda else False,
            collate_fn=collate_episodes
        )

        test_loss, test_acc = validate_or_test(
            maml, test_task_loader, device, args.adaptation_steps_test,
            mode='Testing', test_task_name=test_task_name
        )
        
        print(f"Task: {test_task_name} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        test_results[test_task_name] = {'loss': test_loss, 'accuracy': test_acc}
        avg_test_acc_all_tasks.append(test_acc)
        avg_test_loss_all_tasks.append(test_loss)

    final_summary = {
        'experiment_name': exp_name,
        'args': vars(args),
        'best_validation_accuracy': best_val_acc,
        'best_validation_loss': best_val_loss,
        'individual_task_test_results': test_results,
        'average_test_accuracy_all_tasks': np.mean(avg_test_acc_all_tasks) if avg_test_acc_all_tasks else 0.0,
        'std_test_accuracy_all_tasks': np.std(avg_test_acc_all_tasks) if avg_test_acc_all_tasks else 0.0,
        'average_test_loss_all_tasks': np.mean(avg_test_loss_all_tasks) if avg_test_loss_all_tasks else 0.0,
    }

    with open(output_dir / 'final_summary_results.json', 'w') as f_summary:
        json.dump(final_summary, f_summary, indent=4)

    print("--- Final Summary ---")
    print(f"Experiment: {exp_name}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    if avg_test_acc_all_tasks:
        print(f"Average Test Accuracy (all tasks): {np.mean(avg_test_acc_all_tasks):.4f} +/- {np.std(avg_test_acc_all_tasks):.4f}")
    else:
        print("Average Test Accuracy (all tasks): N/A (no tasks tested or all failed)")
    print(f"Individual test results saved in: {output_dir / 'final_summary_results.json'}")
    print(f"Full metrics history saved in: {output_dir / 'metrics_history.json'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MAML training on all PB tasks with variable support sizes.')

    # --- Paths and Basic Setup ---
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb', help='Directory for PB HDF5 data files.')
    parser.add_argument('--output_base_dir', type=str, default='./results/maml_all_tasks', help='Base directory to save experiment results.')
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'], help='Model architecture.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if CUDA is available.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader. 0 for main process.')

    # --- MAML specific ---
    parser.add_argument('--first_order', action='store_true', help='Use First-Order MAML (FOMAML). If not set, 2nd order MAML is used.')
    
    # --- Meta-learning parameters ---
    parser.add_argument('--meta_batch_size', type=int, default=8, help='Number of tasks per meta-batch.')
    parser.add_argument('--inner_lr', type=float, default=0.05, help='Learning rate for the inner loop adaptation.')
    parser.add_argument('--outer_lr', type=float, default=0.001, help='Learning rate for the outer loop.')
    parser.add_argument('--adaptation_steps_train', type=int, default=5, help='Number of adaptation steps during the meta-training inner loop.')
    parser.add_argument('--adaptation_steps_val', type=int, default=10, help='Number of adaptation steps during validation.')
    parser.add_argument('--adaptation_steps_test', type=int, default=10, help='Number of adaptation steps during meta-testing.')
    
    # Note: VARIABLE_SUPPORT_SIZES and FIXED_QUERY_SIZE are hardcoded in the script for now.
    # These could be made into args if more flexibility is needed for HDF5 structure.
    # For testing phase:
    parser.add_argument('--support_size_test', type=int, default=10, help='Support size for the test set (if using a different one).')
    # This argument is logged but does not affect data loading, which is determined by the HDF5 file structure.
    # It is kept for compatibility with previous logs but should not be the source of truth for query size.
    parser.add_argument('--query_size_test', type=int, default=2, help='Query size for the test set. (LOGGING ONLY)')


    # --- Training loop ---
    parser.add_argument('--epochs', type=int, default=100, help='Number of meta-training epochs.')
    parser.add_argument('--num_meta_batches_per_epoch', type=int, default=32, help='Number of meta-batches per training epoch.') # Matches MAML_NUM_ADAPTATION_SAMPLES in della_datapar
    parser.add_argument('--num_val_meta_batches', type=int, default=25, help='Number of meta-batches for a full validation pass.')
    parser.add_argument('--val_freq', type=int, default=1, help='Frequency (in epochs) to perform validation.')
    
    # --- Early stopping & AMP & Other ---
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--improvement_threshold', type=float, default=0.001, help='Minimum improvement in val_acc to reset patience.')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (AMP) for training if CUDA is available.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for the outer loop optimizer.')
    parser.add_argument('--grad_clip_norm', type=float, default=None, help='Max norm for gradient clipping. Default is None (no clipping).')


    # --- Debug related ---
    parser.add_argument('--debug_bn_grads', action='store_true', help='Enable detailed BN grad logging (if model supports it).')
    
    main_args = parser.parse_args()
    
    exit_code = 0 # Default to success
    try:
        main(main_args)
    except Exception as e:
        print(f"Error during MAML script execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        print(f"MAML script finished with exit code: {exit_code}")
        sys.exit(exit_code) 