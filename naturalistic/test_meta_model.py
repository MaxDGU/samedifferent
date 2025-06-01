#!/usr/bin/env python
# naturalistic/test_meta_model.py

import torch
import torch.nn as nn
import torch.optim as optim # Not strictly needed for testing but good for consistency if MAML needs it
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import sys
import gc
import learn2learn as l2l

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
    from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
    from baselines.models.conv6 import SameDifferentCNN as Conv6CNN
    print("Successfully imported Conv{2,4,6}lrCNN models from baselines.models")
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Please ensure conv2.py, conv4.py, conv6.py are in baselines/models/ and project root is in PYTHONPATH.")
    exit(1)

ARCHITECTURES = {
    'conv2lr': Conv2CNN,
    'conv4lr': Conv4CNN,
    'conv6lr': Conv6CNN
}

def accuracy(predictions, targets):
    """Binary accuracy: assumes predictions are logits [N, 2], targets are [N]."""
    predicted_labels = torch.argmax(predictions, dim=1)
    correct = (predicted_labels == targets).float()
    return correct.mean()

class MetaNaturalisticDataset(Dataset):
    """
    Dataset for loading meta-learning episodes from the naturalistic HDF5 files.
    (Copied from meta_naturalistic_train.py)
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = Path(h5_path)
        self.user_transform = transform 

        from torchvision import transforms as T 
        self.processing_transform = T.Compose([
            T.ToPILImage(), 
            T.Resize((128, 128)),
            T.ToTensor(), 
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self._file = None 
        self.episode_keys = []
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        try:
            self._file = h5py.File(self.h5_path, 'r')
            self.episode_keys = sorted([k for k in self._file.keys() if k.startswith('episode_')])
            if not self.episode_keys:
                raise ValueError(f"No episode groups found in {self.h5_path}")
            print(f"Loaded {len(self.episode_keys)} episodes from {self.h5_path.name}")
        except Exception as e:
            print(f"Error opening or reading keys from {self.h5_path}: {e}")
            if self._file: self._file.close()
            raise

    def __len__(self):
        return len(self.episode_keys)

    def __getitem__(self, idx):
        if not hasattr(self, '_file') or not self._file:
             self._file = h5py.File(self.h5_path, 'r')
        episode_key = self.episode_keys[idx]
        try:
            ep_group = self._file[episode_key]
            support_images = ep_group['support_images'][()]
            support_labels = ep_group['support_labels'][()]
            query_images = ep_group['query_images'][()]
            query_labels = ep_group['query_labels'][()]
        except KeyError as e:
            print(f"Error accessing data for key {episode_key} in {self.h5_path}: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error reading episode {episode_key}: {e}")
            raise

        transformed_support_images = [self.processing_transform(img) for img in support_images]
        support_images_tensor = torch.stack(transformed_support_images)
        transformed_query_images = [self.processing_transform(img) for img in query_images]
        query_images_tensor = torch.stack(transformed_query_images)
        support_labels_tensor = torch.from_numpy(support_labels).long()
        query_labels_tensor = torch.from_numpy(query_labels).long() 

        if self.user_transform:
            support_images_tensor = torch.stack([self.user_transform(img) for img in support_images_tensor])
            query_images_tensor = torch.stack([self.user_transform(img) for img in query_images_tensor])
        return support_images_tensor, support_labels_tensor, query_images_tensor, query_labels_tensor

    def close(self):
        if hasattr(self, '_file') and self._file:
            self._file.close()
            self._file = None
    def __del__(self):
        self.close()

def fast_adapt(batch, learner, loss_fn, adaptation_steps, device, episode_idx_debug="N/A", force_first_order_adapt=False):
    """
    Manual implementation of MAML adaptation and evaluation step.
    (Copied and potentially simplified from meta_naturalistic_train.py - ensure critical parts for eval are kept)
    """
    support_images, support_labels, query_images, query_labels = batch
    support_images, support_labels = support_images.to(device), support_labels.to(device)
    query_images, query_labels = query_images.to(device), query_labels.to(device)

    original_learner_training_state = learner.training
    learner.train() 

    is_adapt_first_order = learner.first_order 
    if force_first_order_adapt:
        is_adapt_first_order = True
    
    for step in range(adaptation_steps):
        with torch.enable_grad(): # Still need grads for adaptation, even if outer loop is eval
            # Ensure params require grad after cloning for adaptation
            for p in learner.module.parameters(): # .module accesses the underlying model
                 p.requires_grad_(True)

            adaptation_logits = learner(support_images)
            adaptation_error = loss_fn(adaptation_logits, support_labels)
            if torch.isnan(adaptation_error) or torch.isinf(adaptation_error):
                print(f"WARNING (fast_adapt {episode_idx_debug}, Inner Step {step+1}): Unstable adaptation_error: {adaptation_error.item()}")
                # return adaptation_error, torch.tensor(0.0) # Early exit if adaptation unstable
        
        learner.adapt(adaptation_error, first_order=is_adapt_first_order)

    learner.train(original_learner_training_state) # Restore mode for query set eval

    evaluation_logits = learner(query_images)
    evaluation_error = loss_fn(evaluation_logits, query_labels)
    evaluation_acc = accuracy(evaluation_logits, query_labels)

    if torch.isnan(evaluation_error) or torch.isinf(evaluation_error):
        print(f"WARNING (fast_adapt {episode_idx_debug}): Unstable evaluation_error: {evaluation_error.item()}")

    return evaluation_error, evaluation_acc

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained MAML model on Naturalistic Same-Different Dataset")
    # Paths
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing test.h5')
    parser.add_argument('--base_log_dir', type=str, required=True, help='Base directory where logs for runs are stored (e.g., logs_naturalistic_meta)')
    # Model and Run specifiers
    parser.add_argument('--model', type=str, required=True, choices=['conv2lr', 'conv4lr', 'conv6lr'], help='Model architecture')
    parser.add_argument('--seed', type=int, required=True, help='Random seed of the trained model')
    # Meta-Learning Hyperparameters (relevant for loading and adaptation)
    parser.add_argument('--inner_lr', type=float, default=1e-5, help='Inner loop learning rate (must match training)')
    parser.add_argument('--inner_steps', type=int, default=5, help='Number of adaptation steps (must match training)')
    parser.add_argument('--first_order', action='store_true', help='Set if the loaded model was trained with first-order MAML.') # For MAML wrapper re-init
    # Testing Settings
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--max_test_episodes', type=int, default=None, help='Maximum number of episodes to use for testing. Default: all in test set.')
    # AMP - Added for consistency with training script structure if any AMP state is saved/needed
    parser.add_argument('--amp_dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'], help='AMP dtype if used.')


    return parser.parse_args()

def main():
    args = parse_args()

    # Setup seed (for reproducibility of any minor random ops if they exist, though testing should be deterministic)
    random.seed(args.seed) # This seed is more for identifying the run
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    
    # Construct the specific log directory for the trained model
    run_log_dir = Path(args.base_log_dir) / args.model / f"seed_{args.seed}"
    best_model_path = run_log_dir / f"{args.model}_best.pth"

    print(f"Attempting to load model for testing: Arch={args.model}, Seed={args.seed}")
    print(f"Args for testing: {args}")
    print(f"Using device: {device}")
    print(f"Looking for best model at: {best_model_path}")

    if not best_model_path.exists():
        print(f"ERROR: Best model checkpoint not found at {best_model_path}. Exiting.")
        return

    # --- Load Model ---
    print(f"Loading base model: {args.model}")
    try:
        model = ARCHITECTURES[args.model]()
    except KeyError:
        print(f"ERROR: Unknown model type: {args.model}")
        return
    
    model.to(device)

    # --- Wrap model with MAML ---
    # Initialize MAML wrapper similarly to how it was during training
    # The `first_order` argument here should match the one used for training the loaded checkpoint.
    maml = l2l.algorithms.MAML(
        model,
        lr=args.inner_lr, # This LR is used by maml.clone() if adapting.
        first_order=args.first_order, 
        allow_unused=True,
        allow_nograd=True 
    )
    maml.to(device)

    # --- Load Checkpoint ---
    print(f"Loading checkpoint from: {best_model_path}")
    try:
        # THIS IS THE KEY CHANGE: weights_only=False
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False) 
        
        # Load base model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded base model state_dict.")

        # Load MAML wrapper state (if present and important for evaluation)
        if 'maml_state_dict' in checkpoint:
            maml.load_state_dict(checkpoint['maml_state_dict'])
            print("Loaded MAML wrapper state_dict.")
        else:
            # If maml_state_dict is not in the checkpoint, the maml wrapper will use the
            # parameters from the 'model' it wrapped, which we just loaded.
            # This is generally fine for evaluation if MAML itself doesn't have learnable params beyond the base model.
            print("MAML state_dict not found in checkpoint or not loaded. MAML wrapper will use the loaded base model's weights.")

        # Load args from checkpoint to verify consistency if needed
        if 'args' in checkpoint:
            train_args = checkpoint['args']
            print(f"Loaded training args from checkpoint: {train_args}")
            # You could add checks here, e.g., if train_args.inner_lr != args.inner_lr, print warning
            if train_args.inner_lr != args.inner_lr:
                print(f"WARNING: Current inner_lr ({args.inner_lr}) differs from training inner_lr ({train_args.inner_lr}).")
            if train_args.inner_steps != args.inner_steps:
                 print(f"WARNING: Current inner_steps ({args.inner_steps}) differs from training inner_steps ({train_args.inner_steps}).")
            if train_args.first_order != args.first_order:
                 print(f"WARNING: Current first_order ({args.first_order}) differs from training first_order ({train_args.first_order}). Model may behave unexpectedly if mismatched.")


        print("Best model checkpoint loaded successfully.")

    except Exception as e:
        print(f"Error loading best model checkpoint: {e}. Skipping testing.")
        return

    # --- Load Test Data ---
    transform = None # No extra transforms for now
    test_h5_path = Path(args.data_dir) / 'test.h5'
    if not test_h5_path.exists():
        print(f"Test data file not found: {test_h5_path}. Skipping testing.")
        return
    
    try:
        test_dataset = MetaNaturalisticDataset(test_h5_path, transform=transform)
        # Determine number of test episodes
        num_test_episodes_to_run = len(test_dataset)
        if args.max_test_episodes is not None and args.max_test_episodes > 0:
            num_test_episodes_to_run = min(len(test_dataset), args.max_test_episodes)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers) # Batch size 1 for meta-testing
        print(f"Testing on {num_test_episodes_to_run} episodes (out of {len(test_dataset)} total in test.h5).")

    except Exception as e:
        print(f"Error loading test dataset: {e}. Skipping testing.")
        if 'test_dataset' in locals() and hasattr(test_dataset, 'close'): test_dataset.close()
        return

    # --- Testing Loop ---
    maml.eval() # Set MAML (and base model) to eval mode
    criterion = nn.CrossEntropyLoss() # Assuming this was the criterion used
    meta_test_loss = 0.0
    meta_test_acc = 0.0
    test_episodes_processed = 0
    
    # AMP setup for testing phase for consistency (if model expects it or for performance)
    # Determine if AMP should be enabled for autocast based on device and availability
    is_cuda_and_amp_enabled = args.device == 'cuda' and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16

    pbar_test = tqdm(test_loader, total=num_test_episodes_to_run, desc="Testing")
    
    with torch.no_grad(): # Outer loop is no_grad for testing
        for batch_idx, batch in enumerate(pbar_test):
            if batch_idx >= num_test_episodes_to_run:
                break

            task_batch = [d.squeeze(0) for d in batch] 
            
            # Autocast for the adaptation and evaluation, consistent with training script structure
            with torch.amp.autocast(device_type=args.device if is_cuda_and_amp_enabled else 'cpu', 
                                    dtype=amp_dtype if is_cuda_and_amp_enabled else torch.float32, 
                                    enabled=is_cuda_and_amp_enabled):
                learner = maml.clone() # Clone for adaptation
                
                # Force first_order for adaptation during meta-testing for speed and stability,
                # as typically done, unless the model specifically needs second-order signals at test time.
                # The 'args.first_order' refers to how the *meta-model* was trained.
                # Adaptation during testing is often simplified.
                evaluation_loss, evaluation_acc = fast_adapt(task_batch,
                                                               learner,
                                                               criterion,
                                                               args.inner_steps,
                                                               device,
                                                               episode_idx_debug=f"test_ep_{batch_idx}",
                                                               force_first_order_adapt=True) 
            
            if torch.isnan(evaluation_loss) or torch.isinf(evaluation_loss):
                print(f"ERROR: NaN/Inf loss ({evaluation_loss.item()}) in test episode {batch_idx}. Skipping episode.")
                del learner, task_batch # , evaluation_loss, evaluation_acc (may not be defined if error early)
                if device == 'cuda': torch.cuda.empty_cache()
                continue


            meta_test_loss += evaluation_loss.item()
            meta_test_acc += evaluation_acc.item()
            test_episodes_processed += 1
            pbar_test.set_postfix(loss=f"{evaluation_loss.item():.4f}", acc=f"{evaluation_acc.item():.3f}")

            del learner, task_batch, evaluation_loss, evaluation_acc
            if device == 'cuda': torch.cuda.empty_cache()

    if test_episodes_processed > 0:
        avg_test_loss = meta_test_loss / test_episodes_processed
        avg_test_acc = meta_test_acc / test_episodes_processed
        print(f"
--- Test Results ---")
        print(f"Model: {args.model}, Seed: {args.seed}")
        print(f"Processed {test_episodes_processed} test episodes.")
        print(f"Average Test Loss: {avg_test_loss:.4f}")
        print(f"Average Test Accuracy: {avg_test_acc:.3f}")
    else:
        print("No episodes processed in the test set, or all resulted in errors.")
    
    gc.collect()
    if hasattr(test_dataset, 'close'): # Ensure HDF5 file is closed
        test_dataset.close()
    print("Testing script finished.")

if __name__ == "__main__":
    main()

