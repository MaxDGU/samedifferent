import os
import h5py
import torch
import torch.nn as nn
import learn2learn as l2l
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import json
import argparse
import sys
from datetime import datetime

# --- Hardcoded Parameters for LR Finding ---
# Intentionally using lower LRs to test for stability
# The original script used inner_lr=0.05, outer_lr=0.001 which likely caused explosion
INNER_LR = 0.001
OUTER_LR = 0.0001
EPOCHS = 5  # Just enough to see if it's stable
BATCH_SIZE = 16 # Smaller batch size for faster iteration
ADAPTATION_STEPS = 5
MODEL_ARCH = 'conv6'
SEED = 42
DATA_DIR = 'data/meta_h5/pb'
SUPPORT_SIZE = 10
# --- End of Hardcoded Parameters ---


# Make sure the meta_baseline directory is in the Python path
# This allows us to import from utils_meta and the legacy models
script_dir = os.path.dirname(os.path.abspath(__file__))
meta_baseline_dir = os.path.join(os.path.dirname(script_dir), 'meta_baseline')
if meta_baseline_dir not in sys.path:
    sys.path.append(meta_baseline_dir)

# Now we can import from the subdirectories
from models.utils_meta import (
    SameDifferentDataset, validate, accuracy, EarlyStopping,
    train_epoch, collate_episodes, load_model
)
from baselines.models.conv6 import SameDifferentCNN as LegacyConv6

def main():
    """Main function to run the learning rate stability test."""
    
    # Use a unique output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_files/lr_finding_runs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Learning Rate Stability Test ---")
    print(f"Output Dir: {output_dir}")
    print(f"Model: Legacy {MODEL_ARCH}")
    print(f"Seed: {SEED}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    print(f"Inner LR: {INNER_LR}, Outer LR: {OUTER_LR}")
    print("------------------------------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Define tasks for PB dataset
    train_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
                   'random_color', 'arrows', 'irregular', 'filled', 'original']
        
        print("\nCreating datasets...")
    train_dataset = SameDifferentDataset(DATA_DIR, train_tasks, 'train', support_sizes=[SUPPORT_SIZE])
    val_dataset = SameDifferentDataset(DATA_DIR, train_tasks, 'val', support_sizes=[SUPPORT_SIZE])
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=collate_episodes)
        
    print(f"\nCreating legacy {MODEL_ARCH} model")
    model = LegacyConv6().to(device)
        
    # Wrap the model with MAML
        maml = l2l.algorithms.MAML(
            model, 
        lr=INNER_LR,
            first_order=False,
            allow_unused=True,
            allow_nograd=True
        )
        
    optimizer = torch.optim.Adam(maml.parameters(), lr=OUTER_LR)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
    print("\nStarting stability test training...")
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        # We will reuse the train_epoch and validate functions, which have their own TQDM bars
                train_loss, train_acc = train_epoch(
            maml, train_loader, optimizer, device, ADAPTATION_STEPS, scaler,
            debug_nan=True  # Add a flag to enable verbose NaN checking
                )
                
        # Check for NaN right after training epoch
        if np.isnan(train_loss) or np.isinf(train_loss):
            print("\n" + "="*50)
            print("TRAINING FAILED: Loss is NaN or Inf. The learning rates are unstable.")
            print(f"  Failed at Epoch: {epoch+1}")
            print(f"  Inner LR: {INNER_LR}, Outer LR: {OUTER_LR}")
            print("="*50)
            sys.exit(1) # Exit with an error code
                
                val_loss, val_acc = validate(
            maml, val_loader, device, adaptation_steps=ADAPTATION_STEPS
                )
                
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
    print("\n" + "="*50)
    print("STABILITY TEST PASSED!")
    print("The model trained for 5 epochs without loss becoming NaN.")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Consider using these learning rates for a full run:")
    print(f"  --inner_lr {INNER_LR}")
    print(f"  --outer_lr {OUTER_LR}")
    print("="*50)


if __name__ == '__main__':
    # Ensure the script can find the necessary modules
    # This is a bit of a hack for script execution, a proper package structure would be better
    current_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
            
    # Need to re-import with the updated path for command-line execution
    from meta_baseline.models.utils_meta import (
        SameDifferentDataset, validate, accuracy, EarlyStopping,
        train_epoch, collate_episodes, load_model
    )
    from baselines.models.conv6 import SameDifferentCNN as LegacyConv6
    
    main() 