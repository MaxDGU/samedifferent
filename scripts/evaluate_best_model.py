import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
import learn2learn as l2l
import json

# Add the root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_for_loading import SameDifferentCNN_from_checkpoint
from meta_baseline.models.utils_meta import SameDifferentDataset, collate_episodes, validate

def main():
    """Main function to evaluate the best-performing model."""
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. Load Model ---
    model_path = 'maml_pbweights_conv6/model_seed_4_pretesting.pt'
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)

    # Use the reverse-engineered model
    model = SameDifferentCNN_from_checkpoint().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load the state dict directly into our new model
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded model_state_dict into the reconstructed model.")
        
    # Now, wrap the loaded model with MAML
    maml = l2l.algorithms.MAML(model, lr=0.05) # lr doesn't matter for testing
        
    print(f"Loaded trained model from {model_path}")

    # --- 3. Load Data ---
    data_dir = 'data/meta_h5/pb'
    task = 'regular' # We'll test on the 'regular' task first
    test_adaptation_steps = 15 # From the training script
    test_support_size = [10] # From the training script

    print(f"\\nLoading '{task}' test dataset...")
    test_dataset = SameDifferentDataset(data_dir, [task], 'test', support_sizes=test_support_size)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_episodes)

    # --- 4. Evaluate Performance ---
    print("Evaluating model performance...")
    
    # The validate function from the training scripts handles the MAML adaptation loop
    test_loss, test_acc = validate(
        maml,
        test_loader,
        device,
        adaptation_steps=test_adaptation_steps,
        is_test=True
    )

    print(f"\\n--- Evaluation Results for Task: '{task}' ---")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Loss: {test_loss:.4f}")
    
    if test_acc > 0.95:
        print("\\nSuccess! Model performance is above 95%. We can now proceed with circuit analysis.")
    else:
        print("\\nWarning: Model performance is below 95%. The results of circuit analysis may not be meaningful.")

if __name__ == '__main__':
    main() 