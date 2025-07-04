import torch
import sys
import os
from collections import OrderedDict

# Add the project root to the Python path
# This is necessary for the script to find the model definitions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baselines.models.conv6 import SameDifferentCNN

def analyze_model_weights(model_path):
    """
    Loads a model's state_dict and checks its architecture against the baseline conv6 model.
    """
    print(f"--- Analyzing: {model_path} ---")
    
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"ERROR: File not found at {model_path}\n")
        return

    try:
        # Instantiate the reference model with the correct architecture
        model = SameDifferentCNN()
        
        # Load the state dict from the provided file
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))

        # PyTorch sometimes saves the model inside an 'model_state_dict' key
        # or as a 'model' key when using DataParallel. Handle these cases.
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
            
        # Remove 'module.' prefix if it exists (from DataParallel)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict

        # Attempt to load the state dictionary into our reference model
        # We use strict=False to see all potential mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print("Architecture Check Results:")
        
        # The old, incorrect architecture had a 'temperature' parameter.
        # Its presence is a strong indicator of a mismatch.
        if 'temperature' in unexpected_keys:
            print("❌ Mismatch! Found 'temperature' key, indicating this is the OLD architecture.")
            unexpected_keys.remove('temperature')
        
        if not missing_keys and not unexpected_keys:
            print("✅ Success! The model architecture matches the 'new' conv6 specification.")
        else:
            # If we already identified the old arch, we don't need another generic error.
            if 'temperature' not in unexpected_keys:
                 print("❌ Mismatch! The model architecture does NOT match the 'new' conv6 specification.")

            if missing_keys:
                print("\nMissing keys in state_dict (model expects these but file doesn't have them):")
                for key in missing_keys:
                    print(f"- {key}")
            if unexpected_keys:
                print("\nUnexpected keys in state_dict (file has these but model doesn't expect them):")
                for key in unexpected_keys:
                    print(f"- {key}")

        # For direct verification, let's print the shape of the first conv layer from the file
        if 'conv1.weight' in state_dict:
            conv1_shape = state_dict['conv1.weight'].shape
            print(f"\nShape of 'conv1.weight' from file: {conv1_shape}")
            if conv1_shape == torch.Size([18, 3, 6, 6]):
                print("This matches the expected conv1 shape for the 'new' architecture.")
            else:
                print("This does NOT match the expected conv1 shape (which is [18, 3, 6, 6]).")
        else:
            print("\nCould not find 'conv1.weight' in the state dictionary.")

    except Exception as e:
        print(f"An error occurred while analyzing the model: {e}")
    
    print("-" * (len(model_path) + 14) + "\n")

if __name__ == "__main__":
    # Define the paths to your model files
    model_path_1 = 'output_files/conv6lr_best (2).pth'
    model_path_2 = 'output_files/conv6lr_best (3).pth'
    
    # Run the analysis
    analyze_model_weights(model_path_1)
    analyze_model_weights(model_path_2) 