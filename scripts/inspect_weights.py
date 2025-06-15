import torch
import argparse
from pathlib import Path

def inspect_weights(file_path: Path):
    """
    Loads a PyTorch model checkpoint and prints the architecture blueprint.
    This includes all layer names (keys) and their corresponding tensor shapes.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Inspecting weights file: {file_path.name} ---")
    
    try:
        # Load the checkpoint to the CPU
        # weights_only=False is necessary if the file contains non-tensor data
        # like optimizer states or args, which is common.
        state_dict = torch.load(file_path, map_location='cpu')

        # Check if the weights are nested in a common dictionary key
        if 'model_state_dict' in state_dict:
            print("Found weights under key: 'model_state_dict'")
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            print("Found weights under key: 'state_dict'")
            state_dict = state_dict['state_dict']
        
        if not state_dict:
            print("Error: The state dictionary is empty.")
            return

        print("\n--- Model Architecture Blueprint ---")
        max_key_len = max(len(k) for k in state_dict.keys())
        for key, value in state_dict.items():
            shape = value.shape
            print(f"{key:<{max_key_len}} | Shape: {shape}")
        
        print("\n--- End of Blueprint ---")

    except Exception as e:
        print(f"\nAn error occurred while reading the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch .pt or .pth file to see its architecture."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="The path to the model weight file (.pt or .pth)."
    )
    args = parser.parse_args()
    
    inspect_weights(Path(args.file_path)) 