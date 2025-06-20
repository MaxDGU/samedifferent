import torch
from pathlib import Path
import argparse
import sys

# --- Setup Project Path ---
# This allows the script to be run from the root of the project
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def has_6_conv_layers(file_path: Path) -> bool:
    """
    Checks if a PyTorch checkpoint file contains weights for 6 convolutional layers.
    Returns True if 'conv1.weight' through 'conv6.weight' are all present.
    """
    try:
        # Load the checkpoint to the CPU to avoid GPU memory issues
        checkpoint = torch.load(file_path, map_location='cpu')

        # The state_dict might be nested inside the checkpoint dictionary
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            return False

        # Define the set of keys we absolutely require for it to be a 6-layer model
        required_keys = {f'conv{i}.weight' for i in range(1, 7)}

        return required_keys.issubset(state_dict.keys())

    except Exception as e:
        # If there's any error loading or reading the file, we assume it's not the model we want.
        # This handles corrupted files, non-pytorch files, etc.
        # print(f"  -> Could not process {file_path.name}: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Recursively search a directory for PyTorch models that have a 6-layer convolutional architecture."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The path to the directory to search."
    )
    args = parser.parse_args()

    target_dir = Path(args.directory)
    if not target_dir.is_dir():
        print(f"Error: Directory not found at '{target_dir}'")
        sys.exit(1)

    print(f"--- Searching for 6-layer CNN models in: {target_dir} ---")
    
    found_files_count = 0
    # Use rglob to recursively find all .pt and .pth files.
    # The glob pattern '*.pt*' handles both extensions.
    for file_path in sorted(target_dir.rglob('*.pt*')):
        if file_path.is_file() and has_6_conv_layers(file_path):
            print(str(file_path))
            found_files_count += 1
            
    if found_files_count == 0:
        print("\n--- No 6-layer models found. ---")
    else:
        print(f"\n--- Found {found_files_count} total model(s) with 6 convolutional layers. ---")

if __name__ == "__main__":
    main() 