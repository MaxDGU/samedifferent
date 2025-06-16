import h5py
import argparse
from pathlib import Path

def inspect_h5(file_path):
    """
    Recursively prints the structure of an HDF5 file.
    """
    def print_structure(name, obj):
        indent = '  ' * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}- {name} (Dataset: shape={obj.shape}, dtype={obj.dtype})")
        else: # It's a group
            print(f"{indent}{name} (Group)")

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"Inspecting file: {file_path}\n")
    with h5py.File(file_path, 'r') as hf:
        if not hf:
            print("File is empty or not a valid HDF5 file.")
            return
        hf.visititems(print_structure)
    print("\nInspection complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect the internal structure of an HDF5 file.")
    parser.add_argument('file_path', type=Path, help="The path to the HDF5 file to inspect.")
    args = parser.parse_args()
    inspect_h5(args.file_path) 