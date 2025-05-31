import os
import h5py
from pathlib import Path

# --- Configuration ---
# Base directory where the HDF5 files are located
# Assumes the script is run from the project root, so data/meta_h5/pb is relative to that
DEFAULT_DATA_DIR = Path("data/meta_h5/pb")

PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
            'random_color', 'arrows', 'irregular', 'filled', 'original']
SUPPORT_SIZES = [4, 6, 8, 10] # Support sizes used by PBDataset for single-task
SPLIT = 'train' # We are interested in the training split

def count_images_in_hdf5_file(filepath):
    """
    Counts the total number of images (support + query from all episodes) in a single HDF5 file.
    """
    total_images_in_file = 0
    try:
        with h5py.File(filepath, 'r') as f:
            if 'support_images' not in f or 'query_images' not in f:
                print(f"    File {filepath.name} is missing 'support_images' or 'query_images' keys. Skipping.")
                return 0

            num_episodes_support = f['support_images'].shape[0]
            s_shape = f['support_images'].shape
            q_shape = f['query_images'].shape

            if len(s_shape) < 2 or len(q_shape) < 2:
                print(f"    File {filepath.name} has unexpected image shapes: S={s_shape}, Q={q_shape}. Skipping.")
                return 0

            s_count = s_shape[1] # Number of support images per episode
            q_count = q_shape[1] # Number of query images per episode
            
            # Ensure number of episodes match for support and query if both exist and are non-empty
            if 'query_images' in f and f['query_images'].shape[0] != num_episodes_support:
                 print(f"    Warning: Mismatch in episode count between support ({num_episodes_support}) and query ({f['query_images'].shape[0]}) in {filepath.name}. Using support episode count.")

            total_images_in_file = num_episodes_support * (s_count + q_count)
            # print(f"    File {filepath.name}: {num_episodes_support} episodes * ({s_count}S + {q_count}Q) = {total_images_in_file} images")
    except Exception as e:
        print(f"    Error reading file {filepath.name}: {e}")
        return 0
    return total_images_in_file

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Count total training images for single-task experiments from HDF5 files.")
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help=f"Directory containing the HDF5 task files (e.g., {DEFAULT_DATA_DIR})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Error: Data directory '{data_dir}' not found or is not a directory.")
        return

    print(f"Counting single-task training images from: {data_dir}")
    print(f"Tasks: {PB_TASKS}")
    print(f"Support Sizes considered for pooling: {SUPPORT_SIZES}")
    print(f"Split: {SPLIT}\n")

    grand_total_images = 0
    task_totals = {}

    for task_name in PB_TASKS:
        print(f"Processing task: {task_name}...")
        current_task_total_images = 0
        for s_size in SUPPORT_SIZES:
            filename = f"{task_name}_support{s_size}_{SPLIT}.h5"
            filepath = data_dir / filename

            if filepath.exists():
                # print(f"  Checking file: {filepath.name}")
                images_in_file = count_images_in_hdf5_file(filepath)
                current_task_total_images += images_in_file
            else:
                print(f"  File not found: {filepath.name}. Skipping.")
        
        task_totals[task_name] = current_task_total_images
        print(f"  Total training images for task '{task_name}': {current_task_total_images}\n")
        grand_total_images += current_task_total_images

    print("\n--- Summary ---")
    for task_name, total in task_totals.items():
        print(f"Task '{task_name}': {total} training images")
    print("--------------------")
    print(f"Grand Total training images (sum across all tasks for '{SPLIT}' split, pooled from S={SUPPORT_SIZES}): {grand_total_images}")
    print("\nNote: This script counts images by summing (num_episodes * (S+Q)) for each relevant HDF5 file,")
    print("replicating how PBDataset in train_single_task_pb.py pools data for single-task training.")

if __name__ == '__main__':
    main() 