import os
import h5py
import numpy as np
import glob
from collections import defaultdict

def inspect_test_data_distribution(data_dir):
    """
    Inspects the label distribution of the test sets in HDF5 files.

    Args:
        data_dir (str): The directory containing the HDF5 files.
    """
    print(f"--- Inspecting Test Data in: {data_dir} ---\n")

    test_files = glob.glob(os.path.join(data_dir, '*_test.h5'))

    if not test_files:
        print("No test HDF5 files found in the specified directory.")
        return

    task_label_counts = defaultdict(lambda: defaultdict(int))
    total_labels_all_tasks = 0

    for file_path in sorted(test_files):
        try:
            task_name = os.path.basename(file_path).split('_support')[0]
            
            with h5py.File(file_path, 'r') as f:
                if 'support_labels' not in f or 'query_labels' not in f:
                    print(f"WARNING: Skipping {file_path} - missing 'support_labels' or 'query_labels'.")
                    continue
                
                support_labels = f['support_labels'][:]
                query_labels = f['query_labels'][:]
                
                combined_labels = np.concatenate([support_labels.flatten(), query_labels.flatten()])
                
                if combined_labels.size == 0:
                    print(f"WARNING: Task {task_name} from file {os.path.basename(file_path)} has NO labels.")
                    continue

                unique, counts = np.unique(combined_labels, return_counts=True)
                
                for label, count in zip(unique, counts):
                    task_label_counts[task_name][int(label)] += count
                
                total_labels_all_tasks += combined_labels.size

        except Exception as e:
            print(f"ERROR: Could not process file {file_path}: {e}")

    print("--- Test Set: Per-Task Label Distribution Check ---")
    if not task_label_counts:
        print("No labels were counted across any tasks.")
        return
        
    for task_name, counts in sorted(task_label_counts.items()):
        total_labels_for_task = sum(counts.values())
        
        label_dist_str = ", ".join([f"Label {l}: {c}" for l, c in sorted(counts.items())])
        
        print(f"  Task {task_name} (test): Total Labels={total_labels_for_task}, Distribution: {label_dist_str}")
        if len(counts) < 2:
            print(f"    WARNING: Task {task_name} (test) has only one class label (or zero labels).")

    print(f"\nTotal labels processed across all test tasks: {total_labels_all_tasks}")
    print("--- End Label Distribution Check ---")


if __name__ == '__main__':
    # The user specified this local path.
    local_data_path = '/Users/maxgupta/Desktop/Princeton/CoCoSci_Lab/samedifferent/same_different_paper/metasamedifferent/data/meta_h5/pb'
    if os.path.exists(local_data_path):
        inspect_test_data_distribution(local_data_path)
    else:
        print(f"ERROR: Data path not found: {local_data_path}") 