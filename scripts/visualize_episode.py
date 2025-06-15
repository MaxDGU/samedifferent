import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_episode(h5_file_path: str, episode_index: int, output_image_path: str):
    """
    Loads a specific episode from an HDF5 file and visualizes its support and query sets.

    Args:
        h5_file_path (str): Path to the HDF5 file.
        episode_index (int): Index of the episode to visualize.
        output_image_path (str): Path to save the generated figure.
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if not ('support_images' in f and 
                    'support_labels' in f and 
                    'query_images' in f and 
                    'query_labels' in f):
                print(f"Error: Dataset keys (support_images, support_labels, query_images, query_labels) not found in {h5_file_path}")
                return

            num_episodes = f['support_images'].shape[0]
            if episode_index >= num_episodes:
                print(f"Error: Episode index {episode_index} is out of bounds. File has {num_episodes} episodes.")
                return

            support_images = f['support_images'][episode_index]
            support_labels = f['support_labels'][episode_index]
            query_images = f['query_images'][episode_index]
            query_labels = f['query_labels'][episode_index]

    except Exception as e:
        print(f"Error loading HDF5 file {h5_file_path}: {e}")
        return

    # Add diagnostic prints for shapes
    print(f"--- Data shapes for episode {episode_index} from {Path(h5_file_path).name} ---")
    print(f"Support images shape: {support_images.shape}")
    print(f"Support labels shape: {support_labels.shape}")
    print(f"Query images shape: {query_images.shape}")
    print(f"Query labels shape: {query_labels.shape}")

    num_support = support_images.shape[0]
    num_query = query_images.shape[0]

    print(f"Number of support samples found: {num_support}")
    print(f"Number of query samples found: {num_query}")
    print("-----------------------------------------------------")

    # Determine layout: try to fit support and query on separate rows if possible
    # Max images per row for comfortable viewing
    max_img_per_row = max(num_support, num_query, 4) # at least 4, or more if sets are larger

    fig_height = 6 # default height
    if num_support > 0 and num_query > 0:
        num_rows_plot = 2
        fig_height = 8
    elif num_support > 0 or num_query > 0:
        num_rows_plot = 1
    else:
        print("Error: No images in support or query set.")
        return
        
    fig, axes = plt.subplots(num_rows_plot, max_img_per_row, figsize=(max_img_per_row * 3, fig_height))
    fig.suptitle(f"Episode {episode_index} from {Path(h5_file_path).name}", fontsize=16)

    # Flatten axes array for easier iteration if it's 2D
    if num_rows_plot > 1 :
        ax_flat = axes.flatten()
    else:
        # if only one row, axes might not be a 2d array if max_img_per_row is 1.
        ax_flat = np.array(axes).flatten()


    current_ax_idx = 0

    # Plot Support Set
    if num_support > 0:
        if num_rows_plot > 1: # If two rows, support is on first row
            fig.text(0.5, 0.94, 'Support Set', ha='center', va='center', fontsize=14, weight='bold')
        
        for i in range(num_support):
            ax = ax_flat[current_ax_idx]
            ax.imshow(support_images[i])
            ax.set_title(f"Label: {support_labels[i]}", fontsize=10)
            ax.axis('off')
            current_ax_idx += 1
    
    # Fill remaining axes in the first row (if any and if two rows are plotted)
    if num_rows_plot > 1:
        while current_ax_idx < max_img_per_row:
            ax_flat[current_ax_idx].axis('off')
            current_ax_idx +=1

    # Plot Query Set
    if num_query > 0:
        # If only one row total, current_ax_idx continues. If two, it starts at max_img_per_row
        if num_rows_plot == 1 and num_support > 0 : # If single row, add a small spacer visually if possible
             # This logic is tricky if support and query are on the same line.
             # For now, just continue plotting. A textual separator might be better.
             pass
        elif num_rows_plot > 1: # If two rows, query starts on the second row
            fig.text(0.5, 0.48, 'Query Set', ha='center', va='center', fontsize=14, weight='bold')
            # current_ax_idx is already at max_img_per_row due to previous fill

        for i in range(num_query):
            if current_ax_idx >= len(ax_flat):
                print(f"Warning: Not enough subplot axes to display all query images. Displaying first {i} images.")
                break
            ax = ax_flat[current_ax_idx]
            ax.imshow(query_images[i])
            ax.set_title(f"Label: {query_labels[i]}", fontsize=10)
            ax.axis('off')
            current_ax_idx += 1

    # Turn off any remaining unused axes
    while current_ax_idx < len(ax_flat):
        ax_flat[current_ax_idx].axis('off')
        current_ax_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust rect to make space for suptitle and section titles

    # Ensure output directory exists
    output_path_obj = Path(output_image_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        plt.savefig(output_image_path)
        print(f"Episode visualization saved to {output_image_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize an episode from an HDF5 file.")
    parser.add_argument("h5_file_path", type=str, help="Path to the HDF5 file (e.g., data/meta_h5/pb/open_support4_train.h5).")
    parser.add_argument("output_image_path", type=str, help="Path to save the output image (e.g., visualizations/episode_visualization.png).")
    parser.add_argument("--episode_index", type=int, default=0, help="Index of the episode to visualize (default: 0).")
    
    args = parser.parse_args()
    
    visualize_episode(args.h5_file_path, args.episode_index, args.output_image_path) 