import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_episode(h5_path, episode_idx=0):
    """
    Visualize a single episode from an h5 file.
    
    Args:
        h5_path: Path to the h5 file
        episode_idx: Index of the episode to visualize (default: 0)
    """
    with h5py.File(h5_path, 'r') as f:
        # Load support and query images/labels
        support_images = f['support_images'][episode_idx]
        support_labels = f['support_labels'][episode_idx]
        query_images = f['query_images'][episode_idx]
        query_labels = f['query_labels'][episode_idx]
        
        # Print shapes and labels for debugging
        print(f"Support images shape: {support_images.shape}")
        print(f"Support labels: {support_labels}")
        print(f"Query images shape: {query_images.shape}")
        print(f"Query labels: {query_labels}")
        
        # Calculate total number of images and determine grid size
        n_support = len(support_images)
        n_query = len(query_images)
        total_images = n_support + n_query
        
        # Calculate number of rows and columns for the grid
        n_cols = min(4, total_images)  # Maximum 4 images per row
        n_rows = (total_images + n_cols - 1) // n_cols
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        
        # Plot support set
        for i in range(n_support):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            ax.imshow(support_images[i])
            ax.axis('off')
            ax.set_title(f'Support {i+1}\nLabel: {support_labels[i]}')
        
        # Plot query set
        for i in range(n_query):
            ax = plt.subplot(n_rows, n_cols, n_support + i + 1)
            ax.imshow(query_images[i])
            ax.axis('off')
            ax.set_title(f'Query {i+1}\nLabel: {query_labels[i]}')
        
        plt.suptitle(f'Episode {episode_idx} from {os.path.basename(h5_path)}', y=1.02)
        plt.tight_layout()
        
        # Save the figure
        save_path = f'episode_{episode_idx}_visualization.png'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to {save_path}")
        plt.close()

def main():
    # Example usage for problem 15, support set size 4
    h5_path = 'data//results_problem_15/support4_train.h5'
    
    # Visualize first episode
    visualize_episode(h5_path, episode_idx=0)
    
    # You can also visualize other episodes by changing the episode_idx
    # For example:
    # visualize_episode(h5_path, episode_idx=1)
    # visualize_episode(h5_path, episode_idx=2)

if __name__ == '__main__':
    main() 