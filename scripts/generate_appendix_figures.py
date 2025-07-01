import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec

def generate_episode_visualization(task_name, support_size, episode_idx=0, data_dir='data/meta_h5/pb', output_dir='figures'):
    """
    Generates and saves a visualization of a specific episode from an HDF5 file,
    matching the style of 'open_support4_ep0_boxed.png'.

    Args:
        task_name (str): The name of the task (e.g., 'wider_line').
        support_size (int): The number of support examples (e.g., 6).
        episode_idx (int): The index of the episode to visualize.
        data_dir (str): The directory where the HDF5 files are located.
        output_dir (str): The directory to save the output visualizations.
    """
    h5_path = os.path.join(data_dir, f'{task_name}_support{support_size}_train.h5')
    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found at {h5_path}")
        # Try test file as a fallback
        h5_path = os.path.join(data_dir, f'{task_name}_support{support_size}_test.h5')
        if not os.path.exists(h5_path):
            print(f"Error: Also not found in test set: {h5_path}")
            return

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        support_images = f['support_images'][episode_idx]
        query_images = f['query_images'][episode_idx]
        support_labels = f['support_labels'][episode_idx]
        query_labels = f['query_labels'][episode_idx]

    n_support = len(support_images)
    n_query = len(query_images)
    
    fig = plt.figure(figsize=(n_support * 2, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.4)

    # --- Support Set ---
    gs_support = gridspec.GridSpecFromSubplotSpec(1, n_support, subplot_spec=gs[0], wspace=0.1)
    ax_support_title = fig.add_subplot(gs[0])
    ax_support_title.set_title('Support Set', fontsize=16, weight='bold', pad=20)
    ax_support_title.axis('off')

    for i in range(n_support):
        ax = fig.add_subplot(gs_support[i])
        ax.imshow(support_images[i])
        ax.set_title(f'Label: {support_labels[i]}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # --- Query Set ---
    gs_query = gridspec.GridSpecFromSubplotSpec(1, n_query, subplot_spec=gs[1], wspace=0.1)
    ax_query_title = fig.add_subplot(gs[1])
    ax_query_title.set_title('Query Set', fontsize=16, weight='bold', pad=20)
    ax_query_title.axis('off')

    for i in range(n_query):
        ax = fig.add_subplot(gs_query[i])
        ax.imshow(query_images[i])
        ax.set_title(f'Label: {query_labels[i]}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Save the figure
    output_filename = f'{task_name}_support{support_size}_ep{episode_idx}_boxed.png'
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

if __name__ == '__main__':
    # Generate the requested figures
    generate_episode_visualization(task_name='wider_line', support_size=6, episode_idx=0)
    generate_episode_visualization(task_name='scrambled', support_size=8, episode_idx=0)
    generate_episode_visualization(task_name='filled', support_size=10, episode_idx=0)
    # Also generate the 'open' task one to ensure it's in the figures directory
    generate_episode_visualization(task_name='open', support_size=4, episode_idx=0)
    print("All requested appendix figures have been regenerated.") 