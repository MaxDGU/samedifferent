import matplotlib.pyplot as plt
import h5py
import numpy as np
from PIL import Image
import os

def create_task_visualization(support_images, query_images, support_labels, query_labels, save_path='task_visualization.png'):
    """Create a visual explanation of the same/different task."""
    # Set up the figure
    plt.figure(figsize=(15, 8))
    
    # Plot support set
    plt.subplot(2, 1, 1)
    plt.title('Support Set (Training Examples)', pad=20)
    
    for i in range(len(support_images)):
        plt.subplot(2, len(support_images), i + 1)
        plt.imshow(support_images[i])
        plt.axis('off')
        plt.title(f'{"Same" if support_labels[i] == 1 else "Different"}')
    
    # Plot query set
    for i in range(len(query_images)):
        plt.subplot(2, len(query_images), len(support_images) + i + 1)
        plt.imshow(query_images[i])
        plt.axis('off')
        plt.title(f'Query {i+1}\n{"Same" if query_labels[i] == 1 else "Different"}')
    
    plt.suptitle('Same/Different Task Example\nModel must learn to classify if two shapes in each image are the same or different', 
                 y=1.05, fontsize=14)
    
    # Add explanatory text
    plt.figtext(0.02, 0.5, 'The model is shown support examples with labels (top row)\n' + 
                'and must predict if new query images (bottom row) show same or different shapes',
                fontsize=10, ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load sample data
    data_path = "data/pb/pb/original_support4_test.h5"
    
    with h5py.File(data_path, 'r') as f:
        # Get first episode
        support_images = f['support_images'][0]
        query_images = f['query_images'][0]
        support_labels = f['support_labels'][0]
        query_labels = f['query_labels'][0]
    
    create_task_visualization(support_images, query_images, support_labels, query_labels)
    print("Visualization saved as task_visualization.png")

if __name__ == "__main__":
    main() 