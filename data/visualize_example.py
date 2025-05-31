import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def create_example_visualization(image_path, save_path='same_different_example.png'):
    """Create a visual explanation of a single same/different image."""
    # Load and display the image
    img = Image.open(image_path)
    
    # Set up the figure
    plt.figure(figsize=(8, 8))
    
    # Display the image
    plt.imshow(np.array(img))
    plt.axis('off')
    
    # Add title and explanation
    plt.title('SVRT Same/Different Task #1', 
             pad=20, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Path to example image
    image_path = "data/meta_h5/pb/original_support4_train_sample.png"
    
    create_example_visualization(image_path)
    print("Visualization saved as same_different_example.png")

if __name__ == "__main__":
    main() 