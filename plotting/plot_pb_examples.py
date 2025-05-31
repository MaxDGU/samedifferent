import matplotlib.pyplot as plt
import numpy as np
from generate_datasets import (
    irregular_gen, regular_gen, open_gen, wider_line_gen, 
    scrambled_gen, random_color_gen, filled_gen, lines_gen, 
    arrows_gen
)

def create_pb_examples_grid():
    # Define tasks and their corresponding generators
    task_generators = {
        'irregular': irregular_gen,
        'regular': regular_gen,
        'open': open_gen,
        'wider_line': wider_line_gen,
        'scrambled': scrambled_gen,
        'random_color': random_color_gen,
        'filled': filled_gen,
        'lines': lines_gen,
        'arrows': arrows_gen
    }
    
    # Create figure with appropriate size and proper spacing
    fig = plt.figure(figsize=(20, 4))
    
    # Create a grid of subplots with space for row labels
    gs = plt.GridSpec(2, len(task_generators) + 1, width_ratios=[0.2] + [1]*len(task_generators))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Add row labels in the first column
    ax_same = fig.add_subplot(gs[0, 0])
    ax_same.text(1, 0.5, 'Same', rotation=0, ha='right', va='center', fontsize=16)
    ax_same.axis('off')
    
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_diff.text(1, 0.5, 'Different', rotation=0, ha='right', va='center', fontsize=16)
    ax_diff.axis('off')
    
    # Iterate through tasks
    for col, (task, generator) in enumerate(task_generators.items()):
        # Generate a same example (label=1)
        same_batch = next(generator(batch_size=1, category_type='same'))
        same_img = same_batch[0][0]  # Get first image from batch
        
        # Generate a different example (label=0)
        diff_batch = next(generator(batch_size=1, category_type='different'))
        diff_img = diff_batch[0][0]  # Get first image from batch
        
        # Create subplots with boxes around them
        ax_same = fig.add_subplot(gs[0, col + 1])
        ax_diff = fig.add_subplot(gs[1, col + 1])
        
        # Plot images
        ax_same.imshow(same_img, cmap='gray')
        ax_diff.imshow(diff_img, cmap='gray')
        
        # Set title for each column (task)
        ax_same.set_title(task.replace('_', ' ').title(), pad=5, fontsize=16)
        
        # Add boxes around the plots
        for ax in [ax_same, ax_diff]:
            ax.set_xticks([])
            ax.set_yticks([])
            # Make the box visible
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('pb_examples_grid.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print("Grid saved as pb_examples_grid.png")

if __name__ == '__main__':
    create_pb_examples_grid() 