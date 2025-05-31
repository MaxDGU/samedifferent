import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    # Read the compiled results
    with open('results/leave_one_out_seeds/compiled_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract task results
    task_data = []
    for task, data in results['individual_task_results'].items():
        task_data.append({
            'Task': task,
            'Accuracy': data['mean_accuracy'] * 100,  # Convert to percentage
            'Std': data['std_accuracy'] * 100,  # Convert to percentage
        })
    
    # Create DataFrame
    df = pd.DataFrame(task_data)
    
    # Set the style
    sns.set_theme(style="ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create figure with larger size
    plt.figure(figsize=(12, 6))
    
    # Create bar plot with viridis palette
    colors = sns.color_palette("viridis", n_colors=len(df))
    ax = sns.barplot(
        data=df,
        x='Task',
        y='Accuracy',
        palette=colors
    )
    
    # Add error bars manually
    for i, bar in enumerate(ax.patches):
        x = bar.get_x() + bar.get_width()/2
        height = bar.get_height()
        std = df['Std'].iloc[i]
        ax.vlines(x, height - std, height + std, color='gray', alpha=0.8, linewidth=1)
        ax.hlines(height + std, x - 0.05, x + 0.05, color='gray', alpha=0.8, linewidth=1)
        ax.hlines(height - std, x - 0.05, x + 0.05, color='gray', alpha=0.8, linewidth=1)
    
    # Customize the plot
    plt.title('Mean Accuracy by Task', pad=20, fontsize=14)
    plt.xlabel('Task', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + df['Std'].iloc[i] + 1,  # Place text above error bars
            f'{df["Accuracy"].iloc[i]:.1f}%',
            ha='center',
            va='bottom'
        )
    
    # Adjust layout to prevent label cutoff
    plt.ylim(0, 105)  # Set y-axis limit to accommodate labels
    
    # Add a light grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/leave_one_out_seeds/accuracy_by_task.png', 
                dpi=300, 
                bbox_inches='tight')
    print("Plot saved as accuracy_by_task.png")

if __name__ == '__main__':
    main() 