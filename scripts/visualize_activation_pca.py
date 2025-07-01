import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import learn2learn as l2l
from tqdm import tqdm
import os
import numpy as np
import argparse
import json
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the dataset and model definition from our other script
from scripts.find_and_ablate_circuit import SameDifferentDataset, custom_collate
from meta_baseline.models.conv6lr import SameDifferentCNN

# --- Globals for Activation Hook ---
activation_store = {}

def get_activation(name):
    """Hook to store the output of a layer."""
    def hook(model, input, output):
        activation_store[name] = output.detach()
    return hook

def main(args):
    """Main function to generate and plot activation PCA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Step 1: Load the pre-trained baseline model
    print("Loading pre-trained model...")
    model = SameDifferentCNN(num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=False, allow_unused=True)
    maml.eval()

    # Step 2: Register the forward hook on the target layer
    target_layer = maml.module.layer4[1] # Final conv layer
    target_layer.register_forward_hook(get_activation('final_conv'))
    
    # Step 3: Set up the DataLoader
    print("Setting up data loader...")
    PB_TASKS = [
        "trignometric", "spatial", "wider_line", "scrambled", "open",
        "filled", "rounded", "simple", "arrows", "lines"
    ]
    val_dataset = SameDifferentDataset(args.pb_data_dir, PB_TASKS, 'val', support_sizes=args.val_support_size)
    # Important: Use batch_size=1 and no custom collate for this analysis
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Step 4: Loop through data to collect activations
    print("Collecting activations...")
    all_activations = []
    all_labels = []
    
    for i, episode in enumerate(tqdm(val_loader, desc="Processing Tasks")):
        learner = maml.clone()

        support_images = episode['support_images'][0].to(device)
        support_labels = episode['support_labels'][0].to(device)
        query_images = episode['query_images'][0].to(device)
        
        # Get pre-adaptation activations
        _ = learner(query_images)
        pre_activ = activation_store['final_conv'].mean(dim=[2, 3]).cpu().numpy().flatten()

        # Adapt
        for _ in range(args.adaptation_steps):
            preds = learner(support_images)
            loss = F.cross_entropy(preds, support_labels)
            learner.adapt(loss, allow_unused=True)

        # Get post-adaptation activations
        _ = learner(query_images)
        post_activ = activation_store['final_conv'].mean(dim=[2, 3]).cpu().numpy().flatten()
        
        # Store results
        task_name = os.path.basename(episode['file_path'][0]).split('_')[0]
        all_activations.extend([pre_activ, post_activ])
        all_labels.append({'task': task_name, 'state': 'pre-adaptation'})
        all_labels.append({'task': task_name, 'state': 'post-adaptation'})

        if i >= 49: # Limit to 50 tasks for a cleaner plot
            break

    # Step 5: Perform PCA
    print("Performing PCA...")
    all_activations = np.array(all_activations)
    pca = PCA(n_components=2)
    components = pca.fit_transform(all_activations)
    
    # Step 6: Plot the results
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 12))
    
    tasks = np.array([label['task'] for label in all_labels])
    states = np.array([label['state'] for label in all_labels])
    
    unique_tasks = np.unique(tasks)
    colors = plt.cm.get_cmap('tab10', len(unique_tasks))
    task_to_color = {task: color for task, color in zip(unique_tasks, colors)}

    # Plot scatter points
    for i in range(len(components)):
        plt.scatter(
            components[i, 0], components[i, 1],
            color=task_to_color[tasks[i]],
            marker='o' if states[i] == 'pre-adaptation' else 'x',
            s=100,
            alpha=0.8,
            label=f"{tasks[i]} ({states[i]})" if i < 2*len(unique_tasks) else "" # Avoid duplicate labels
        )

    # Plot arrows
    for i in range(0, len(components), 2):
        pre_point = components[i]
        post_point = components[i+1]
        plt.annotate(
            '', xy=post_point, xytext=pre_point,
            arrowprops=dict(arrowstyle='->', color=task_to_color[tasks[i]], lw=1.5)
        )

    plt.title('PCA of Final Conv Layer Activations (Before vs. After Adaptation)', fontsize=20, pad=20)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    
    # Create a clean legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=task) for task, color in task_to_color.items()]
    legend_elements.append(Line2D([0], [0], marker='o', color='gray', label='Pre-Adaptation', markersize=10, linestyle='None'))
    legend_elements.append(Line2D([0], [0], marker='x', color='gray', label='Post-Adaptation', markersize=10, linestyle='None'))
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(args.output_path, dpi=300)
    print(f"PCA plot saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize model activations using PCA.")
    
    # --- Directory and Path Arguments ---
    parser.add_argument('--model_path', type=str, default='circuit_analysis_results/baseline_model.pt',
                        help="Path to the trained baseline model.")
    parser.add_argument('--pb_data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb',
                        help="Path to the meta-h5/pb data directory.")
    parser.add_argument('--output_path', type=str, default='visualizations/activation_pca.png',
                        help="Path to save the output PCA plot.")

    # --- MAML/Data Arguments ---
    parser.add_argument('--batch_size', type=int, default=1, help="Meta-batch size (use 1 for this analysis).")
    parser.add_argument('--val_support_size', type=int, nargs='+', default=[10], 
                        help="Support set size for validation.")
    parser.add_argument('--fast_lr', type=float, default=0.01, help="Learning rate for inner loop adaptation.")
    parser.add_argument('--adaptation_steps', type=int, default=5, help="Number of adaptation steps.")

    args = parser.parse_args()
    main(args) 