import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Add the root directory to the path to allow imports from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_baseline.models import Conv2CNN as SameDifferentCNN
from baselines.models.utils import PB_TASKS
from circuit_analysis.analyzer import CircuitAnalyzer
from meta_baseline.models.utils_meta import SameDifferentDataset, collate_episodes  # dataset and collate compatible with each other

def calculate_accuracy(model, data_loader, device, analyzer=None, layer_to_ablate=None, channel_to_ablate=None):
    """Calculates the model's accuracy on a given dataset."""
    model.eval()
    correct = 0
    total = 0
    
    if analyzer and layer_to_ablate and channel_to_ablate is not None:
        analyzer.ablate_layer(layer_to_ablate, channel_to_ablate)

    with torch.no_grad():
        for batch in data_loader:
            # Handle both flattened-image batches and episode-based batches
            if 'image' in batch and 'label' in batch:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
            elif 'query_images' in batch and 'query_labels' in batch:
                # Episode batch: flatten query set across episodes
                q_imgs = batch['query_images']  # [B, M, C, H, W]
                q_lbls = batch['query_labels']  # [B, M]

                B, M, C, H, W = q_imgs.shape
                images = q_imgs.view(B * M, C, H, W).to(device)
                labels = q_lbls.view(-1).to(device)
            else:
                raise KeyError("Batch must contain either ('image','label') or ('query_images','query_labels') keys.")
            
            if analyzer:
                outputs, _ = analyzer.get_activations(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    if analyzer:
        analyzer.remove_hooks()
        
    return correct / total

def main():
    """Main function to run the ablation experiment."""
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model ---
    # Pretrained PB meta-baseline conv2 model trained on full PB range
    model_path = '/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv2/seed_46/best_model.pt'
    model = SameDifferentCNN()
    
    # When loading a model saved with nn.DataParallel, the state dict keys are prefixed with 'module.'
    # We need to handle this to load the model correctly.
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    if next(iter(state_dict)).startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    print(f"Loaded trained model from {model_path}")

    # --- 3. Load Data ---
    data_dir = 'data/meta_h5/pb'
    # Use the full PB dataset: all 10 tasks and all standard support sizes
    tasks = PB_TASKS  # ['regular', 'lines', 'open', 'wider_line', 'scrambled', 'random_color', 'arrows', 'irregular', 'filled', 'original']
    test_dataset = SameDifferentDataset(data_dir, tasks, 'test', support_sizes=[4, 6, 8, 10])
    # Use smaller batch size to manage memory with 8K episodes
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_episodes)
    print(f"Loaded full PB test dataset with {len(tasks)} tasks and support sizes [4,6,8,10].")

    # --- 4. Initialize Analyzer ---
    analyzer = CircuitAnalyzer(model)

    # --- 5. Run Ablation Experiment ---
    layer_to_ablate = 'conv2'
    layer = analyzer._layer_map[layer_to_ablate]
    num_channels = layer.out_channels
    
    print(f"\n--- Starting Ablation Experiment on '{layer_to_ablate}' ---")
    
    # First, get baseline accuracy without any ablation
    baseline_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Baseline accuracy on full PB test set: {baseline_accuracy:.4f}")

    # Now, ablate each channel and measure the accuracy drop
    print(f"Will test {num_channels} channels in {layer_to_ablate}")
    accuracy_drops = {}
    for channel_idx in tqdm(range(num_channels), desc=f"Ablating channels in {layer_to_ablate}"):
        ablated_accuracy = calculate_accuracy(model, test_loader, device, analyzer, layer_to_ablate, channel_idx)
        accuracy_drops[channel_idx] = baseline_accuracy - ablated_accuracy
        
        # Log progress and save intermediate results every 50 channels
        if (channel_idx + 1) % 50 == 0:
            print(f"Completed {channel_idx + 1}/{num_channels} channels. Current drop: {accuracy_drops[channel_idx]:.4f}")
            
            # Save intermediate results
            intermediate_results = {
                'baseline_accuracy': baseline_accuracy,
                'accuracy_drops': accuracy_drops,
                'completed_channels': channel_idx + 1,
                'total_channels': num_channels,
                'layer': layer_to_ablate
            }
            with open(f'intermediate_ablation_results_{channel_idx + 1}.json', 'w') as f:
                json.dump(intermediate_results, f, indent=2)

    # --- 6. Report Results ---
    print("\n--- Ablation Results ---")
    # Sort channels by the magnitude of accuracy drop
    sorted_channels = sorted(accuracy_drops.items(), key=lambda item: item[1], reverse=True)

    print(f"Top 5 most critical channels in '{layer_to_ablate}' for task 'full PB':")
    for i in range(min(5, len(sorted_channels))):
        channel_idx, drop = sorted_channels[i]
        print(f"  - Channel {channel_idx}: Accuracy drop of {drop:.4f}")

    # --- 7. Save Final Results ---
    final_results = {
        'baseline_accuracy': baseline_accuracy,
        'accuracy_drops': accuracy_drops,
        'sorted_channels': sorted_channels,
        'model_path': model_path,
        'layer_ablated': layer_to_ablate,
        'total_channels': num_channels,
        'dataset_info': {
            'tasks': tasks,
            'support_sizes': [4, 6, 8, 10],
            'total_episodes': len(test_dataset)
        }
    }
    
    final_filename = f'ablation_results_{layer_to_ablate}_final.json'
    with open(final_filename, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nFinal results saved to {final_filename}")

if __name__ == '__main__':
    main() 