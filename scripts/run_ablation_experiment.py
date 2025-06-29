import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the root directory to the path to allow imports from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.models.conv4 import SameDifferentCNN
from baselines.models.utils import SameDifferentDataset
from circuit_analysis.analyzer import CircuitAnalyzer

def calculate_accuracy(model, data_loader, device, analyzer=None, layer_to_ablate=None, channel_to_ablate=None):
    """Calculates the model's accuracy on a given dataset."""
    model.eval()
    correct = 0
    total = 0
    
    if analyzer and layer_to_ablate and channel_to_ablate is not None:
        analyzer.ablate_layer(layer_to_ablate, channel_to_ablate)

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
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
    model_path = './results/naturalistic/vanilla/conv4/seed_42/best_model.pt'
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
    task = 'regular'
    # Use a smaller support size for faster testing if needed
    test_dataset = SameDifferentDataset(data_dir, [task], 'test', support_sizes=[4]) 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Loaded '{task}' test dataset.")

    # --- 4. Initialize Analyzer ---
    analyzer = CircuitAnalyzer(model)

    # --- 5. Run Ablation Experiment ---
    layer_to_ablate = 'conv4'
    layer = analyzer._layer_map[layer_to_ablate]
    num_channels = layer.out_channels
    
    print(f"\n--- Starting Ablation Experiment on '{layer_to_ablate}' ---")
    
    # First, get baseline accuracy without any ablation
    baseline_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Baseline accuracy on '{task}': {baseline_accuracy:.4f}")

    # Now, ablate each channel and measure the accuracy drop
    accuracy_drops = {}
    for channel_idx in tqdm(range(num_channels), desc=f"Ablating channels in {layer_to_ablate}"):
        ablated_accuracy = calculate_accuracy(model, test_loader, device, analyzer, layer_to_ablate, channel_idx)
        accuracy_drops[channel_idx] = baseline_accuracy - ablated_accuracy
        
    # --- 6. Report Results ---
    print("\n--- Ablation Results ---")
    # Sort channels by the magnitude of accuracy drop
    sorted_channels = sorted(accuracy_drops.items(), key=lambda item: item[1], reverse=True)

    print(f"Top 5 most critical channels in '{layer_to_ablate}' for task '{task}':")
    for i in range(min(5, len(sorted_channels))):
        channel_idx, drop = sorted_channels[i]
        print(f"  - Channel {channel_idx}: Accuracy drop of {drop:.4f}")

if __name__ == '__main__':
    main() 