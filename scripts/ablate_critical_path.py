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

def calculate_accuracy(model, data_loader, device, analyzer=None, path_to_ablate=None):
    """Calculates the model's accuracy, optionally with a path ablated."""
    model.eval()
    correct = 0
    total = 0
    
    if analyzer and path_to_ablate:
        analyzer.ablate_path(path_to_ablate)

    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if analyzer and path_to_ablate:
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
    """Main function to ablate a critical path and measure the impact."""
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model ---
    model_path = './results/naturalistic/vanilla/conv4/seed_42/best_model.pt'
    model = SameDifferentCNN()
    
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    if next(iter(state_dict)).startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    print(f"Loaded trained model from {model_path}")

    # --- 3. Load Data ---
    data_dir = 'data/meta_h5/pb'
    task = 'regular'
    test_dataset = SameDifferentDataset(data_dir, [task], 'test', support_sizes=[4]) 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Loaded '{task}' test dataset.")

    # --- 4. Initialize Analyzer ---
    analyzer = CircuitAnalyzer(model)

    # --- 5. Define Critical Path and Run Experiment ---
    critical_path = [('conv1', 2), ('conv2', 6), ('conv3', 4), ('conv4', 74)]
    print(f"\\n--- Ablating Critical Path ---")
    print(f"Path: {critical_path}")
    
    # Get baseline accuracy
    baseline_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")

    # Get accuracy with the path ablated
    ablated_accuracy = calculate_accuracy(model, test_loader, device, analyzer, path_to_ablate=critical_path)
    print(f"Accuracy after ablating path: {ablated_accuracy:.4f}")
    
    # --- 6. Report Results ---
    accuracy_drop = baseline_accuracy - ablated_accuracy
    print(f"\\nTotal accuracy drop from ablating the path: {accuracy_drop:.4f}")

if __name__ == '__main__':
    main() 