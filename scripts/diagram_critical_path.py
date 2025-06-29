import torch
import torch.nn as nn
import os
import sys

# Add the root directory to the path to allow imports from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.models.conv4 import SameDifferentCNN
from circuit_analysis.analyzer import CircuitAnalyzer

def generate_diagram_definition(critical_path):
    """Generates a Mermaid diagram definition string from a critical path."""
    
    diagram_str = "graph TD;\\n"
    diagram_str += "    subgraph Model Architecture;\\n"
    
    # Define nodes for all conv layers
    layers = ["conv1", "conv2", "conv3", "conv4"]
    for layer in layers:
        diagram_str += f"        {layer};\\n"
        
    diagram_str += "    end;\\n\\n"
    
    # Add connections for the critical path
    for i in range(len(critical_path) - 1):
        from_layer, from_channel = critical_path[i]
        to_layer, to_channel = critical_path[i+1]
        diagram_str += f"    {from_layer}[{from_layer}<br/>Channel {from_channel}] --> {to_layer}[{to_layer}<br/>Channel {to_channel}];\\n"
        
    # Highlight the nodes in the critical path
    for layer, channel in critical_path:
        diagram_str += f"    style {layer} fill:#f9f,stroke:#333,stroke-width:4px;\\n"
        
    return diagram_str

def main():
    """Main function to trace and diagram a critical path."""
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

    # --- 3. Initialize Analyzer and Trace Path ---
    analyzer = CircuitAnalyzer(model)
    target_layer = 'conv4'
    # From our ablation experiment, channel 74 was most critical.
    target_channel = 74 
    
    print(f"\\n--- Tracing Critical Path from {target_layer} Channel {target_channel} ---")
    critical_path = analyzer.trace_path_backwards(target_layer, target_channel)
    
    if not critical_path:
        print("Could not trace the critical path.")
        return

    print("Identified Critical Path:")
    for layer, channel in critical_path:
        print(f"  - Layer: {layer}, Channel: {channel}")
        
    # --- 4. Generate and Print Diagram Definition ---
    diagram_definition = generate_diagram_definition(critical_path)
    print("\\n--- Mermaid Diagram Definition ---")
    print("Copy the following definition into a Mermaid viewer to see the diagram:")
    print(diagram_definition)


if __name__ == '__main__':
    main() 