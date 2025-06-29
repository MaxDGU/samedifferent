import torch
import torch.nn as nn
import os
import sys
from PIL import Image

# Add the root directory to the path to allow imports from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.models.conv4 import SameDifferentCNN
from circuit_analysis.analyzer import CircuitAnalyzer

def main():
    """Main function to visualize the features of a critical path."""
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = "visualizations/critical_path_features"
    os.makedirs(output_dir, exist_ok=True)

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

    # --- 3. Initialize Analyzer ---
    analyzer = CircuitAnalyzer(model)

    # --- 4. Define and Visualize Critical Path ---
    critical_path = [('conv1', 2), ('conv2', 6), ('conv3', 4), ('conv4', 74)]
    print(f"\\n--- Visualizing Features for Critical Path ---")
    
    for layer_name, channel_idx in critical_path:
        print(f"Visualizing: Layer '{layer_name}', Channel {channel_idx}")
        
        # Generate the visualization
        visualization_image = analyzer.visualize_channel(
            layer_name, 
            channel_idx,
            steps=500,
            lr=0.05
        )
        
        # Save the result
        image_path = os.path.join(output_dir, f"{layer_name}_channel_{channel_idx}.png")
        visualization_image.save(image_path)
        print(f"  -> Saved visualization to {image_path}")

    print("\\nFinished visualizing the critical path.")

if __name__ == '__main__':
    main() 