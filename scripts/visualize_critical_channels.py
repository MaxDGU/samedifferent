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
    """Main function to visualize a critical channel."""
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = "visualizations/feature_visualizations"
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
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    print(f"Loaded trained model from {model_path}")

    # --- 3. Initialize Analyzer ---
    analyzer = CircuitAnalyzer(model)

    # --- 4. Visualize a Critical Channel ---
    # From our previous experiment, channel 74 was the most critical in conv4
    layer_to_visualize = 'conv4'
    channel_to_visualize = 74
    
    print(f"\n--- Visualizing Channel {channel_to_visualize} of layer '{layer_to_visualize}' ---")
    
    # Generate the visualization
    visualization_image = analyzer.visualize_channel(
        layer_to_visualize, 
        channel_to_visualize,
        steps=500, # More steps for a clearer image
        lr=0.05
    )
    
    # --- 5. Save the Result ---
    image_path = os.path.join(output_dir, f"{layer_to_visualize}_channel_{channel_to_visualize}.png")
    visualization_image.save(image_path)
    
    print(f"Saved visualization to {image_path}")
    print("Open the image to see what the channel has learned to detect.")

if __name__ == '__main__':
    main() 