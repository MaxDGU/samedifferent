import torch
import torch.nn as nn
from functools import partial
from torchvision import transforms
import numpy as np

class CircuitAnalyzer:
    """
    A class to analyze the circuits of a PyTorch model.
    It provides tools for activation extraction, ablation studies, feature visualization, and more.
    """
    def __init__(self, model: nn.Module):
        """
        Initializes the CircuitAnalyzer.

        Args:
            model (nn.Module): The model to analyze.
        """
        self.model = model
        self.model.eval()
        self.hooks = {}
        self.activations = {}
        self._layer_map = self._get_layer_map()

    def _get_layer_map(self):
        """Creates a map of layer names to layer objects."""
        layer_map = {}
        for name, layer in self.model.named_modules():
            # We only want to hook leaf modules
            if not list(layer.children()):
                layer_map[name] = layer
        return layer_map

    def get_all_layers(self):
        """Returns a list of all layer names in the model."""
        return list(self._layer_map.keys())

    def _hook_fn(self, layer_name, edit_fn=None):
        """A factory for hook functions that store and optionally edit activations."""
        def hook(model, input, output):
            if edit_fn:
                output = edit_fn(output)
            self.activations[layer_name] = output.detach()
            return output
        return hook

    def register_hook(self, layer_names, edit_fn=None):
        """
        Registers a forward hook on the specified layers.

        Args:
            layer_names (list or str): A list of layer names or a single layer name.
            edit_fn (function, optional): A function to apply to the layer's output.
                                          It should take the output tensor and return an edited tensor.
        """
        if isinstance(layer_names, str):
            layer_names = [layer_names]
        
        for layer_name in layer_names:
            if layer_name in self.hooks:
                # To modify a hook, it must be removed first
                self.hooks[layer_name].remove()
            
            if layer_name in self._layer_map:
                layer = self._layer_map[layer_name]
                self.hooks[layer_name] = layer.register_forward_hook(self._hook_fn(layer_name, edit_fn))
            else:
                raise ValueError(f"Layer '{layer_name}' not found in the model.")

    def _get_ablation_fn(self, indices, value=0):
        """Creates a function to ablate (zero out) specific indices in a tensor."""
        def ablation_fn(tensor):
            cloned_tensor = tensor.clone()
            # This works for both FC layers (dim 1) and Conv layers (dim 1 is channels)
            cloned_tensor[:, indices] = value
            return cloned_tensor
        return ablation_fn

    def ablate_layer(self, layer_name, indices, value=0):
        """
        Ablates (e.g., zeros out) specific channels or neurons in a layer's output.

        Args:
            layer_name (str): The name of the layer to ablate.
            indices (int or list[int]): The index or indices of the channel/neuron to ablate.
            value (float, optional): The value to set the ablated elements to. Defaults to 0.
        """
        if isinstance(indices, int):
            indices = [indices]
        
        ablation_fn = self._get_ablation_fn(indices, value)
        self.register_hook(layer_name, edit_fn=ablation_fn)

    def ablate_path(self, path, value=0):
        """
        Ablates all channels/neurons in a given path.

        Args:
            path (list[tuple]): A list of (layer_name, channel_idx) tuples.
            value (float, optional): The value to set the ablated elements to. Defaults to 0.
        """
        for layer_name, channel_idx in path:
            # This will register a unique hook for each layer in the path
            self.ablate_layer(layer_name, channel_idx, value)

    def get_activations(self, input_tensor):
        """
        Performs a forward pass and returns the captured activations.

        Args:
            input_tensor (torch.Tensor): The input to the model.

        Returns:
            dict: A dictionary of activations from the hooked layers.
        """
        self.activations = {}
        with torch.no_grad():
            output = self.model(input_tensor)
        return output, self.activations

    def visualize_channel(self, layer_name, channel_idx, image_size=(128, 128), lr=0.1, steps=100, l2_reg=1e-4):
        """
        Generates an image that maximally activates a specific channel in a convolutional layer.

        Args:
            layer_name (str): The name of the convolutional layer.
            channel_idx (int): The index of the channel to visualize.
            image_size (tuple, optional): The size of the image to generate. Defaults to (128, 128).
            lr (float, optional): The learning rate for the optimization. Defaults to 0.1.
            steps (int, optional): The number of optimization steps. Defaults to 100.
            l2_reg (float, optional): The L2 regularization strength. Defaults to 1e-4.

        Returns:
            PIL.Image.Image: The generated image.
        """
        # Start with a random noise image
        device = next(self.model.parameters()).device
        image = torch.randn(1, 3, *image_size, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([image], lr=lr)

        # We need to hook the target layer to get its activation
        self.register_hook(layer_name)

        for _ in range(steps):
            optimizer.zero_grad()
            self.model(image)
            activation = self.activations[layer_name]
            
            # We want to maximize the mean activation of the target channel
            # The negative sign is because optimizers minimize
            loss = -torch.mean(activation[0, channel_idx]) + l2_reg * torch.norm(image, 2)
            loss.backward()
            optimizer.step()

        self.remove_hooks()

        # Convert the tensor to an image
        image = image.detach().cpu().squeeze(0)
        image = (image - image.min()) / (image.max() - image.min()) # Normalize to [0, 1]
        
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(image)
        
        return pil_image

    def trace_path_backwards(self, target_layer_name, target_channel_idx):
        """
        Traces the most influential path backward from a target channel.

        Args:
            target_layer_name (str): The name of the starting layer.
            target_channel_idx (int): The index of the starting channel in the target layer.

        Returns:
            list: A list of tuples, where each tuple contains (layer_name, channel_idx)
                  representing the most influential path.
        """
        path = [(target_layer_name, target_channel_idx)]
        
        # Get all conv layers in order
        all_layers = self.get_all_layers()
        conv_layers = [name for name in all_layers if isinstance(self._layer_map[name], nn.Conv2d)]
        
        # Start from the layer before our target
        try:
            start_index = conv_layers.index(target_layer_name) - 1
        except ValueError:
            print(f"Error: Target layer '{target_layer_name}' not found or is not a Conv layer.")
            return []

        current_channel = target_channel_idx
        for i in range(start_index, -1, -1):
            current_layer_name = conv_layers[i+1]
            prev_layer_name = conv_layers[i]
            
            current_layer = self._layer_map[current_layer_name]
            weights = current_layer.weight.data.clone() # (out_channels, in_channels, H, W)
            
            # Get the weights connecting to our current channel of interest
            weights_for_channel = weights[current_channel, :, :, :]
            
            # Find the most influential input channel by summing the absolute weights
            # across the kernel dimensions
            influence = torch.sum(torch.abs(weights_for_channel), dim=(1, 2))
            most_influential_channel = torch.argmax(influence).item()
            
            path.insert(0, (prev_layer_name, most_influential_channel))
            current_channel = most_influential_channel
            
        return path

    def remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks = {}
        self.activations = {}

    def __del__(self):
        self.remove_hooks()

if __name__ == '__main__':
    # Example Usage
    # This example assumes you have a trained model and some data.
    
    # 1. First, let's import the model we examined earlier
    # To do this, we need to make sure the path is set up correctly.
    # Since this script is in circuit_analysis, and the model is in baselines, we need to adjust the path.
    import sys
    import os
    # Add the parent directory to the path to allow imports from other directories.
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from baselines.models.conv4 import SameDifferentCNN

    # 2. Instantiate the model.
    # We'll use the default parameters for this example.
    model = SameDifferentCNN()
    
    # 3. Create some dummy input data.
    # The model expects a batch of images of size 128x128.
    dummy_input = torch.randn(1, 3, 128, 128)

    # 4. Initialize the CircuitAnalyzer.
    analyzer = CircuitAnalyzer(model)

    # 5. Get a list of all layers in the model.
    all_layers = analyzer.get_all_layers()
    print("Available layers in the model:")
    for layer in all_layers:
        print(f"- {layer}")

    # 6. Register hooks on a few convolutional layers.
    layers_to_hook = ['conv1', 'conv2', 'conv3', 'conv4']
    analyzer.register_hook(layers_to_hook)
    print(f"\nRegistered hooks on: {', '.join(layers_to_hook)}")

    # 7. Perform a forward pass and get the activations.
    output, activations = analyzer.get_activations(dummy_input)

    print("\nCaptured activations:")
    for layer_name, activation in activations.items():
        print(f"- {layer_name}: {activation.shape}")
        
    # 8. Clean up by removing hooks.
    analyzer.remove_hooks()
    print("\nHooks removed.")

    print("--- 2. Testing Ablation ---")
    target_layer = 'conv2'
    channel_to_ablate = 5
    
    # Get baseline output (no ablation)
    output_before_ablation, _ = analyzer.get_activations(dummy_input)

    # Register ablation hook
    print(f"Ablating channel {channel_to_ablate} of layer '{target_layer}'")
    analyzer.ablate_layer(target_layer, channel_to_ablate)

    # Get output with ablation
    output_after_ablation, activations_after_ablation = analyzer.get_activations(dummy_input)
    
    print(f"Shape of '{target_layer}' activation after ablation: {activations_after_ablation[target_layer].shape}")
    
    # Verify that the channel is ablated
    ablated_channel_values = activations_after_ablation[target_layer][0, channel_to_ablate, :, :]
    print(f"Values of ablated channel: {ablated_channel_values}")
    assert torch.all(ablated_channel_values == 0), "Ablation failed!"
    print("Ablation successful: The target channel is all zeros.")

    # Compare model output
    print(f"\nModel output (logits) before ablation: {output_before_ablation.squeeze()}")
    print(f"Model output (logits) after ablation:  {output_after_ablation.squeeze()}")
    
    # The change in output demonstrates the causal effect of the ablated channel.
    output_diff = torch.sum(torch.abs(output_before_ablation - output_after_ablation))
    print(f"Sum of absolute difference in output: {output_diff.item():.4f}")
    
    analyzer.remove_hooks()
    print("\nHooks removed.") 