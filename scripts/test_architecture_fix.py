#!/usr/bin/env python3
"""
Quick test to verify architecture fix works correctly.
This script tests if the Conv2 and Conv4 models now output classification logits.
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from baselines.models import Conv2CNN, Conv4CNN, Conv6CNN

def test_model_outputs():
    """Test that all models output 2-dimensional classification logits."""
    print("Testing Model Outputs")
    print("=" * 50)
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 128, 128)
    
    models = {
        'Conv2': Conv2CNN(),
        'Conv4': Conv4CNN(),
        'Conv6': Conv6CNN()
    }
    
    for name, model in models.items():
        print(f"\nTesting {name}:")
        model.eval()
        
        with torch.no_grad():
            output = model(test_input)
            
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected shape: ({batch_size}, 2)")
        
        # Check if output is correct shape for classification
        if output.shape == (batch_size, 2):
            print(f"  ✓ {name} outputs correct classification logits")
        else:
            print(f"  ✗ {name} outputs wrong shape!")
            
        # Check if output values are reasonable (not all zeros or extremely large)
        output_mean = output.mean().item()
        output_std = output.std().item()
        print(f"  Output mean: {output_mean:.4f}, std: {output_std:.4f}")
        
        # Test with BCEWithLogitsLoss (same as used in training)
        criterion = nn.BCEWithLogitsLoss()
        # Create fake labels
        labels = torch.randint(0, 2, (batch_size,)).float()
        
        try:
            # Convert 2D logits to 1D for BCE loss (same as training script)
            if output.dim() > 1 and output.shape[1] > 1:
                output_1d = output[:, 1] - output[:, 0]
            else:
                output_1d = output.squeeze()
            
            loss = criterion(output_1d, labels)
            print(f"  ✓ BCE loss works: {loss.item():.4f}")
        except Exception as e:
            print(f"  ✗ BCE loss failed: {e}")
        
        print()

def test_gradient_flow():
    """Test that gradients flow properly."""
    print("Testing Gradient Flow")
    print("=" * 50)
    
    model = Conv4CNN()
    criterion = nn.BCEWithLogitsLoss()
    
    # Create test batch
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    model.train()
    output = model(test_input)
    
    # Convert to 1D for BCE loss
    if output.dim() > 1 and output.shape[1] > 1:
        output_1d = output[:, 1] - output[:, 0]
    else:
        output_1d = output.squeeze()
    
    loss = criterion(output_1d, labels)
    loss.backward()
    
    # Check if gradients are flowing
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    
    if has_gradients:
        print("✓ Gradients are flowing properly")
    else:
        print("✗ No gradients found!")
    
    print(f"Loss: {loss.item():.4f}")
    print()

if __name__ == '__main__':
    test_model_outputs()
    test_gradient_flow()
    print("Architecture fix test completed!") 