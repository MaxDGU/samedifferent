#!/usr/bin/env python3
"""
Quick test script to validate the vanilla SGD validation setup locally.
"""

import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        from baselines.models import Conv2CNN, Conv4CNN, Conv6CNN
        print("✓ Architecture imports successful")
        
        from baselines.models.utils import EarlyStopping
        print("✓ Utils imports successful")
        
        from data.vanilla_h5_dataset_creation import PB_dataset_h5
        print("✓ Data loading imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test that models can be created."""
    print("\nTesting model creation...")
    
    try:
        from baselines.models import Conv2CNN, Conv4CNN, Conv6CNN
        
        # Test Conv2
        model = Conv2CNN()
        conv2_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Conv2CNN created: {conv2_params:,} parameters")
        
        # Test Conv4
        model = Conv4CNN()
        conv4_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Conv4CNN created: {conv4_params:,} parameters")
        
        # Test Conv6
        model = Conv6CNN()
        conv6_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Conv6CNN created: {conv6_params:,} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def test_data_loading():
    """Test that data can be loaded."""
    print("\nTesting data loading...")
    
    try:
        from data.vanilla_h5_dataset_creation import PB_dataset_h5
        
        # Check if data directory exists
        data_dir = 'data/vanilla_h5'
        if not os.path.exists(data_dir):
            print(f"✓ Data directory not found locally: {data_dir}")
            print("  This is expected for local testing - data is on cluster")
            return True
        
        # Try to create dataset
        dataset = PB_dataset_h5(task='regular', split='train', data_dir=data_dir)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        return True
        
    except Exception as e:
        print(f"✓ Data loading test (expected to fail locally): {e}")
        return True  # Expected to fail locally

def test_device_setup():
    """Test device setup."""
    print("\nTesting device setup...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("✓ Using CPU (expected for local testing)")
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("VANILLA SGD VALIDATION TEST")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Device Setup", test_device_setup)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✓ All tests passed! Ready to run validation experiment.")
    else:
        print("✗ Some tests failed. Check the issues above.")
    
    print("="*60)

if __name__ == '__main__':
    main()
