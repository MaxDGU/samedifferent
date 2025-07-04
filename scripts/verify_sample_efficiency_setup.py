#!/usr/bin/env python3
"""
Verification Script for Sample Efficiency Comparison

This script verifies that all dependencies, data, and configurations 
are properly set up before running the full sample efficiency comparison.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Added to Python path: {project_root}")

def check_dependencies():
    """Check if all required packages are available."""
    print("=== Checking Dependencies ===")
    
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'h5py', 'tqdm', 'learn2learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_availability():
    """Check if all required data files exist."""
    print("\n=== Checking Data Availability ===")
    
    data_dir = "data/meta_h5/pb"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    print(f"✅ Data directory exists: {data_dir}")
    
    # Check for required tasks
    required_tasks = [
        'regular', 'lines', 'open', 'wider_line', 'scrambled',
        'random_color', 'arrows', 'irregular', 'filled', 'original'
    ]
    
    support_sizes = [4, 6, 8, 10]
    splits = ['train', 'val', 'test']
    
    missing_files = []
    total_files = 0
    found_files = 0
    
    for task in required_tasks:
        for support_size in support_sizes:
            for split in splits:
                filename = f"{task}_support{support_size}_{split}.h5"
                filepath = os.path.join(data_dir, filename)
                total_files += 1
                
                if os.path.exists(filepath):
                    found_files += 1
                else:
                    missing_files.append(filename)
    
    print(f"Found {found_files}/{total_files} required H5 files")
    
    if missing_files:
        print(f"❌ Missing files: {len(missing_files)}")
        for file in missing_files[:5]:  # Show first 5 missing files
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return False
    
    print("✅ All required data files found")
    return True

def check_model_imports():
    """Check if model classes can be imported."""
    print("\n=== Checking Model Imports ===")
    
    try:
        from meta_baseline.models.conv6lr import SameDifferentCNN
        print("✅ SameDifferentCNN - Available")
    except ImportError as e:
        print(f"❌ SameDifferentCNN - Import failed: {e}")
        return False
    
    try:
        from meta_baseline.models.utils_meta import SameDifferentDataset
        print("✅ SameDifferentDataset - Available")
    except ImportError as e:
        print(f"❌ SameDifferentDataset - Import failed: {e}")
        return False
    
    try:
        import h5py
        print("✅ h5py (for vanilla data loading) - Available")
    except ImportError as e:
        print(f"❌ h5py (for vanilla data loading) - Import failed: {e}")
        return False
    
    return True

def test_data_loading():
    """Test that data can be loaded correctly."""
    print("\n=== Testing Data Loading ===")
    
    try:
        # Test vanilla dataset loading with inline implementation
        print("Testing vanilla dataset loading...")
        
        # Simple inline test of H5 loading and flattening
        import h5py
        data_dir = 'data/meta_h5/pb'
        test_file = os.path.join(data_dir, 'regular_support4_train.h5')
        
        if os.path.exists(test_file):
            with h5py.File(test_file, 'r') as f:
                support_images = f['support_images'][:]  # Test reading
                support_labels = f['support_labels'][:]  # Test reading
                num_episodes, support_size, H, W, C = support_images.shape
                
                # Test flattening
                flat_images = support_images.reshape(-1, H, W, C)
                flat_labels = support_labels.reshape(-1)
                
                print(f"✅ Vanilla dataset loading test passed")
                print(f"  Original shape: {support_images.shape}")
                print(f"  Flattened: {flat_images.shape[0]} samples")
                print(f"  Sample image shape: {flat_images[0].shape}")
                print(f"  Sample label: {flat_labels[0]}")
        else:
            print(f"❌ Test file not found: {test_file}")
            return False
        
    except Exception as e:
        print(f"❌ Vanilla dataset loading failed: {e}")
        return False
    
    try:
        from meta_baseline.models.utils_meta import SameDifferentDataset
        
        # Test meta-learning dataset loading
        print("Testing meta-learning dataset loading...")
        dataset = SameDifferentDataset(
            data_dir='data/meta_h5/pb',
            tasks=['regular'], 
            split='train',
            support_sizes=[4]
        )
        print(f"✅ Meta-learning dataset loaded: {len(dataset)} episodes")
        
    except Exception as e:
        print(f"❌ Meta-learning dataset loading failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that models can be created."""
    print("\n=== Testing Model Creation ===")
    
    try:
        from meta_baseline.models.conv6lr import SameDifferentCNN
        
        model = SameDifferentCNN()
        print("✅ Conv6 model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 128, 128)
        output = model(dummy_input)
        print(f"✅ Forward pass successful: {output.shape}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    return True

def check_gpu_availability():
    """Check GPU availability."""
    print("\n=== Checking GPU Availability ===")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available - will run on CPU (slow but acceptable for testing)")
        return True  # Changed to True since CPU is acceptable for testing

def check_output_directories():
    """Check/create output directories."""
    print("\n=== Checking Output Directories ===")
    
    directories = [
        'results/sample_efficiency_comparison',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory ready: {directory}")
    
    return True

def main():
    """Run all verification checks."""
    print("Sample Efficiency Comparison - Setup Verification")
    print("=" * 55)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Data Availability", check_data_availability),
        ("Model Imports", check_model_imports),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("GPU Availability", check_gpu_availability),
        ("Output Directories", check_output_directories)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {check_name} - Error: {e}")
    
    print("\n" + "=" * 55)
    print(f"Verification Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All checks passed! Ready to run sample efficiency comparison.")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues before running.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 