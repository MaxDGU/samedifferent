#!/usr/bin/env python3
"""
Quick Test for Sample Efficiency Comparison

This script runs a very short test to validate that all three training methods
work correctly with the new aligned parameters and validation checks.
"""

import os
import sys
import subprocess

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    print("Quick Sample Efficiency Validation Test")
    print("=" * 50)
    
    # Check if required directories exist
    data_dir = "data/meta_h5/pb"
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found: {data_dir}")
        print("You may need to adjust the data paths for your local setup.")
    
    print("Testing all three methods with new aligned parameters...")
    print("- FOMAML: inner_lr=0.05, outer_lr=0.001")
    print("- Second-Order MAML: inner_lr=0.05, outer_lr=0.001") 
    print("- Vanilla SGD: lr=0.0001")
    print()
    
    # Run a very quick test with minimal parameters to validate all methods work
    cmd = [
        "python", "scripts/sample_efficiency_comparison.py",
        "--epochs", "1",  # Just 1 epoch for quick validation
        "--meta_batch_size", "2",  # Small batch size
        "--vanilla_batch_size", "8",
        "--val_frequency", "50",  # Validate every 50 batches for quick feedback
        "--save_dir", "results/sample_efficiency_validation_test",
        "--methods", "fomaml", "second_order", "vanilla",  # Test all methods
        "--seed", "123"  # Use different seed for testing
    ]
    
    print("Running validation test command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ VALIDATION TEST PASSED!")
        print("All three methods completed successfully with new parameters.")
        print()
        print("Key output highlights:")
        print("-" * 30)
        
        # Look for key indicators in the output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if ('Parameters:' in line or 
                'Model parameters:' in line or
                'Training Summary' in line or
                'Final validation accuracy:' in line or
                'Total improvement:' in line):
                print(f"  {line.strip()}")
        
        print()
        print("The experiment is ready to run with aligned parameters!")
        
    except subprocess.CalledProcessError as e:
        print("❌ VALIDATION TEST FAILED!")
        print(f"Exit code: {e.returncode}")
        print()
        print("STDOUT:")
        print("-" * 20)
        print(e.stdout)
        print()
        print("STDERR:")
        print("-" * 20)
        print(e.stderr)
        print()
        print("Please check the error above and fix any issues before running full experiment.")
        
    except FileNotFoundError:
        print("❌ ERROR: Could not find the sample efficiency script.")
        print("Make sure you're running from the correct directory.")

if __name__ == '__main__':
    main() 