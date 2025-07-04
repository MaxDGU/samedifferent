#!/usr/bin/env python3
"""
Test Sample Efficiency Comparison (Local Version)

This is a simplified version for local testing with smaller parameters.
"""

import os
import sys
import subprocess

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    print("Testing Sample Efficiency Comparison (Local Version)")
    print("=" * 50)
    
    # Check if required directories exist
    data_dir = "data/meta_h5/pb"
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found: {data_dir}")
        print("You may need to adjust the data paths for your local setup.")
        print("The script uses the same H5 files for both meta-learning and vanilla SGD.")
    
    # Run a quick test with minimal parameters
    cmd = [
        "python", "scripts/sample_efficiency_comparison.py",
        "--epochs", "2",
        "--meta_batch_size", "2",
        "--vanilla_batch_size", "8",
        "--val_frequency", "100",  # Validate every 100 batches for local testing
        "--save_dir", "results/sample_efficiency_test",
        "--methods", "vanilla"  # Test with vanilla SGD first since it was stalling
    ]
    
    print("Running test command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Test completed successfully!")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Test failed with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: Could not run the script. Make sure you're in the correct directory.")

if __name__ == '__main__':
    main() 