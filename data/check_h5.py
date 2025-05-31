#!/usr/bin/env python
"""
Check if we can open an h5 file
"""

import h5py
import os

def main():
    # Path to an h5 file
    h5_path = "data/meta_h5/pb/arrows_support10_test.h5"
    
    # Check if the file exists
    if not os.path.exists(h5_path):
        print(f"File does not exist: {h5_path}")
        return
    
    # Try to open the file
    try:
        with h5py.File(h5_path, 'r') as f:
            # Print the keys
            print(f"Keys in the h5 file: {list(f.keys())}")
            
            # Print the shape of the first dataset
            for key in f.keys():
                print(f"Shape of {key}: {f[key].shape}")
                
            print("Successfully opened the h5 file!")
    except Exception as e:
        print(f"Error opening the h5 file: {str(e)}")

if __name__ == "__main__":
    main() 