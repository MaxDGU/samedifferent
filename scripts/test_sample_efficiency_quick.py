#!/usr/bin/env python3
"""
Quick Test for Sample Efficiency Comparison

This script runs a minimal version of the sample efficiency comparison 
to verify everything works before running the full experiment.
"""

import os
import sys
import subprocess
import argparse

# Add project root to Python path
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_quick_test():
    """Run a quick test with minimal parameters."""
    print("Sample Efficiency Comparison - Quick Test")
    print("=" * 45)
    
    # Check if verification script exists and run it
    if os.path.exists("scripts/verify_sample_efficiency_setup.py"):
        print("Running verification checks...")
        try:
            result = subprocess.run([
                "python", "scripts/verify_sample_efficiency_setup.py"
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode != 0:
                print("❌ Verification failed. Please fix issues before continuing.")
                return False
                
        except Exception as e:
            print(f"❌ Error running verification: {e}")
            return False
    
    # Run quick test with minimal parameters
    print("\nRunning quick test with minimal parameters...")
    
    cmd = [
        "python", "scripts/sample_efficiency_comparison.py",
        "--epochs", "2",
        "--seed", "42",
        "--meta_batch_size", "4",
        "--vanilla_batch_size", "8",
        "--inner_lr", "0.01",
        "--outer_lr", "0.001",
        "--vanilla_lr", "0.001",
        "--adaptation_steps", "3",
        "--val_frequency", "5",
        "--save_dir", "results/sample_efficiency_test",
        "--methods", "vanilla",  # Start with just vanilla SGD
        "--data_dir", "data/meta_h5/pb"
    ]
    
    print("Command:", " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Quick test completed successfully!")
            
            # Try to generate a plot
            plot_cmd = [
                "python", "scripts/plot_sample_efficiency_results.py",
                "--results_dir", "results/sample_efficiency_test/seed_42"
            ]
            
            print("\nGenerating test plot...")
            plot_result = subprocess.run(plot_cmd, capture_output=True, text=True)
            
            if plot_result.returncode == 0:
                print("✅ Plot generation successful!")
                print("Test results saved to: results/sample_efficiency_test/seed_42/")
            else:
                print("❌ Plot generation failed:")
                print(plot_result.stdout)
                print(plot_result.stderr)
            
            return True
        else:
            print(f"\n❌ Quick test failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running quick test: {e}")
        return False

def run_full_methods_test():
    """Run a test with all three methods but minimal parameters."""
    print("\nRunning test with all three methods...")
    
    cmd = [
        "python", "scripts/sample_efficiency_comparison.py",
        "--epochs", "3",
        "--seed", "42",
        "--meta_batch_size", "4",
        "--vanilla_batch_size", "8",
        "--inner_lr", "0.01",
        "--outer_lr", "0.001",
        "--vanilla_lr", "0.001",
        "--adaptation_steps", "3",
        "--val_frequency", "5",
        "--save_dir", "results/sample_efficiency_full_test",
        "--methods", "fomaml", "second_order", "vanilla",
        "--data_dir", "data/meta_h5/pb"
    ]
    
    print("Command:", " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Full methods test completed successfully!")
            
            # Generate comparison plot
            plot_cmd = [
                "python", "scripts/plot_sample_efficiency_results.py",
                "--results_dir", "results/sample_efficiency_full_test/seed_42",
                "--individual_plots"
            ]
            
            print("\nGenerating comparison plot...")
            plot_result = subprocess.run(plot_cmd, capture_output=True, text=True)
            
            if plot_result.returncode == 0:
                print("✅ Comparison plot generated successfully!")
                print("Full test results saved to: results/sample_efficiency_full_test/seed_42/")
            else:
                print("❌ Plot generation failed:")
                print(plot_result.stdout)
                print(plot_result.stderr)
            
            return True
        else:
            print(f"\n❌ Full methods test failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running full methods test: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Quick Test for Sample Efficiency Comparison')
    parser.add_argument('--full', action='store_true', 
                       help='Run test with all three methods (takes longer)')
    args = parser.parse_args()
    
    # Run basic test first
    if not run_quick_test():
        print("\n❌ Basic test failed. Please fix issues before proceeding.")
        return False
    
    # Run full methods test if requested
    if args.full:
        if not run_full_methods_test():
            print("\n❌ Full methods test failed.")
            return False
    
    print("\n" + "=" * 45)
    print("✅ All tests passed! Ready for full experiment on della.")
    print("\nNext steps:")
    print("1. Copy files to della")
    print("2. Run: sbatch run_sample_efficiency_comparison.slurm")
    print("3. Monitor with: squeue -u $USER")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 