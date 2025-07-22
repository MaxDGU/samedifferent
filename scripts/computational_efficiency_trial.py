#!/usr/bin/env python3
"""
Computational Efficiency Trial Run

Quick trial version of the computational efficiency comparison for testing on Della.
Uses minimal parameters to verify the script works before running the full experiment.
"""

import subprocess
import sys
import argparse

def run_trial():
    """Run a quick trial of the computational efficiency comparison."""
    
    # Trial parameters - very small for quick testing
    trial_args = [
        'python', 'scripts/computational_efficiency_comparison.py',
        '--max_training_batches', '3',        # Very few batches
        '--max_test_episodes', '10',          # Few test episodes  
        '--max_adaptation_steps', '5',        # Fewer adaptation steps
        '--meta_batch_size', '4',             # Smaller batch size
        '--training_adaptation_steps', '3',   # Fewer training steps
        '--save_dir', 'results/computational_efficiency_trial'
    ]
    
    print("🧪 COMPUTATIONAL EFFICIENCY TRIAL RUN")
    print("=" * 50)
    print("Running minimal test to verify script functionality...")
    print("Parameters:")
    print("  - Training batches: 3")
    print("  - Test episodes: 10") 
    print("  - Max adaptation steps: 5")
    print("  - Meta batch size: 4")
    print("=" * 50)
    
    try:
        result = subprocess.run(trial_args, check=True, capture_output=True, text=True)
        print("✅ TRIAL RUN SUCCESSFUL!")
        print("\nOutput:")
        print(result.stdout)
        
        if result.stderr:
            print("\nWarnings:")
            print(result.stderr)
            
        print("\n🎉 Script is working correctly. Ready for full experiment!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ TRIAL RUN FAILED!")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Trial run for computational efficiency comparison')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    success = run_trial()
    
    if success:
        print("\n📋 Next steps:")
        print("1. If trial run successful, run full experiment with SLURM:")
        print("   sbatch run_computational_efficiency.slurm")
        print("2. Results will be saved to results/computational_efficiency/")
        sys.exit(0)
    else:
        print("\n🔧 Fix any errors before running the full experiment")
        sys.exit(1)

if __name__ == '__main__':
    main() 