#!/usr/bin/env python3
"""
Trial run for three-way adaptation efficiency comparison.
This runs with minimal parameters to verify the approach works.
"""

import subprocess
import sys
import os

def main():
    """Run a quick trial of the three-way adaptation efficiency test."""
    
    print("🚀 THREE-WAY ADAPTATION EFFICIENCY TRIAL")
    print("=" * 70)
    print("Testing the script with minimal parameters")
    print("NOTE: This will use local paths - may not find actual pretrained models")
    print("Main test should be run on Della with actual pretrained model paths")
    print("=" * 70)
    
    # Trial parameters (very small for quick testing)
    cmd = [
        "python", "scripts/test_adaptation_efficiency_three_way.py",
        "--naturalistic_data_path", "data/naturalistic/test.h5",  # Local path (may not exist)
        "--meta_model_base_path", "results/meta_baselines/conv6",  # Local path (may not exist)
        "--vanilla_model_base_path", "results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular",  # Local path (may not exist)
        "--save_dir", "results/three_way_adaptation_trial",
        "--inner_lr", "0.01",
        "--vanilla_lr", "0.001",
        "--max_test_episodes", "5",        # Very small for trial
        "--max_adaptation_steps", "5",     # Fewer steps for trial
        "--target_accuracies", "55.0", "60.0"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n🎉 TRIAL COMPLETED SUCCESSFULLY!")
        print("The script structure works. Now run on Della with actual pretrained models.")
        print("Use: sbatch run_three_way_adaptation_efficiency.slurm")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TRIAL FAILED with return code {e.returncode}")
        print("This may be expected if pretrained models are not available locally.")
        print("Check the error above for any structural issues with the script.")
        return False
    except Exception as e:
        print(f"\n❌ TRIAL FAILED with error: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
