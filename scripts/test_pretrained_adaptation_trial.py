#!/usr/bin/env python3
"""
Trial run for pretrained adaptation efficiency test.
This runs with minimal parameters to verify the approach works.
"""

import subprocess
import sys

def main():
    """Run a quick trial of the pretrained adaptation efficiency test."""
    
    print("🚀 PRETRAINED ADAPTATION EFFICIENCY TRIAL")
    print("=" * 60)
    print("Testing the script with minimal parameters")
    print("NOTE: This will use local model paths - may not find actual pretrained models")
    print("Main test should be run on Della with actual pretrained model paths")
    print("=" * 60)
    
    # Trial parameters (very small for quick testing)
    cmd = [
        "python", "scripts/test_adaptation_efficiency_pretrained.py",
        "--data_dir", "data/meta_h5/pb",
        "--model_base_path", "results/meta_baselines/conv6",  # Local path (may not exist)
        "--save_dir", "results/pretrained_adaptation_trial",
        "--batch_size", "4",
        "--inner_lr", "0.01", 
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
        print("Use: sbatch run_pretrained_adaptation_efficiency.slurm")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TRIAL FAILED with return code {e.returncode}")
        print("This may be expected if pretrained models are not available locally.")
        print("Check the error above for any structural issues with the script.")
        return False
    except Exception as e:
        print(f"\n❌ TRIAL FAILED with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 