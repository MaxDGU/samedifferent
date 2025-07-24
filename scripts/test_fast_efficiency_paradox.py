#!/usr/bin/env python3
"""
Quick Test for Fast Efficiency Paradox Experiment

This script validates the fast efficiency paradox implementation
with minimal parameters before running the full experiment.
"""

import os
import sys
import torch
import numpy as np
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_fast_efficiency_paradox():
    """Run a quick test of the fast efficiency paradox experiment"""
    
    print("🧪 TESTING FAST EFFICIENCY PARADOX EXPERIMENT")
    print("="*50)
    
    try:
        from scripts.efficiency_paradox_fast import FastEfficiencyParadoxExperiment
        print("✅ Successfully imported FastEfficiencyParadoxExperiment")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Create minimal test arguments
    class TestArgs:
        def __init__(self):
            self.save_dir = 'test_results/fast_efficiency_paradox_test'
            self.test_episodes = 5  # Very few for testing
            self.inner_lr = 0.001
            self.vanilla_lr = 0.001
    
    args = TestArgs()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment
    experiment = FastEfficiencyParadoxExperiment(args, device)
    
    try:
        print("\n🔬 Testing Phase 1: Training Efficiency Synthesis...")
        training_data = experiment.synthesize_training_efficiency_from_existing()
        
        # Validate synthesized data
        assert 'maml' in training_data
        assert 'vanilla' in training_data
        assert len(training_data['maml']) == 4  # 4 data scales
        assert len(training_data['vanilla']) == 4
        print("✅ Training efficiency synthesis working correctly")
        
        print("\n⚡ Testing Phase 2: Model Loading...")
        models = experiment._load_existing_pretrained_models()
        assert 'maml' in models
        assert 'vanilla' in models
        print("✅ Model loading working correctly")
        
        print("\n🎯 Testing Phase 3: Episode Generation...")
        episodes = experiment._generate_synthetic_episodes(shot_size=4, num_episodes=5)
        assert len(episodes) == 5
        assert episodes[0]['support_images'].shape == (4, 3, 64, 64)
        print("✅ Episode generation working correctly")
        
        print("\n📊 Testing Phase 4: Adaptation Testing...")
        # Test with minimal episodes
        experiment.args.test_episodes = 3
        maml_metrics = experiment._fast_test_maml_adaptation(models['maml'], 4, 3)
        vanilla_metrics = experiment._fast_test_vanilla_adaptation(models['vanilla'], 4, 3)
        
        assert 'steps_to_70' in maml_metrics
        assert 'final_accuracy' in maml_metrics
        assert 'adaptation_curve' in maml_metrics
        print("✅ Adaptation testing working correctly")
        
        print("\n📈 Testing Phase 5: Statistical Analysis...")
        # Mock some adaptation data for testing
        experiment.results['adaptation_efficiency_measured'] = {
            'maml': [{'shot_size': 4, 'steps_to_70': 3.0, 'final_accuracy': 75.0}],
            'vanilla': [{'shot_size': 4, 'steps_to_70': 8.0, 'final_accuracy': 65.0}]
        }
        paradox_results = experiment.analyze_efficiency_paradox()
        
        assert 'paradox_validated' in paradox_results
        assert 'training_cost' in paradox_results
        assert 'adaptation_benefit' in paradox_results
        print("✅ Statistical analysis working correctly")
        
        print("\n🎨 Testing Phase 6: Visualization...")
        viz_path = experiment.create_fast_visualization()
        print(f"✅ Visualization created: {viz_path}")
        
        print("\n💾 Testing Phase 7: Save Results...")
        experiment.save_results()
        print("✅ Results saved successfully")
        
        # Clean up test files
        import shutil
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        print("✅ Test cleanup completed")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The fast efficiency paradox experiment is ready to run.")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all required imports work"""
    print("\n🔧 TESTING IMPORTS")
    print("="*30)
    
    required_imports = [
        'torch', 'numpy', 'matplotlib', 'scipy', 'learn2learn',
        'meta_baseline.models.conv6lr'
    ]
    
    for module in required_imports:
        try:
            if module == 'meta_baseline.models.conv6lr':
                from meta_baseline.models.conv6lr import SameDifferentCNN
                print(f"✅ {module}")
            else:
                __import__(module)
                print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            return False
    
    print("✅ All imports working correctly")
    return True

def main():
    parser = argparse.ArgumentParser(description='Test Fast Efficiency Paradox Experiment')
    parser.add_argument('--test_imports_only', action='store_true',
                       help='Only test imports, not the full experiment')
    
    args = parser.parse_args()
    
    print("🧪 FAST EFFICIENCY PARADOX EXPERIMENT - TESTING SUITE")
    print("="*60)
    
    if args.test_imports_only:
        success = test_imports()
    else:
        # Test imports first
        if not test_imports():
            print("\n❌ Import tests failed! Fix imports before running experiment.")
            return 1
        
        # Test full experiment
        success = test_fast_efficiency_paradox()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("The fast efficiency paradox experiment is ready to run.")
        print("\nTo run the full experiment:")
        print("  Local: python scripts/efficiency_paradox_fast.py")
        print("  Cluster: sbatch run_fast_efficiency_paradox.slurm")
    else:
        print("\n❌ TESTS FAILED!")
        print("Please fix the issues before running the experiment.")
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main()) 