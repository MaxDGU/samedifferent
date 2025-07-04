import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import random
from scipy import stats

# Add the root directory to the path to allow imports from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_baseline.models import Conv2CNN as SameDifferentCNN
from baselines.models.utils import PB_TASKS
from circuit_analysis.analyzer import CircuitAnalyzer
from meta_baseline.models.utils_meta import SameDifferentDataset, collate_episodes  # dataset and collate compatible with each other

def calculate_accuracy(model, data_loader, device, analyzer=None, layer_to_ablate=None, channel_to_ablate=None):
    """Calculates the model's accuracy on a given dataset."""
    model.eval()
    correct = 0
    total = 0
    
    if analyzer and layer_to_ablate and channel_to_ablate is not None:
        analyzer.ablate_layer(layer_to_ablate, channel_to_ablate)

    with torch.no_grad():
        for batch in data_loader:
            # Handle both flattened-image batches and episode-based batches
            if 'image' in batch and 'label' in batch:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
            elif 'query_images' in batch and 'query_labels' in batch:
                # Episode batch: flatten query set across episodes
                q_imgs = batch['query_images']  # [B, M, C, H, W]
                q_lbls = batch['query_labels']  # [B, M]

                B, M, C, H, W = q_imgs.shape
                images = q_imgs.view(B * M, C, H, W).to(device)
                labels = q_lbls.view(-1).to(device)
            else:
                raise KeyError("Batch must contain either ('image','label') or ('query_images','query_labels') keys.")
            
            if analyzer:
                outputs, _ = analyzer.get_activations(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    if analyzer:
        analyzer.remove_hooks()
        
    return correct / total

def calculate_accuracy_by_class(model, data_loader, device, analyzer=None, layer_to_ablate=None, neurons_to_ablate=None):
    """Calculates model accuracy separated by SAME (0) and DIFFERENT (1) classes."""
    model.eval()
    
    # Counters for each class
    correct_same = 0
    correct_diff = 0
    total_same = 0
    total_diff = 0
    
    if analyzer and layer_to_ablate and neurons_to_ablate is not None:
        analyzer.ablate_layer(layer_to_ablate, neurons_to_ablate)

    with torch.no_grad():
        for batch in data_loader:
            # Handle both flattened-image batches and episode-based batches
            if 'image' in batch and 'label' in batch:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
            elif 'query_images' in batch and 'query_labels' in batch:
                # Episode batch: flatten query set across episodes
                q_imgs = batch['query_images']  # [B, M, C, H, W]
                q_lbls = batch['query_labels']  # [B, M]

                B, M, C, H, W = q_imgs.shape
                images = q_imgs.view(B * M, C, H, W).to(device)
                labels = q_lbls.view(-1).to(device)
            else:
                raise KeyError("Batch must contain either ('image','label') or ('query_images','query_labels') keys.")
            
            if analyzer:
                outputs, _ = analyzer.get_activations(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            
            # Separate by class
            same_mask = (labels == 0)  # SAME class
            diff_mask = (labels == 1)  # DIFFERENT class
            
            correct_same += (predicted[same_mask] == labels[same_mask]).sum().item()
            correct_diff += (predicted[diff_mask] == labels[diff_mask]).sum().item()
            total_same += same_mask.sum().item()
            total_diff += diff_mask.sum().item()
            
    if analyzer:
        analyzer.remove_hooks()
        
    # Calculate accuracies
    same_acc = correct_same / total_same if total_same > 0 else 0
    diff_acc = correct_diff / total_diff if total_diff > 0 else 0
    overall_acc = (correct_same + correct_diff) / (total_same + total_diff)
    
    return {
        'overall_accuracy': overall_acc,
        'same_accuracy': same_acc,
        'different_accuracy': diff_acc,
        'total_same': total_same,
        'total_different': total_diff
    }

def run_targeted_fc_ablations(model, test_loader, device, analyzer, results):
    """Run ablations on FC neurons identified as selective for DIFFERENT stimuli."""
    
    # Top DIFFERENT-selective neurons from circuit analysis results
    # Based on /scratch/gpfs/mg7411/samedifferent/circuit_analysis_conv2_results/circuit_analysis_results.json
    different_neurons = [
        {'layer': 'fc_layers.0', 'neuron': 289, 'selectivity': -3.50, 'description': 'Top DIFFERENT neuron (fc1)'},
        {'layer': 'fc_layers.0', 'neuron': 21, 'selectivity': -0.44, 'description': 'Second DIFFERENT neuron (fc1)'},
        {'layer': 'fc_layers.0', 'neuron': 413, 'selectivity': -0.42, 'description': 'Third DIFFERENT neuron (fc1)'},
        {'layer': 'fc_layers.2', 'neuron': 892, 'selectivity': -0.014, 'description': 'FC3 DIFFERENT neuron'},
        {'layer': 'fc_layers.2', 'neuron': 308, 'selectivity': -0.013, 'description': 'FC3 DIFFERENT neuron 2'},
        {'layer': 'fc_layers.2', 'neuron': 960, 'selectivity': -0.011, 'description': 'FC3 DIFFERENT neuron 3'},
    ]
    
    print("\n--- Testing Targeted DIFFERENT-Selective FC Neurons ---")
    
    for neuron_info in tqdm(different_neurons, desc="Testing DIFFERENT neurons"):
        layer_name = neuron_info['layer']
        neuron_idx = neuron_info['neuron']
        
        # Test individual neuron ablation
        accuracy_results = calculate_accuracy_by_class(
            model, test_loader, device, analyzer, layer_name, neuron_idx
        )
        
        results['targeted_fc_ablations'][f"{layer_name}_neuron_{neuron_idx}"] = {
            'neuron_info': neuron_info,
            'accuracy_results': accuracy_results
        }
        
        print(f"{layer_name} neuron {neuron_idx}: "
              f"Overall: {accuracy_results['overall_accuracy']:.4f}, "
              f"SAME: {accuracy_results['same_accuracy']:.4f}, "
              f"DIFFERENT: {accuracy_results['different_accuracy']:.4f}")

def run_random_fc_ablations(model, test_loader, device, analyzer, results, num_random=10):
    """Run ablations on random FC neurons as controls."""
    
    print(f"\n--- Testing {num_random} Random FC Neurons as Controls ---")
    
    # Get available FC layer sizes (for conv2 model)
    layer_sizes = {
        'fc_layers.0': 1024,  # fc1
        'fc_layers.1': 1024,  # fc2
        'fc_layers.2': 1024   # fc3
    }
    
    random.seed(42)  # For reproducibility
    
    for i in tqdm(range(num_random), desc="Testing random FC neurons"):
        # Randomly select layer and neuron
        layer_name = random.choice(list(layer_sizes.keys()))
        neuron_idx = random.randint(0, layer_sizes[layer_name] - 1)
        
        accuracy_results = calculate_accuracy_by_class(
            model, test_loader, device, analyzer, layer_name, neuron_idx
        )
        
        results['random_fc_ablations'][f"random_{i}_{layer_name}_neuron_{neuron_idx}"] = {
            'layer': layer_name,
            'neuron': neuron_idx,
            'accuracy_results': accuracy_results
        }

def run_combined_fc_ablations(model, test_loader, device, analyzer, results):
    """Run ablations on combinations of top DIFFERENT neurons."""
    
    print("\n--- Testing Combined FC Ablations ---")
    
    # Test combinations of top DIFFERENT neurons
    combinations = [
        {'neurons': [289, 21], 'layer': 'fc_layers.0', 'description': 'Top 2 FC1 DIFFERENT neurons'},
        {'neurons': [289, 21, 413], 'layer': 'fc_layers.0', 'description': 'Top 3 FC1 DIFFERENT neurons'},
        {'neurons': [892, 308, 960], 'layer': 'fc_layers.2', 'description': 'Top 3 FC3 DIFFERENT neurons'},
    ]
    
    for combo in tqdm(combinations, desc="Testing FC combinations"):
        layer_name = combo['layer']
        neurons = combo['neurons']
        
        accuracy_results = calculate_accuracy_by_class(
            model, test_loader, device, analyzer, layer_name, neurons
        )
        
        results['combined_fc_ablations'][f"{layer_name}_neurons_{'_'.join(map(str, neurons))}"] = {
            'combination_info': combo,
            'accuracy_results': accuracy_results
        }
        
        print(f"{combo['description']}: "
              f"Overall: {accuracy_results['overall_accuracy']:.4f}, "
              f"SAME: {accuracy_results['same_accuracy']:.4f}, "
              f"DIFFERENT: {accuracy_results['different_accuracy']:.4f}")

def analyze_fc_results(results, baseline_results):
    """Analyze and summarize the FC ablation results."""
    
    print("\n" + "="*80)
    print("FC LAYER ABLATION RESULTS SUMMARY")
    print("="*80)
    
    # Baseline performance
    print(f"\nBaseline Performance:")
    print(f"  Overall: {baseline_results['overall_accuracy']:.4f}")
    print(f"  SAME: {baseline_results['same_accuracy']:.4f}")
    print(f"  DIFFERENT: {baseline_results['different_accuracy']:.4f}")
    
    # Analyze targeted ablations
    print(f"\nTargeted DIFFERENT-Selective FC Neuron Ablations:")
    targeted_effects = []
    for name, data in results['targeted_fc_ablations'].items():
        acc_results = data['accuracy_results']
        neuron_info = data['neuron_info']
        
        overall_drop = baseline_results['overall_accuracy'] - acc_results['overall_accuracy']
        same_drop = baseline_results['same_accuracy'] - acc_results['same_accuracy']
        diff_drop = baseline_results['different_accuracy'] - acc_results['different_accuracy']
        
        targeted_effects.append({
            'name': name,
            'overall_drop': overall_drop,
            'same_drop': same_drop,
            'diff_drop': diff_drop,
            'selectivity': neuron_info['selectivity']
        })
        
        print(f"  {name}: Overall drop: {overall_drop:.4f}, "
              f"SAME drop: {same_drop:.4f}, DIFFERENT drop: {diff_drop:.4f}")
    
    # Analyze random ablations
    print(f"\nRandom FC Neuron Ablations (Controls):")
    random_effects = []
    for name, data in results['random_fc_ablations'].items():
        acc_results = data['accuracy_results']
        
        overall_drop = baseline_results['overall_accuracy'] - acc_results['overall_accuracy']
        same_drop = baseline_results['same_accuracy'] - acc_results['same_accuracy']
        diff_drop = baseline_results['different_accuracy'] - acc_results['different_accuracy']
        
        random_effects.append({
            'name': name,
            'overall_drop': overall_drop,
            'same_drop': same_drop,
            'diff_drop': diff_drop
        })
    
    # Summary statistics for random ablations
    if random_effects:
        random_overall_drops = [effect['overall_drop'] for effect in random_effects]
        random_same_drops = [effect['same_drop'] for effect in random_effects]
        random_diff_drops = [effect['diff_drop'] for effect in random_effects]
        
        print(f"  Random FC ablations mean overall drop: {np.mean(random_overall_drops):.4f} ± {np.std(random_overall_drops):.4f}")
        print(f"  Random FC ablations mean SAME drop: {np.mean(random_same_drops):.4f} ± {np.std(random_same_drops):.4f}")
        print(f"  Random FC ablations mean DIFFERENT drop: {np.mean(random_diff_drops):.4f} ± {np.std(random_diff_drops):.4f}")
    
    # Analyze combined ablations
    print(f"\nCombined FC Neuron Ablations:")
    for name, data in results['combined_fc_ablations'].items():
        acc_results = data['accuracy_results']
        combo_info = data['combination_info']
        
        overall_drop = baseline_results['overall_accuracy'] - acc_results['overall_accuracy']
        same_drop = baseline_results['same_accuracy'] - acc_results['same_accuracy']
        diff_drop = baseline_results['different_accuracy'] - acc_results['different_accuracy']
        
        print(f"  {combo_info['description']}: Overall drop: {overall_drop:.4f}, "
              f"SAME drop: {same_drop:.4f}, DIFFERENT drop: {diff_drop:.4f}")
    
    # Statistical analysis
    print(f"\nStatistical Analysis:")
    targeted_overall_drops = [effect['overall_drop'] for effect in targeted_effects]
    
    if targeted_overall_drops:
        print(f"  Targeted FC neurons mean overall drop: {np.mean(targeted_overall_drops):.4f} ± {np.std(targeted_overall_drops):.4f}")
    
    # Compare targeted vs random
    if len(targeted_overall_drops) > 0 and len(random_overall_drops) > 0:
        t_stat, p_value = stats.ttest_ind(targeted_overall_drops, random_overall_drops)
        print(f"  T-test comparing targeted vs random FC drops: t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT: Targeted FC neurons cause significantly larger drops than random!")
        else:
            print(f"  ✗ Not significant: Targeted vs random FC effects are not significantly different")

def main():
    """Main function to run the comprehensive ablation experiment."""
    # --- 1. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # --- 2. Load Model ---
    # Pretrained PB meta-baseline conv2 model trained on full PB range
    model_path = '/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv2/seed_46/best_model.pt'
    model = SameDifferentCNN()
    
    # When loading a model saved with nn.DataParallel, the state dict keys are prefixed with 'module.'
    # We need to handle this to load the model correctly.
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    if next(iter(state_dict)).startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    print(f"Loaded trained model from {model_path}")

    # --- 3. Load Data ---
    data_dir = 'data/meta_h5/pb'
    # Use the full PB dataset: all 10 tasks and all standard support sizes
    tasks = PB_TASKS  # ['regular', 'lines', 'open', 'wider_line', 'scrambled', 'random_color', 'arrows', 'irregular', 'filled', 'original']
    test_dataset = SameDifferentDataset(data_dir, tasks, 'test', support_sizes=[4, 6, 8, 10])
    # Use smaller batch size to manage memory with 8K episodes
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_episodes)
    print(f"Loaded full PB test dataset with {len(tasks)} tasks and support sizes [4,6,8,10].")

    # --- 4. Initialize Analyzer ---
    analyzer = CircuitAnalyzer(model)
    
    # --- 5. Get Baseline Performance ---
    print("\n--- Computing Baseline Performance ---")
    baseline_results = calculate_accuracy_by_class(model, test_loader, device)
    
    print(f"Baseline Performance:")
    print(f"  Overall accuracy: {baseline_results['overall_accuracy']:.4f}")
    print(f"  SAME accuracy: {baseline_results['same_accuracy']:.4f}")
    print(f"  DIFFERENT accuracy: {baseline_results['different_accuracy']:.4f}")
    
    # --- 6. Initialize Results Structure ---
    results = {
        'baseline': baseline_results,
        'conv_ablations': {},
        'targeted_fc_ablations': {},
        'random_fc_ablations': {},
        'combined_fc_ablations': {},
        'model_path': model_path,
        'device': str(device)
    }

    # --- 7. Run Conv Layer Ablation Experiment ---
    layer_to_ablate = 'conv2'
    layer = analyzer._layer_map[layer_to_ablate]
    num_channels = layer.out_channels
    
    print(f"\n--- Starting Conv Layer Ablation Experiment on '{layer_to_ablate}' ---")

    # Now, ablate each channel and measure the accuracy drop
    print(f"Will test {num_channels} channels in {layer_to_ablate}")
    accuracy_drops = {}
    for channel_idx in tqdm(range(num_channels), desc=f"Ablating channels in {layer_to_ablate}"):
        ablated_accuracy = calculate_accuracy(model, test_loader, device, analyzer, layer_to_ablate, channel_idx)
        accuracy_drops[channel_idx] = baseline_results['overall_accuracy'] - ablated_accuracy
        
        # Log progress and save intermediate results every 50 channels
        if (channel_idx + 1) % 50 == 0:
            print(f"Completed {channel_idx + 1}/{num_channels} channels. Current drop: {accuracy_drops[channel_idx]:.4f}")
            
            # Save intermediate results
            intermediate_results = {
                'baseline_accuracy': baseline_results['overall_accuracy'],
                'accuracy_drops': accuracy_drops,
                'completed_channels': channel_idx + 1,
                'total_channels': num_channels,
                'layer': layer_to_ablate
            }
            with open(f'intermediate_ablation_results_{channel_idx + 1}.json', 'w') as f:
                json.dump(intermediate_results, f, indent=2)

    # Store conv results
    results['conv_ablations'] = {
        'layer': layer_to_ablate,
        'accuracy_drops': accuracy_drops,
        'num_channels': num_channels
    }

    # --- 8. Run FC Layer Ablation Experiments ---
    run_targeted_fc_ablations(model, test_loader, device, analyzer, results)
    run_random_fc_ablations(model, test_loader, device, analyzer, results, num_random=10)
    run_combined_fc_ablations(model, test_loader, device, analyzer, results)
    
    # --- 9. Analyze FC Results ---
    analyze_fc_results(results, baseline_results)

    # --- 10. Report Conv Results ---
    print("\n--- Ablation Results ---")
    # Sort channels by the magnitude of accuracy drop
    sorted_channels = sorted(accuracy_drops.items(), key=lambda item: item[1], reverse=True)

    print(f"Top 5 most critical channels in '{layer_to_ablate}' for task 'full PB':")
    for i in range(min(5, len(sorted_channels))):
        channel_idx, drop = sorted_channels[i]
        print(f"  - Channel {channel_idx}: Accuracy drop of {drop:.4f}")

    # --- 11. Save Final Results ---
    final_results = {
        'baseline_results': baseline_results,
        'conv_results': {
            'accuracy_drops': accuracy_drops,
            'sorted_channels': sorted_channels,
            'layer_ablated': layer_to_ablate,
            'total_channels': num_channels
        },
        'fc_results': {
            'targeted_ablations': results['targeted_fc_ablations'],
            'random_ablations': results['random_fc_ablations'],
            'combined_ablations': results['combined_fc_ablations']
        },
        'model_path': model_path,
        'dataset_info': {
            'tasks': tasks,
            'support_sizes': [4, 6, 8, 10],
            'total_episodes': len(test_dataset)
        }
    }
    
    final_filename = f'comprehensive_ablation_results_conv2_and_fc.json'
    with open(final_filename, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nFinal results saved to {final_filename}")

if __name__ == '__main__':
    main() 