import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import ttest_ind
import learn2learn as l2l

# Add the root directory to the path to allow imports from other directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the conv2lr model architecture
from meta_baseline.models.conv2lr import SameDifferentCNN
from meta_baseline.models.utils_meta import SameDifferentDataset, collate_episodes
from circuit_analysis.analyzer import CircuitAnalyzer

class SelectivityAnalyzer:
    """Class to analyze neuron selectivity for SAME vs DIFFERENT responses"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activation_store = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture activations"""
        def get_activation(name):
            def hook(model, input, output):
                # Store mean-pooled activations to reduce dimensionality
                if len(output.shape) == 4:  # Conv layers (B, C, H, W)
                    self.activation_store[name] = output.mean(dim=(2, 3)).detach()
                else:  # FC layers (B, C)
                    self.activation_store[name] = output.detach()
            return hook
        
        # Register hooks for all layers
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)) and not list(layer.children()):
                layer.register_forward_hook(get_activation(name))
    
    def analyze_selectivity(self, data_loader, max_batches=50):
        """
        Analyze neuron selectivity for SAME vs DIFFERENT responses
        Returns neurons sorted by selectivity (negative = DIFFERENT-preferring)
        """
        print("Analyzing neuron selectivity...")
        
        # Collect activations by label
        activations_by_label = defaultdict(lambda: defaultdict(list))
        batch_count = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Collecting activations", total=min(len(data_loader), max_batches)):
                if batch_count >= max_batches:
                    break
                
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass to get activations
                _ = self.model(images)
                
                # Store activations by label
                labels_np = labels.cpu().numpy()
                for i, label in enumerate(labels_np):
                    for layer_name in self.activation_store:
                        activation = self.activation_store[layer_name][i].cpu().numpy()
                        activations_by_label[layer_name][label].append(activation)
                
                batch_count += 1
        
        # Analyze selectivity for each layer
        selectivity_results = {}
        
        for layer_name, label_data in activations_by_label.items():
            if len(label_data) < 2:  # Need both labels
                continue
            
            same_activations = np.array(label_data.get(1, []))  # Label 1 = SAME
            diff_activations = np.array(label_data.get(0, []))  # Label 0 = DIFFERENT
            
            if len(same_activations) == 0 or len(diff_activations) == 0:
                continue
            
            # Calculate selectivity for each neuron
            num_neurons = same_activations.shape[1] if len(same_activations.shape) > 1 else 1
            neuron_selectivities = []
            
            for neuron_idx in range(num_neurons):
                if len(same_activations.shape) > 1:
                    same_vals = same_activations[:, neuron_idx]
                    diff_vals = diff_activations[:, neuron_idx]
                else:
                    same_vals = same_activations
                    diff_vals = diff_activations
                
                # Calculate selectivity index: (mean_same - mean_diff) / (mean_same + mean_diff)
                mean_same = np.mean(same_vals)
                mean_diff = np.mean(diff_vals)
                
                if (mean_same + mean_diff) > 0:
                    selectivity = (mean_same - mean_diff) / (mean_same + mean_diff)
                else:
                    selectivity = 0
                
                # Statistical significance test
                try:
                    t_stat, p_value = ttest_ind(same_vals, diff_vals)
                except:
                    t_stat, p_value = 0, 1
                
                neuron_selectivities.append({
                    'layer_name': layer_name,
                    'neuron_idx': neuron_idx,
                    'selectivity': selectivity,
                    'mean_same': mean_same,
                    'mean_diff': mean_diff,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.01
                })
            
            selectivity_results[layer_name] = neuron_selectivities
        
        return selectivity_results
    
    def find_top_different_neurons(self, selectivity_results, top_k=3):
        """Find the top K neurons most selective for DIFFERENT responses"""
        all_neurons = []
        
        for layer_name, neurons in selectivity_results.items():
            # Filter for significant neurons with negative selectivity (DIFFERENT-preferring)
            diff_neurons = [n for n in neurons if n['significant'] and n['selectivity'] < 0]
            all_neurons.extend(diff_neurons)
        
        # Sort by selectivity (most negative = most DIFFERENT-selective)
        all_neurons.sort(key=lambda x: x['selectivity'])
        
        return all_neurons[:top_k]

def train_quick_model(device, data_dir, output_dir, epochs=20):
    """Train a quick conv2lr model locally for ablation testing"""
    print(f"Training a quick conv2lr model locally...")
    
    # Create model
    model = SameDifferentCNN().to(device)
    
    # Create MAML wrapper
    maml = l2l.algorithms.MAML(
        model,
        lr=0.05,
        first_order=False,
        allow_unused=True,
        allow_nograd=True
    )
    
    optimizer = torch.optim.Adam(maml.parameters(), lr=0.001)
    
    # Create datasets - use fewer tasks for quick training
    train_tasks = ['regular', 'lines', 'filled']  # Just 3 tasks for quick training
    train_dataset = SameDifferentDataset(data_dir, train_tasks, 'train', support_sizes=[4])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, 
                            num_workers=2, pin_memory=True, collate_fn=collate_episodes)
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            optimizer.zero_grad()
            
            learner = maml.clone()
            
            # Get support and query data
            support_images = batch['support_images'].to(device).squeeze(0)
            support_labels = batch['support_labels'].to(device).squeeze(0)
            query_images = batch['query_images'].to(device).squeeze(0)
            query_labels = batch['query_labels'].to(device).squeeze(0)
            
            # Adaptation steps
            for _ in range(3):  # 3 adaptation steps
                preds = learner(support_images)
                loss = torch.nn.functional.cross_entropy(preds, support_labels)
                learner.adapt(loss, allow_unused=True)
            
            # Query loss
            query_preds = learner(query_images)
            query_loss = torch.nn.functional.cross_entropy(query_preds, query_labels)
            
            query_loss.backward()
            optimizer.step()
            
            total_loss += query_loss.item()
            num_batches += 1
            
            if num_batches >= 50:  # Limit batches per epoch for quick training
                break
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'quick_trained_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
        'architecture': 'conv2lr'
    }, model_path)
    
    print(f"Quick model saved to {model_path}")
    return model_path

def calculate_accuracy(model, data_loader, device, analyzer=None, ablation_spec=None):
    """Calculate model accuracy with optional ablation"""
    model.eval()
    correct = 0
    total = 0
    
    # Apply ablation if specified
    if analyzer and ablation_spec:
        if ablation_spec['type'] == 'single':
            analyzer.ablate_layer(ablation_spec['layer'], ablation_spec['neuron'])
        elif ablation_spec['type'] == 'multiple':
            for layer, neuron in ablation_spec['neurons']:
                analyzer.ablate_layer(layer, neuron)
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
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

def main():
    """Main function to run the enhanced ablation experiment"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU. This will be slow.")
    print(f"Using device: {device}")
    
    # Local data directories
    data_dir = 'data/meta_h5/pb'
    output_dir = 'results/local_ablation_experiment'
    
    # Check if we have a trained model, if not train one quickly
    model_path = os.path.join(output_dir, 'quick_trained_model.pt')
    
    if not os.path.exists(model_path):
        print("No trained model found. Training a quick model locally...")
        model_path = train_quick_model(device, data_dir, output_dir, epochs=10)
    else:
        print(f"Using existing trained model: {model_path}")
    
    # Load the trained model
    model = SameDifferentCNN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")
    
    # Create test dataset - use regular task for testing
    test_tasks = ['regular']
    test_dataset = SameDifferentDataset(data_dir, test_tasks, 'test', support_sizes=[4])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"Loaded test dataset with {len(test_dataset)} examples")
    
    # Step 1: Analyze selectivity to find DIFFERENT-selective neurons
    print("\n=== STEP 1: SELECTIVITY ANALYSIS ===")
    selectivity_analyzer = SelectivityAnalyzer(model, device)
    selectivity_results = selectivity_analyzer.analyze_selectivity(test_loader, max_batches=20)
    
    # Find top DIFFERENT-selective neurons
    top_different_neurons = selectivity_analyzer.find_top_different_neurons(selectivity_results, top_k=3)
    
    print(f"\nTop 3 DIFFERENT-selective neurons:")
    for i, neuron in enumerate(top_different_neurons):
        print(f"  {i+1}. {neuron['layer_name']} neuron {neuron['neuron_idx']}: "
              f"selectivity={neuron['selectivity']:.4f}, p={neuron['p_value']:.4f}")
    
    if len(top_different_neurons) == 0:
        print("No statistically significant DIFFERENT-selective neurons found.")
        print("This might be due to:")
        print("1. Limited training (only 10 epochs)")
        print("2. Random initialization dominating")
        print("3. Need more training data")
        print("\nProceeding with most selective neurons regardless of significance...")
        
        # Get top 3 most DIFFERENT-selective neurons regardless of significance
        all_neurons = []
        for layer_name, neurons in selectivity_results.items():
            diff_neurons = [n for n in neurons if n['selectivity'] < 0]
            all_neurons.extend(diff_neurons)
        
        all_neurons.sort(key=lambda x: x['selectivity'])
        top_different_neurons = all_neurons[:3]
        
        print(f"\nTop 3 most DIFFERENT-selective neurons (regardless of significance):")
        for i, neuron in enumerate(top_different_neurons):
            print(f"  {i+1}. {neuron['layer_name']} neuron {neuron['neuron_idx']}: "
                  f"selectivity={neuron['selectivity']:.4f}, p={neuron['p_value']:.4f}")
    
    # Step 2: Baseline accuracy
    print("\n=== STEP 2: BASELINE ACCURACY ===")
    baseline_accuracy = calculate_accuracy(model, test_loader, device)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Step 3: Targeted ablation of DIFFERENT-selective neurons
    print("\n=== STEP 3: TARGETED ABLATION ===")
    analyzer = CircuitAnalyzer(model)
    
    # Test individual ablations
    individual_results = {}
    for i, neuron in enumerate(top_different_neurons):
        ablation_spec = {
            'type': 'single',
            'layer': neuron['layer_name'],
            'neuron': neuron['neuron_idx']
        }
        
        ablated_accuracy = calculate_accuracy(model, test_loader, device, analyzer, ablation_spec)
        accuracy_drop = baseline_accuracy - ablated_accuracy
        individual_results[i] = {
            'neuron': neuron,
            'ablated_accuracy': ablated_accuracy,
            'accuracy_drop': accuracy_drop
        }
        
        print(f"Ablating {neuron['layer_name']} neuron {neuron['neuron_idx']}: "
              f"accuracy={ablated_accuracy:.4f}, drop={accuracy_drop:.4f}")
    
    # Test combined ablation of top 2-3 neurons
    print("\nTesting combined ablation of top DIFFERENT-selective neurons:")
    for k in [2, 3]:
        if k <= len(top_different_neurons):
            neurons_to_ablate = [(n['layer_name'], n['neuron_idx']) for n in top_different_neurons[:k]]
            ablation_spec = {
                'type': 'multiple',
                'neurons': neurons_to_ablate
            }
            
            combined_accuracy = calculate_accuracy(model, test_loader, device, analyzer, ablation_spec)
            combined_drop = baseline_accuracy - combined_accuracy
            
            print(f"Combined ablation of top {k} neurons: "
                  f"accuracy={combined_accuracy:.4f}, drop={combined_drop:.4f}")
    
    # Step 4: Random ablation controls
    print("\n=== STEP 4: RANDOM ABLATION CONTROLS ===")
    
    # Get all available neurons for random sampling
    all_neurons = []
    for layer_name, neurons in selectivity_results.items():
        for neuron in neurons:
            all_neurons.append((layer_name, neuron['neuron_idx']))
    
    # Test random ablations
    random_results = []
    num_random_tests = 10
    
    print(f"Testing {num_random_tests} random ablations...")
    for i in range(num_random_tests):
        # Sample 3 random neurons
        random_neurons = random.sample(all_neurons, min(3, len(all_neurons)))
        
        ablation_spec = {
            'type': 'multiple',
            'neurons': random_neurons
        }
        
        random_accuracy = calculate_accuracy(model, test_loader, device, analyzer, ablation_spec)
        random_drop = baseline_accuracy - random_accuracy
        random_results.append(random_drop)
        
        print(f"Random ablation {i+1}: accuracy={random_accuracy:.4f}, drop={random_drop:.4f}")
    
    # Step 5: Statistical comparison
    print("\n=== STEP 5: STATISTICAL COMPARISON ===")
    
    # Compare targeted vs random ablations
    targeted_drops = [res['accuracy_drop'] for res in individual_results.values()]
    mean_targeted_drop = np.mean(targeted_drops)
    mean_random_drop = np.mean(random_results)
    std_random_drop = np.std(random_results)
    
    print(f"Mean targeted ablation drop: {mean_targeted_drop:.4f}")
    print(f"Mean random ablation drop: {mean_random_drop:.4f} ± {std_random_drop:.4f}")
    print(f"Difference: {mean_targeted_drop - mean_random_drop:.4f}")
    
    # Statistical significance test
    try:
        t_stat, p_value = ttest_ind(targeted_drops, random_results)
        print(f"T-test p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("*** SIGNIFICANT: Targeted ablations cause significantly more damage than random!")
        else:
            print("Not significant: Targeted ablations not significantly different from random")
    except:
        print("Could not compute statistical test")
    
    # Step 6: Save results
    print("\n=== STEP 6: SAVING RESULTS ===")
    
    results = {
        'baseline_accuracy': baseline_accuracy,
        'top_different_neurons': top_different_neurons,
        'individual_ablation_results': individual_results,
        'random_ablation_results': random_results,
        'statistical_comparison': {
            'mean_targeted_drop': mean_targeted_drop,
            'mean_random_drop': mean_random_drop,
            'std_random_drop': std_random_drop,
            'statistical_test': {
                't_stat': t_stat if 't_stat' in locals() else None,
                'p_value': p_value if 'p_value' in locals() else None
            }
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'enhanced_ablation_results.json'), 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = value.item()
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Results saved to {os.path.join(output_dir, 'enhanced_ablation_results.json')}")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"✓ Trained local conv2lr model (quick training)")
    print(f"✓ Identified {len(top_different_neurons)} top DIFFERENT-selective neurons")
    print(f"✓ Tested targeted ablations vs {num_random_tests} random controls")
    print(f"✓ Targeted ablations caused {mean_targeted_drop:.4f} average accuracy drop")
    print(f"✓ Random ablations caused {mean_random_drop:.4f} average accuracy drop")
    if 't_stat' in locals() and 'p_value' in locals() and p_value < 0.05:
        print("✓ CAUSAL EVIDENCE: Targeted ablations significantly more damaging than random!")
    else:
        print("⚠ No significant causal evidence found (may need more training)")

if __name__ == '__main__':
    main() 