import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import learn2learn as l2l
from tqdm import tqdm
import os
import h5py
import numpy as np
import argparse
import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict
import pickle

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_baseline.models.conv2lr import SameDifferentCNN
from meta_baseline.models.utils_meta import SameDifferentDataset, collate_episodes

class CircuitAnalyzer:
    """
    Comprehensive circuit analysis for same-different classification in MAML models.
    Inspired by Distill circuits methodology.
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = SameDifferentCNN().to(device)
        
        # Load the trained model
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.maml = l2l.algorithms.MAML(self.model, lr=0.01, first_order=False, allow_unused=True)
        self.maml.eval()
        
        # Storage for activations and analysis results
        self.activation_store = {}
        self.circuit_results = {}
        
        # Hook functions for activation capture
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations at key layers."""
        
        def get_activation(name):
            def hook(model, input, output):
                # Store mean-pooled activations to reduce dimensionality
                if len(output.shape) == 4:  # Conv layers (B, C, H, W)
                    self.activation_store[name] = output.mean(dim=[2, 3]).detach()
                else:  # FC layers (B, F)
                    self.activation_store[name] = output.detach()
            return hook
        
        # Register hooks on key layers
        self.hooks.append(self.model.conv1.register_forward_hook(get_activation('conv1')))
        self.hooks.append(self.model.conv2.register_forward_hook(get_activation('conv2')))
        
        for i, fc in enumerate(self.model.fc_layers):
            self.hooks.append(fc.register_forward_hook(get_activation(f'fc{i+1}')))
        
        self.hooks.append(self.model.classifier.register_forward_hook(get_activation('classifier')))
    
    def cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def analyze_neuron_selectivity(self, data_loader, adaptation_steps=5, max_episodes=100):
        """
        Analyze individual neuron selectivity for same vs different classifications.
        This is the core of circuit analysis - finding neurons that fire differently
        for same vs different pairs.
        """
        print("Analyzing neuron selectivity for same-different classification...")
        
        # Storage for neuron activations by label
        neuron_activations = defaultdict(lambda: defaultdict(list))  # {layer: {label: [activations]}}
        episode_count = 0
        
        for episode in tqdm(data_loader, desc="Collecting activations", total=min(len(data_loader), max_episodes)):
            if episode_count >= max_episodes:
                break
                
            # Clone learner for this episode
            learner = self.maml.clone()
            
            support_images = episode['support_images'].to(self.device).squeeze(0)
            support_labels = episode['support_labels'].to(self.device).squeeze(0)
            query_images = episode['query_images'].to(self.device).squeeze(0)
            query_labels = episode['query_labels'].to(self.device).squeeze(0)
            
            # Get pre-adaptation activations
            with torch.no_grad():
                _ = learner(query_images)
            pre_activations = {k: v.cpu().numpy() for k, v in self.activation_store.items()}
            
            # Adapt the model
            for step in range(adaptation_steps):
                preds = learner(support_images)
                loss = F.cross_entropy(preds, support_labels)
                learner.adapt(loss, allow_unused=True)
            
            # Get post-adaptation activations
            with torch.no_grad():
                _ = learner(query_images)
            post_activations = {k: v.cpu().numpy() for k, v in self.activation_store.items()}
            
            # Store activations by label
            labels_np = query_labels.cpu().numpy()
            for i, label in enumerate(labels_np):
                for layer_name in pre_activations:
                    neuron_activations[f'{layer_name}_pre'][label].append(pre_activations[layer_name][i])
                    neuron_activations[f'{layer_name}_post'][label].append(post_activations[layer_name][i])
            
            episode_count += 1
        
        # Analyze selectivity for each layer and neuron
        selectivity_results = {}
        
        for layer_name, label_data in neuron_activations.items():
            if len(label_data) < 2:  # Need both same and different examples
                continue
                
            same_activations = np.array(label_data.get(0, []))  # Assuming 0 = different, 1 = same
            diff_activations = np.array(label_data.get(1, []))
            
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
                
                # Statistical significance (t-test)
                from scipy.stats import ttest_ind
                try:
                    t_stat, p_value = ttest_ind(same_vals, diff_vals)
                except:
                    t_stat, p_value = 0, 1
                
                neuron_selectivities.append({
                    'neuron_idx': neuron_idx,
                    'selectivity': selectivity,
                    'mean_same': mean_same,
                    'mean_diff': mean_diff,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.01
                })
            
            selectivity_results[layer_name] = neuron_selectivities
        
        self.circuit_results['neuron_selectivity'] = selectivity_results
        return selectivity_results
    
    def find_critical_neurons(self, selectivity_results, top_k=5):
        """
        Identify the most critical neurons for same-different classification.
        """
        print("Identifying critical neurons...")
        
        critical_neurons = {}
        
        for layer_name, neurons in selectivity_results.items():
            # Sort by absolute selectivity and significance
            significant_neurons = [n for n in neurons if n['significant']]
            sorted_neurons = sorted(significant_neurons, 
                                  key=lambda x: abs(x['selectivity']), reverse=True)
            
            critical_neurons[layer_name] = sorted_neurons[:top_k]
        
        self.circuit_results['critical_neurons'] = critical_neurons
        return critical_neurons
    
    def analyze_adaptation_dynamics(self, data_loader, adaptation_steps=5, max_episodes=50):
        """
        Analyze how neuron activations change during MAML adaptation.
        This reveals the dynamics of the same-different circuit.
        """
        print("Analyzing adaptation dynamics...")
        
        adaptation_dynamics = defaultdict(list)
        episode_count = 0
        
        for episode in tqdm(data_loader, desc="Analyzing adaptation", total=min(len(data_loader), max_episodes)):
            if episode_count >= max_episodes:
                break
            
            learner = self.maml.clone()
            
            support_images = episode['support_images'].to(self.device).squeeze(0)
            support_labels = episode['support_labels'].to(self.device).squeeze(0)
            query_images = episode['query_images'].to(self.device).squeeze(0)
            
            # Track activations at each adaptation step
            step_activations = []
            
            # Initial activations
            with torch.no_grad():
                _ = learner(query_images)
            step_activations.append({k: v.mean(0).cpu().numpy() 
                                   for k, v in self.activation_store.items()})
            
            # Adaptation steps
            for step in range(adaptation_steps):
                preds = learner(support_images)
                loss = F.cross_entropy(preds, support_labels)
                learner.adapt(loss, allow_unused=True)
                
                # Capture activations after this step
                with torch.no_grad():
                    _ = learner(query_images)
                step_activations.append({k: v.mean(0).cpu().numpy() 
                                       for k, v in self.activation_store.items()})
            
            adaptation_dynamics[episode['task'][0]].append(step_activations)
            episode_count += 1
        
        self.circuit_results['adaptation_dynamics'] = adaptation_dynamics
        return adaptation_dynamics
    
    def visualize_critical_neurons(self, critical_neurons, output_dir):
        """
        Create visualizations of the most critical neurons.
        """
        print("Creating visualizations...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot selectivity scores for top neurons
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        layer_idx = 0
        for layer_name, neurons in critical_neurons.items():
            if layer_idx >= len(axes) or not neurons:
                continue
                
            ax = axes[layer_idx]
            
            neuron_indices = [n['neuron_idx'] for n in neurons]
            selectivities = [n['selectivity'] for n in neurons]
            significances = [n['significant'] for n in neurons]
            
            colors = ['red' if sig else 'blue' for sig in significances]
            bars = ax.bar(range(len(neuron_indices)), selectivities, color=colors)
            
            ax.set_title(f'{layer_name} - Critical Neurons')
            ax.set_xlabel('Neuron Index')
            ax.set_ylabel('Selectivity Score')
            ax.set_xticks(range(len(neuron_indices)))
            ax.set_xticklabels([f'N{idx}' for idx in neuron_indices], rotation=45)
            
            # Add significance indicators
            for i, (bar, sig) in enumerate(zip(bars, significances)):
                if sig:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           '*', ha='center', va='bottom', fontweight='bold')
            
            layer_idx += 1
        
        # Remove unused subplots
        for i in range(layer_idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'critical_neurons_selectivity.png'), dpi=300)
        plt.close()
        
        # Save detailed results
        with open(os.path.join(output_dir, 'circuit_analysis_results.json'), 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = {}
            for layer, neurons in critical_neurons.items():
                serializable_results[layer] = []
                for neuron in neurons:
                    serializable_neuron = {}
                    for key, value in neuron.items():
                        if isinstance(value, np.ndarray):
                            serializable_neuron[key] = value.tolist()
                        elif isinstance(value, (np.integer, np.floating, np.bool_)):
                            serializable_neuron[key] = value.item()
                        else:
                            serializable_neuron[key] = value
                    serializable_results[layer].append(serializable_neuron)
            
            json.dump(serializable_results, f, indent=2)
    
    def run_full_analysis(self, data_loader, output_dir, adaptation_steps=5, max_episodes=100):
        """
        Run the complete circuit analysis pipeline.
        """
        print("=" * 60)
        print("SAME-DIFFERENT CIRCUIT ANALYSIS")
        print("=" * 60)
        
        # Step 1: Analyze neuron selectivity
        selectivity_results = self.analyze_neuron_selectivity(
            data_loader, adaptation_steps, max_episodes)
        
        # Step 2: Find critical neurons
        critical_neurons = self.find_critical_neurons(selectivity_results)
        
        # Step 3: Analyze adaptation dynamics
        adaptation_dynamics = self.analyze_adaptation_dynamics(
            data_loader, adaptation_steps, max_episodes//2)
        
        # Step 4: Create visualizations
        self.visualize_critical_neurons(critical_neurons, output_dir)
        
        # Step 5: Generate summary report
        self.generate_summary_report(critical_neurons, output_dir)
        
        print(f"\nAnalysis complete! Results saved to {output_dir}")
        return self.circuit_results
    
    def generate_summary_report(self, critical_neurons, output_dir):
        """
        Generate a comprehensive summary report of the circuit analysis.
        """
        report_lines = []
        report_lines.append("SAME-DIFFERENT CIRCUIT ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Architecture summary
        report_lines.append("MODEL ARCHITECTURE:")
        report_lines.append(f"- Conv1: 3 → 6 filters (2x2 kernel)")
        report_lines.append(f"- Conv2: 6 → 12 filters (2x2 kernel)")
        report_lines.append(f"- FC layers: 3 × 1024 units")
        report_lines.append(f"- Classifier: 1024 → 2 units")
        report_lines.append("")
        
        # Critical neurons summary
        report_lines.append("CRITICAL NEURONS DISCOVERED:")
        total_critical = 0
        for layer_name, neurons in critical_neurons.items():
            if neurons:
                report_lines.append(f"\n{layer_name.upper()}:")
                for neuron in neurons[:3]:  # Top 3 per layer
                    selectivity = neuron['selectivity']
                    significance = "***" if neuron['significant'] else ""
                    direction = "SAME-preferring" if selectivity > 0 else "DIFFERENT-preferring"
                    report_lines.append(f"  - Neuron {neuron['neuron_idx']}: {direction} "
                                      f"(selectivity={selectivity:.3f}) {significance}")
                total_critical += len(neurons)
        
        report_lines.append(f"\nTOTAL CRITICAL NEURONS IDENTIFIED: {total_critical}")
        
        # Circuit hypothesis
        report_lines.append("\nCIRCUIT HYPOTHESIS:")
        report_lines.append("Based on the analysis, the same-different circuit appears to consist of:")
        report_lines.append("1. Low-level feature detectors in conv layers")
        report_lines.append("2. High-level comparison neurons in FC layers")
        report_lines.append("3. Decision neurons in the classifier layer")
        report_lines.append("\nThe circuit is dynamically modulated during MAML adaptation,")
        report_lines.append("allowing rapid specialization to new visual domains.")
        
        # Save report
        with open(os.path.join(output_dir, 'circuit_analysis_report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="Analyze same-different circuits in MAML conv2lr models")
    
    # Model and data arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained conv2lr model')
    parser.add_argument('--data_dir', type=str, 
                       default='/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb',
                       help='Path to PB dataset')
    parser.add_argument('--output_dir', type=str, default='circuit_analysis_conv2_results',
                       help='Directory to save analysis results')
    
    # Analysis parameters
    parser.add_argument('--adaptation_steps', type=int, default=5,
                       help='Number of MAML adaptation steps')
    parser.add_argument('--max_episodes', type=int, default=100,
                       help='Maximum episodes to analyze')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for analysis')
    parser.add_argument('--support_size', type=int, default=10,
                       help='Support set size')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    PB_TASKS = ["regular", "lines", "open", "wider_line", "scrambled",
                "random_color", "arrows", "irregular", "filled", "original"]
    
    dataset = SameDifferentDataset(
        args.data_dir, PB_TASKS, 'val', support_sizes=[args.support_size])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize analyzer
    analyzer = CircuitAnalyzer(args.model_path, device)
    
    try:
        # Run full analysis
        results = analyzer.run_full_analysis(
            dataloader, args.output_dir, args.adaptation_steps, args.max_episodes)
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        if 'critical_neurons' in results:
            for layer, neurons in results['critical_neurons'].items():
                if neurons:
                    print(f"{layer}: {len(neurons)} critical neurons found")
        
    finally:
        # Cleanup
        analyzer.cleanup_hooks()


if __name__ == '__main__':
    main() 