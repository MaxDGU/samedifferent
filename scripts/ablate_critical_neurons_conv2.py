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
import pandas as pd
from collections import defaultdict

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meta_baseline.models.conv2lr import SameDifferentCNN
from meta_baseline.models.utils_meta import SameDifferentDataset

class NeuronAblator:
    """
    Targeted ablation experiment for testing the causal role of critical neurons
    in same-different classification.
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = SameDifferentCNN().to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.maml = l2l.algorithms.MAML(self.model, lr=0.01, first_order=False, allow_unused=True)
        self.maml.eval()
        self.ablation_hooks = []
    
    def _create_ablation_hook(self, neuron_indices):
        """Creates a hook to zero out a list of neurons."""
        def hook(module, input, output):
            # Zero out all specified neuron indices
            output[:, neuron_indices] = 0
            return output
        return hook
    
    def _register_ablation_hook(self, layer_name, neuron_indices):
        """Registers an ablation hook on the specified layer for a list of neurons."""
        if layer_name == 'fc1':
            target_layer = self.model.fc_layers[0]
        else:
            raise ValueError(f"Ablation for layer '{layer_name}' not implemented.")
        
        # Ensure neuron_indices is a list
        if not isinstance(neuron_indices, list):
            neuron_indices = [neuron_indices]
            
        hook = target_layer.register_forward_hook(self._create_ablation_hook(neuron_indices))
        self.ablation_hooks.append(hook)
    
    def _cleanup_ablation_hooks(self):
        for hook in self.ablation_hooks:
            hook.remove()
        self.ablation_hooks = []
    
    def evaluate(self, data_loader, adaptation_steps=5, ablated_layer=None, ablated_neurons=None, max_episodes=100):
        if ablated_layer and ablated_neurons is not None:
            print(f"INFO: Ablating {ablated_layer} neurons {ablated_neurons}")
            self._register_ablation_hook(ablated_layer, ablated_neurons)
        
        label_results = {0: {'correct': 0, 'total': 0}, 1: {'correct': 0, 'total': 0}}
        
        for episode_count, episode in enumerate(tqdm(data_loader, desc="Evaluating", total=min(len(data_loader), max_episodes))):
            if episode_count >= max_episodes:
                break
                
            learner = self.maml.clone()
            
            support_images = episode['support_images'].to(self.device).squeeze(0)
            support_labels = episode['support_labels'].to(self.device).squeeze(0)
            query_images = episode['query_images'].to(self.device).squeeze(0)
            query_labels = episode['query_labels'].to(self.device).squeeze(0)
            
            for step in range(adaptation_steps):
                preds = learner(support_images)
                loss = F.cross_entropy(preds, support_labels)
                learner.adapt(loss, allow_unused=True)
            
            with torch.no_grad():
                query_preds = learner(query_images)
                predicted_labels = query_preds.argmax(dim=1)
                
                for true_label, pred_label in zip(query_labels.cpu().numpy(), predicted_labels.cpu().numpy()):
                    label_results[true_label]['total'] += 1
                    if true_label == pred_label:
                        label_results[true_label]['correct'] += 1
        
        self._cleanup_ablation_hooks()
        
        total_samples = label_results[0]['total'] + label_results[1]['total']
        total_correct = label_results[0]['correct'] + label_results[1]['correct']

        return {
            'overall_accuracy': (total_correct / total_samples) * 100 if total_samples > 0 else 0,
            'same_accuracy': (label_results[1]['correct'] / label_results[1]['total']) * 100 if label_results[1]['total'] > 0 else 0,
            'different_accuracy': (label_results[0]['correct'] / label_results[0]['total']) * 100 if label_results[0]['total'] > 0 else 0,
        }
    
def create_visualizations(results, output_dir):
    conditions = [
        'Baseline', 
        'Ablate SAME\nNeuron 590', 
        'Ablate DIFFERENT\nNeuron 289',
        'Ablate Top 3 SAME\nNeurons',
        'Ablate Top 2 DIFFERENT\nNeurons'
    ]
    same_accs = [
        results['baseline']['same_accuracy'],
        results['same_neuron_ablated']['same_accuracy'],
        results['different_neuron_ablated']['same_accuracy'],
        results['top_3_same_ablated']['same_accuracy'],
        results['top_2_diff_ablated']['same_accuracy']
    ]
    diff_accs = [
        results['baseline']['different_accuracy'],
        results['same_neuron_ablated']['different_accuracy'],
        results['different_neuron_ablated']['different_accuracy'],
        results['top_3_same_ablated']['different_accuracy'],
        results['top_2_diff_ablated']['different_accuracy']
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    x = np.arange(len(conditions))

    colors = ['#1f77b4', '#d62728', '#ff7f0e', '#9467bd', '#2ca02c']
    for ax, accs, title in [(ax1, same_accs, 'Impact on SAME Accuracy'), (ax2, diff_accs, 'Impact on DIFFERENT Accuracy')]:
        bars = ax.bar(x, accs, color=colors, alpha=0.8)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=10, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_accuracy_comparison.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Ablate critical neurons in MAML conv2lr models")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained conv2lr model')
    parser.add_argument('--data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb', help='Path to PB dataset')
    parser.add_argument('--output_dir', type=str, default='ablation_results_conv2', help='Directory to save results')
    parser.add_argument('--adaptation_steps', type=int, default=5, help='Number of MAML adaptation steps')
    parser.add_argument('--max_episodes', type=int, default=200, help='Maximum episodes to test for robust stats')
    parser.add_argument('--support_size', type=int, default=10, help='Support set size')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    PB_TASKS = ["regular", "lines", "open", "wider_line", "scrambled", "random_color", "arrows", "irregular", "filled", "original"]
    dataset = SameDifferentDataset(args.data_dir, PB_TASKS, 'val', support_sizes=[args.support_size])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    ablator = NeuronAblator(args.model_path, device)
    results = {}
    
    print("\n--- Running Baseline Evaluation (No Ablation) ---")
    results['baseline'] = ablator.evaluate(dataloader, args.adaptation_steps, max_episodes=args.max_episodes)
    
    print("\n--- Running Ablation on SAME-preferring Neuron 590 (fc1) ---")
    results['same_neuron_ablated'] = ablator.evaluate(dataloader, args.adaptation_steps, ablated_layer='fc1', ablated_neurons=590, max_episodes=args.max_episodes)
    
    print("\n--- Running Ablation on DIFFERENT-preferring Neuron 289 (fc1) ---")
    results['different_neuron_ablated'] = ablator.evaluate(dataloader, args.adaptation_steps, ablated_layer='fc1', ablated_neurons=289, max_episodes=args.max_episodes)
    
    print("\n--- Running Group Ablation on Top 3 SAME-preferring Neurons (fc1) ---")
    results['top_3_same_ablated'] = ablator.evaluate(dataloader, args.adaptation_steps, ablated_layer='fc1', ablated_neurons=[590, 691, 690], max_episodes=args.max_episodes)
    
    print("\n--- Running Group Ablation on Top 2 DIFFERENT-preferring Neurons (fc1) ---")
    results['top_2_diff_ablated'] = ablator.evaluate(dataloader, args.adaptation_steps, ablated_layer='fc1', ablated_neurons=[289, 21], max_episodes=args.max_episodes)
    
    # --- Report and Save ---
    print("\n" + "="*60 + "\nABLATION EXPERIMENT SUMMARY\n" + "="*60)
    df = pd.DataFrame(results).T
    df['overall_impact'] = df['overall_accuracy'] - df.loc['baseline', 'overall_accuracy']
    df['same_impact'] = df['same_accuracy'] - df.loc['baseline', 'same_accuracy']
    df['different_impact'] = df['different_accuracy'] - df.loc['baseline', 'different_accuracy']
    print(df[['same_accuracy', 'different_accuracy', 'same_impact', 'different_impact']].round(2))
    
    results_path = os.path.join(args.output_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {results_path}")
    
    create_visualizations(results, args.output_dir)
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main() 