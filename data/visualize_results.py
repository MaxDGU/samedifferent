import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize model accuracy results")
    parser.add_argument("--meta-results-dir", type=str, default="naturalistic",
                        help="Directory containing meta model results JSON files")
    parser.add_argument("--output-file", type=str, default="model_accuracy_comparison.png",
                        help="Output file path for the chart")
    args = parser.parse_args()

    # Set the style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))

    # Load results
    with open('naturalistic/test_results_vanilla.json', 'r') as f:
        vanilla_results = json.load(f)

    with open('naturalistic/test_results_meta6.json', 'r') as f:
        meta6_results = json.load(f)

    # Load meta results with the exact filenames
    conv2_meta_path = os.path.join(args.meta_results_dir, 'conv2_meta_results.json')
    conv4_meta_path = os.path.join(args.meta_results_dir, 'conv4_meta_results.json')
    
    have_conv2_meta = os.path.exists(conv2_meta_path)
    have_conv4_meta = os.path.exists(conv4_meta_path)
    
    if have_conv2_meta:
        with open(conv2_meta_path, 'r') as f:
            conv2_meta_results = json.load(f)
            print(f"Loaded Conv2-Meta results: {conv2_meta_results['mean_accuracy']:.4f} ± {conv2_meta_results['stderr_accuracy']:.4f}")
    else:
        print(f"Warning: Could not find Conv2-Meta results at {conv2_meta_path}")
    
    if have_conv4_meta:
        with open(conv4_meta_path, 'r') as f:
            conv4_meta_results = json.load(f)
            print(f"Loaded Conv4-Meta results: {conv4_meta_results['mean_accuracy']:.4f} ± {conv4_meta_results['stderr_accuracy']:.4f}")
    else:
        print(f"Warning: Could not find Conv4-Meta results at {conv4_meta_path}")

    # Process vanilla results
    model_names = []
    accuracies = []
    std_errors = []

    # Process conv2 results
    conv2_accs = [vanilla_results['conv2'][seed]['test_accuracy'] for seed in vanilla_results['conv2']]
    model_names.append('Conv2')
    accuracies.append(np.mean(conv2_accs))
    std_errors.append(np.std(conv2_accs) / np.sqrt(len(conv2_accs)))

    # Process conv2-meta results if available
    if have_conv2_meta:
        model_names.append('Conv2-Meta')
        accuracies.append(conv2_meta_results['mean_accuracy'])
        std_errors.append(conv2_meta_results['stderr_accuracy'])

    # Process conv4 results
    conv4_accs = [vanilla_results['conv4'][seed]['test_accuracy'] for seed in vanilla_results['conv4']]
    model_names.append('Conv4')
    accuracies.append(np.mean(conv4_accs))
    std_errors.append(np.std(conv4_accs) / np.sqrt(len(conv4_accs)))

    # Process conv4-meta results if available
    if have_conv4_meta:
        model_names.append('Conv4-Meta')
        accuracies.append(conv4_meta_results['mean_accuracy'])
        std_errors.append(conv4_meta_results['stderr_accuracy'])

    # Process conv6 results
    conv6_accs = [vanilla_results['conv6'][seed]['test_accuracy'] for seed in vanilla_results['conv6']]
    model_names.append('Conv6')
    accuracies.append(np.mean(conv6_accs))
    std_errors.append(np.std(conv6_accs) / np.sqrt(len(conv6_accs)))

    # Process meta6 results
    meta6_accs = [meta6_results[seed]['test_accuracy'] for seed in meta6_results]
    model_names.append('Conv6-Meta')
    accuracies.append(np.mean(meta6_accs))
    std_errors.append(np.std(meta6_accs) / np.sqrt(len(meta6_accs)))

    # Create bar chart with grouped bars for vanilla and meta versions
    x_pos = np.arange(len(model_names))
    
    # Color vanilla and meta models differently
    colors = []
    for name in model_names:
        if 'Meta' in name:
            colors.append('orange')
        else:
            colors.append('steelblue')
    
    bars = plt.bar(x_pos, accuracies, yerr=std_errors, align='center', alpha=0.8, 
            ecolor='black', capsize=10, color=colors)

    # Add labels and title
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Naturalistic Test Accuracy by Model Architecture', fontsize=14)
    plt.xticks(x_pos, model_names, rotation=45, ha='right')

    # Add a reference line at chance level (0.5)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance level')

    # Add value labels on top of each bar
    for i, v in enumerate(accuracies):
        plt.text(i, v + std_errors[i] + 0.01, f'{v:.3f}', ha='center')

    # Add legend for model types
    vanilla_patch = plt.Rectangle((0,0),1,1, color='steelblue', alpha=0.8)
    meta_patch = plt.Rectangle((0,0),1,1, color='orange', alpha=0.8)
    plt.legend([vanilla_patch, meta_patch, plt.Line2D([0], [0], color='r', linestyle='--')], 
               ['Vanilla Models', 'Meta-Learning Models', 'Chance Level'])

    # Limit y-axis slightly above the highest bar for clarity
    plt.ylim(0.45, max(np.array(accuracies) + np.array(std_errors)) + 0.05)

    # Print summary statistics
    print("\nModel Performance Summary:")
    for i, model in enumerate(model_names):
        print(f"{model}: {accuracies[i]:.4f} ± {std_errors[i]:.4f}")

    # Save the figure
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"\nChart saved to {args.output_file}")
    plt.show()

if __name__ == "__main__":
    main() 