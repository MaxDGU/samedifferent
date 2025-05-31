#!/usr/bin/env python3
import os
import glob
import pickle
import json
import numpy as np
import argparse

def extract_model_results(results_dir, model_type, output_file):
    """
    Extract accuracy statistics from all seed results for a given model type
    and save them to a JSON file.
    
    Args:
        results_dir: Directory containing the result pkl files
        model_type: Model type (e.g., 'conv2-meta', 'conv4-meta')
        output_file: Path to save the JSON output
    """
    # Find all result files for this model type
    pattern = os.path.join(results_dir, f"{model_type}_seed_*_results.pkl")
    result_files = glob.glob(pattern)
    
    if not result_files:
        print(f"No result files found for {model_type} with pattern: {pattern}")
        return
    
    print(f"Found {len(result_files)} result files for {model_type}")
    
    # Load and process each result file
    accuracies = {}
    
    for file_path in result_files:
        # Extract seed from filename
        filename = os.path.basename(file_path)
        seed = filename.split("_seed_")[1].split("_")[0]
        
        # Load results
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
                
            # Extract accuracy
            accuracy = results['accuracy']
            accuracies[seed] = {
                "test_accuracy": accuracy
            }
            print(f"Seed {seed}: Accuracy = {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not accuracies:
        print(f"No valid results found for {model_type}")
        return
    
    # Calculate statistics
    acc_values = [accuracies[seed]["test_accuracy"] for seed in accuracies]
    mean_acc = np.mean(acc_values)
    std_acc = np.std(acc_values)
    stderr_acc = std_acc / np.sqrt(len(acc_values))
    
    # Print summary statistics
    print(f"\n{model_type} Summary:")
    print(f"Average Accuracy: {mean_acc:.4f}")
    print(f"Standard Deviation: {std_acc:.4f}")
    print(f"Standard Error: {stderr_acc:.4f}")
    print(f"Number of Seeds: {len(accuracies)}")
    
    # Save to JSON file
    output = {
        "model_type": model_type,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "stderr_accuracy": stderr_acc,
        "num_seeds": len(accuracies),
        "seed_results": accuracies
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract model results from pickle files")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing result pkl files")
    parser.add_argument("--model-types", type=str, nargs="+", required=True, 
                        help="Model types to process (e.g., conv2-meta conv4-meta)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save JSON output files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each model type
    for model_type in args.model_types:
        output_file = os.path.join(args.output_dir, f"{model_type}_results.json")
        extract_model_results(args.results_dir, model_type, output_file)

if __name__ == "__main__":
    main() 