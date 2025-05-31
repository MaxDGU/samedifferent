import os
import json
import numpy as np
from pathlib import Path

# Define tasks
PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
            'random_color', 'arrows', 'irregular', 'filled', 'original']
SEEDS = list(range(42, 47))  # 6 seeds (0-5)

def main():
    # Update base path to match the full path structure
    results_dir = '/gpfs/mg7411/results/leave_one_out_seeds'
    output_path = os.path.join(results_dir, 'compiled_results.json')
    
    # Verify results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found at {results_dir}")
        return
    
    print(f"Looking for results in: {results_dir}")
    
    # Store all results
    all_results = {}
    missing_files = []
    
    # For each PB task
    for task in PB_TASKS:
        print(f"\nProcessing task: {task}")
        task_accuracies = []
        task_results = {'individual_seeds': {}}
        
        # For each seed
        for seed in SEEDS:
            metrics_path = os.path.join(results_dir, task, f'seed_{seed}', 'results.json')
            
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        # Extract accuracy from test_metrics
                        if 'test_metrics' in metrics and 'accuracy' in metrics['test_metrics']:
                            accuracy = metrics['test_metrics']['accuracy']
                            task_accuracies.append(accuracy)
                            task_results['individual_seeds'][f'seed_{seed}'] = {
                                'accuracy': accuracy,
                                'loss': metrics['test_metrics'].get('loss'),
                                'adaptation_steps': metrics['test_metrics'].get('adaptation_steps')
                            }
                            print(f"  Seed {seed}: Found accuracy = {accuracy*100:.2f}%")
                        else:
                            print(f"  Seed {seed}: No test_metrics/accuracy found in metrics file")
                except json.JSONDecodeError:
                    print(f"  Error: Could not parse metrics file for {task} seed {seed}")
                except Exception as e:
                    print(f"  Error processing {task} seed {seed}: {str(e)}")
            else:
                missing_files.append(metrics_path)
                print(f"  Seed {seed}: No metrics file found (this is expected for incomplete runs)")
        
        if task_accuracies:
            task_results.update({
                'mean_accuracy': float(np.mean(task_accuracies)),
                'std_accuracy': float(np.std(task_accuracies)),
                'num_seeds': len(task_accuracies)
            })
            all_results[task] = task_results
            print(f"  Task summary: {len(task_accuracies)} seeds found, "
                  f"mean = {task_results['mean_accuracy']*100:.2f}% ± "
                  f"{task_results['std_accuracy']*100:.2f}%")
        else:
            print(f"  Warning: No results found for task {task}")
    
    if not all_results:
        print("\nError: No results found for any task!")
        if missing_files:
            print("\nMissing files (expected for incomplete runs):")
            for f in missing_files:
                print(f"  {f}")
        return
    
    # Calculate overall statistics
    all_means = [results['mean_accuracy'] for results in all_results.values() if 'mean_accuracy' in results]
    if all_means:
        overall_stats = {
            'overall_mean': float(np.mean(all_means)),
            'overall_std': float(np.std(all_means)),
            'num_tasks': len(all_means),
            'individual_task_results': all_results
        }
        
        # Save compiled results
        with open(output_path, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"\nSaved compiled results to: {output_path}")
        
        # Print summary
        print('\nResults Summary:')
        print(f'Overall Mean Accuracy: {overall_stats["overall_mean"]*100:.2f}% ± {overall_stats["overall_std"]*100:.2f}%')
        print(f'Number of tasks with results: {overall_stats["num_tasks"]}/{len(PB_TASKS)}')
        print('\nPer-Task Results:')
        for task, results in all_results.items():
            if 'mean_accuracy' in results:
                print(f'{task}: {results["mean_accuracy"]*100:.2f}% ± {results["std_accuracy"]*100:.2f}% (n={results["num_seeds"]})')
    else:
        print("\nError: Could not calculate overall statistics - no valid results found")

if __name__ == '__main__':
    main() 