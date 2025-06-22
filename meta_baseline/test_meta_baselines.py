import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
import argparse
import learn2learn as l2l
import sys

from .models.conv2lr import SameDifferentCNN as Conv2CNN
from .models.conv4lr import SameDifferentCNN as Conv4CNN
from .models.conv6lr import SameDifferentCNN as Conv6CNN
from .models.utils_meta import SameDifferentDataset, collate_episodes, validate

PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]
ARCHITECTURES = {
    'conv2': Conv2CNN,
    'conv4': Conv4CNN,
    'conv6': Conv6CNN
}

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description="Run testing on pre-trained meta-learning models.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the PB dataset')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing the trained models and to save results')
    parser.add_argument('--architecture', type=str, required=True, choices=['conv2', 'conv4', 'conv6'], help='Model architecture to use')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for the model to test')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--test_support_size', type=int, nargs='+', default=[10], help='A list of support sizes for testing')
    parser.add_argument('--test_adaptation_steps', type=int, default=15, help='Number of adaptation steps during testing')
    parser.add_argument('--inner_lr', type=float, default=0.001, help='Inner loop learning rate (must match training)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    set_seed(args.seed)

    arch_dir = os.path.join(args.results_dir, args.architecture, f'seed_{args.seed}')
    model_path = os.path.join(arch_dir, 'best_model.pt')

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nCreating {args.architecture} model")
    model = ARCHITECTURES[args.architecture]().to(device)

    maml = l2l.algorithms.MAML(
        model,
        lr=args.inner_lr,
        first_order=False,
        allow_unused=True,
        allow_nograd=True
    )

    print(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"ERROR: Failed to load model state: {e}", file=sys.stderr)
        sys.exit(1)


    print("\nTesting on individual tasks...")
    test_results = {}
    for task in PB_TASKS:
        print(f"\nTesting on task: {task}")
        try:
            test_dataset = SameDifferentDataset(args.data_dir, [task], 'test', support_sizes=args.test_support_size)
            if len(test_dataset) == 0:
                print(f"  WARNING: No data found for task {task}. Skipping.")
                continue

            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                   num_workers=1, pin_memory=True,
                                   collate_fn=collate_episodes)

            torch.cuda.empty_cache()

            test_loss, test_acc = validate(
                maml, test_loader, device,
                args.test_adaptation_steps
            )

            test_results[task] = {
                'loss': test_loss,
                'accuracy': test_acc
            }
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        except Exception as e:
            print(f"  ERROR testing task {task}: {e}")
            test_results[task] = {'error': str(e)}

    results = {
        'test_results': test_results,
        'model_file': model_path,
        'args': vars(args)
    }

    output_filename = os.path.join(arch_dir, 'results.json')
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nTesting results saved to: {output_filename}")


if __name__ == '__main__':
    main() 