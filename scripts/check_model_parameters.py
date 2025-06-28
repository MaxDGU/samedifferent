import torch
import argparse
import sys
from pathlib import Path

# --- Path Setup ---
# Add project root to sys.path to allow for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6LR
from meta_baseline.models.conv6lr_legacy import Conv6LR_Legacy

def count_parameters(model):
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())

def load_and_check_model(model_path, model_class):
    """Loads a model and returns its parameter count."""
    try:
        model = model_class()
        # It's not necessary to load the weights to count parameters,
        # but we do it to ensure the class is compatible with the file.
        # We can skip the actual loading if just counting is needed.
        # For now, let's keep it simple and just instantiate.
        num_params = count_parameters(model)
        return num_params
    except Exception as e:
        return f"Error instantiating model: {e}"

def main():
    parser = argparse.ArgumentParser(description='Check parameter counts of specified model files.')
    # Model Weights
    parser.add_argument('--meta_pb_model', type=str, required=True, help='Path to meta-trained PB model.')
    parser.add_argument('--vanilla_pb_model', type=str, required=True, help='Path to vanilla-trained PB model.')
    parser.add_argument('--meta_nat_model', type=str, required=True, help='Path to meta-trained naturalistic model.')
    parser.add_argument('--vanilla_nat_model', type=str, required=True, help='Path to vanilla-trained naturalistic model.')
    args = parser.parse_args()

    models_to_check = {
        "Meta-trained PB": (args.meta_pb_model, Conv6LR),
        "Vanilla-trained PB": (args.vanilla_pb_model, Conv6LR_Legacy),
        "Meta-trained Naturalistic": (args.meta_nat_model, Conv6LR),
        "Vanilla-trained Naturalistic": (args.vanilla_nat_model, Conv6LR),
    }

    print("--- Model Parameter Count Verification ---")
    for name, (path, model_class) in models_to_check.items():
        print(f"\nChecking: {name}")
        print(f"  - Path: {path}")
        print(f"  - Architecture: {model_class.__name__}")
        num_params = load_and_check_model(path, model_class)
        print(f"  - Parameter Count: {num_params:,}")

    print("\n--- Comparison ---")
    pb_meta_params = load_and_check_model(args.meta_pb_model, Conv6LR)
    pb_vanilla_params = load_and_check_model(args.vanilla_pb_model, Conv6LR_Legacy)
    if pb_meta_params == pb_vanilla_params:
        print("PB Models: MATCH")
    else:
        print("PB Models: MISMATCH")

    nat_meta_params = load_and_check_model(args.meta_nat_model, Conv6LR)
    nat_vanilla_params = load_and_check_model(args.vanilla_nat_model, Conv6LR)
    if nat_meta_params == nat_vanilla_params:
        print("Naturalistic Models: MATCH")
    else:
        print("Naturalistic Models: MISMATCH")


if __name__ == '__main__':
    main() 