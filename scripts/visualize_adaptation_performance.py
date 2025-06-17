import torch
import torch.nn as nn
import numpy as np
import h5py
import argparse
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import copy

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Model Imports ---
try:
    from baselines.models.conv6 import SameDifferentCNN as VanillaModel
    from scripts.temp_model import PB_Conv6 as MetaModel
    print("Successfully imported model architectures.")
except ImportError as e:
    print(f"Fatal Error importing models: {e}. A dummy class will be used.")
    sys.exit(1)

class NaturalisticDataset(Dataset):
    """
    A simple dataset for loading images and labels from the naturalistic HDF5 files.
    Reads top-level 'images' and 'labels' datasets.
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self._file = None
        
        with h5py.File(self.h5_path, 'r') as hf:
            self.dataset_len = len(hf['images'])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        
        image = self._file['images'][idx]
        label = self._file['labels'][idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.from_numpy(np.array(label)).long()

def evaluate_model(model, loader, device):
    """Evaluates the model's accuracy on the given data loader."""
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            predicted_labels = torch.argmax(predictions, dim=1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples if total_samples > 0 else 0

def run_adaptation_and_eval(model_class, model_path, loaders, args, device, model_type):
    """Loads a model, adapts it, and evaluates performance at each step."""
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 128) if model_type == 'vanilla' else (35, 35)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loaders['train'].dataset.transform = transform
    loaders['test'].dataset.transform = transform
    
    model = model_class()
    if model_path:
        print(f"  Loading weights from {model_path}...")
        if model_type == 'meta':
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))

    learner = copy.deepcopy(model).to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    accuracies = []
    
    # Evaluate before any adaptation
    initial_acc = evaluate_model(learner, loaders['test'], device)
    accuracies.append(initial_acc)
    print(f"  Initial accuracy: {initial_acc:.4f}")

    train_iterator = iter(loaders['train'])
    learner.train()
    for step in range(args.adaptation_steps):
        try:
            images, labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(loaders['train']) # Reset iterator
            images, labels = next(train_iterator)

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = learner(images)
        error = loss_fn(predictions, labels)
        error.backward()
        optimizer.step()
        
        step_acc = evaluate_model(learner, loaders['test'], device)
        accuracies.append(step_acc)

    return accuracies

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loaders ---
    train_h5 = Path(args.data_dir) / 'train.h5'
    test_h5 = Path(args.data_dir) / 'test.h5'
    
    train_loader = DataLoader(NaturalisticDataset(train_h5), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(NaturalisticDataset(test_h5), batch_size=args.batch_size)
    loaders = {'train': train_loader, 'test': test_loader}

    # --- Seeds and Data Collection ---
    vanilla_seeds = [0, 1, 2, 3, 4]
    meta_seeds = [3, 4, 5, 6, 7]
    all_vanilla_curves, all_meta_curves = [], []

    # --- Process Vanilla Models ---
    print("\n--- Adapting Vanilla Models ---")
    for seed in vanilla_seeds:
        path = Path(args.vanilla_models_dir) / f"regular/conv6/seed_{seed}/initial_model.pth"
        print(f"Processing seed {seed}...")
        try:
            curve = run_adaptation_and_eval(VanillaModel, path, loaders, args, device, 'vanilla')
            all_vanilla_curves.append(curve)
        except Exception as e:
            print(f"  ERROR processing vanilla seed {seed}: {e}")

    # --- Process Meta Models ---
    print("\n--- Adapting Meta-Learned Models ---")
    for seed in meta_seeds:
        path = Path(args.meta_models_dir) / f"model_seed_{seed}_pretesting.pt"
        print(f"Processing seed {seed}...")
        try:
            curve = run_adaptation_and_eval(MetaModel, path, loaders, args, device, 'meta')
            all_meta_curves.append(curve)
        except Exception as e:
            print(f"  ERROR processing meta seed {seed}: {e}")

    # --- Plotting ---
    print("\n--- Generating Plot ---")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if all_vanilla_curves:
        vanilla_curves = np.array(all_vanilla_curves)
        mean_vanilla = vanilla_curves.mean(axis=0)
        std_vanilla = vanilla_curves.std(axis=0)
        ax.plot(range(len(mean_vanilla)), mean_vanilla, label='Vanilla Model', color='royalblue')
        ax.fill_between(range(len(mean_vanilla)), mean_vanilla - std_vanilla, mean_vanilla + std_vanilla, color='royalblue', alpha=0.2)

    if all_meta_curves:
        meta_curves = np.array(all_meta_curves)
        mean_meta = meta_curves.mean(axis=0)
        std_meta = meta_curves.std(axis=0)
        ax.plot(range(len(mean_meta)), mean_meta, label='Meta-Learned Model (PB-trained)', color='darkorange')
        ax.fill_between(range(len(mean_meta)), mean_meta - std_meta, mean_meta + std_meta, color='darkorange', alpha=0.2)

    ax.set_xlabel('Adaptation Steps on Naturalistic Data')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Adaptation Performance: PB-Trained vs. Vanilla on Naturalistic Task')
    ax.legend()
    ax.grid(True)
    plt.ylim(0, 1.0)

    plot_path = output_dir / 'adaptation_performance.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize adaptation performance of models transferring from PB to Naturalistic data.")
    
    # --- Paths ---
    parser.add_argument('--vanilla_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/single_task/results/pb_single_task', help='Base directory for initial Vanilla-PB models.')
    parser.add_argument('--meta_models_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6', help='Directory for trained Meta-PB models.')
    parser.add_argument('--data_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/data/naturalistic', help='Directory containing naturalistic train.h5 and test.h5.')
    parser.add_argument('--output_dir', type=str, default='/scratch/gpfs/mg7411/samedifferent/visualizations/adaptation_performance', help='Directory to save the output plot.')
    
    # --- Hyperparameters ---
    parser.add_argument('--lr', type=float, default=0.001, help='Unified learning rate for adaptation.')
    parser.add_argument('--adaptation_steps', type=int, default=20, help='Number of adaptation steps to perform and evaluate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for adaptation.')

    args = parser.parse_args()
    main(args) 