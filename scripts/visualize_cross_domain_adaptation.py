import os
import torch
import numpy as np
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import learn2learn as l2l
import h5py
import sys
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import copy

# --- Path Setup ---
# Add project root to sys.path to allow for imports like `meta_baseline`
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from meta_baseline.models.conv6lr import SameDifferentCNN as Conv6LR
from meta_baseline.models.conv6lr_legacy import Conv6LR_Legacy

# --- Data Loading Classes (copied from other scripts) ---

class MetaNaturalisticDataset(Dataset):
    """
    Dataset for loading meta-learning episodes from the naturalistic HDF5 files.
    """
    def __init__(self, h5_path, transform=None):
        self.h5_path = Path(h5_path)
        self.user_transform = transform
        self.processing_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._file = None
        self.episode_keys = []
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        try:
            self._file = h5py.File(self.h5_path, 'r')
            self.episode_keys = sorted([k for k in self._file.keys() if k.startswith('episode_')])
        except Exception as e:
            if self._file: self._file.close()
            raise

    def __len__(self):
        return len(self.episode_keys)

    def __getitem__(self, idx):
        if not self._file:
            self._file = h5py.File(self.h5_path, 'r')
        episode_key = self.episode_keys[idx]
        ep_group = self._file[episode_key]
        support_images = ep_group['support_images'][()]
        support_labels = ep_group['support_labels'][()]
        query_images = ep_group['query_images'][()]
        query_labels = ep_group['query_labels'][()]

        transformed_support = torch.stack([self.processing_transform(img) for img in support_images])
        transformed_query = torch.stack([self.processing_transform(img) for img in query_images])
        return transformed_support, torch.from_numpy(support_labels).long(), transformed_query, torch.from_numpy(query_labels).long()

    def close(self):
        if self._file: self._file.close(); self._file = None
    def __del__(self): self.close()

class SameDifferentDataset(Dataset):
    """
    Dataset for loading PB same-different task data.
    """
    def __init__(self, data_dir, tasks, split, support_sizes=[10]):
        self.data_dir = data_dir
        self.tasks = tasks
        self.split = split
        self.support_sizes = support_sizes
        self.episode_files = []
        for task in tasks:
            for support_size in support_sizes:
                file_path = os.path.join(data_dir, f'{task}_support{support_size}_{split}.h5')
                if os.path.exists(file_path):
                    self.episode_files.append({'file_path': file_path, 'task': task, 'support_size': support_size})
        if not self.episode_files: raise ValueError(f"No valid files found for tasks {tasks} in {data_dir}")
        self.total_episodes, self.file_episode_counts = 0, []
        for file_info in self.episode_files:
            with h5py.File(file_info['file_path'], 'r') as f:
                num_episodes = f['support_images'].shape[0]
                self.file_episode_counts.append(num_episodes)
                self.total_episodes += num_episodes

    def __len__(self):
        return self.total_episodes

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.file_episode_counts[file_idx]:
            idx -= self.file_episode_counts[file_idx]
            file_idx += 1
        file_info = self.episode_files[file_idx]
        with h5py.File(file_info['file_path'], 'r') as f:
            support_images = torch.from_numpy(f['support_images'][idx]).float() / 255.0
            support_labels = torch.from_numpy(f['support_labels'][idx]).long()
        support_images = support_images.permute(0, 3, 1, 2)
        return {'support_images': support_images, 'support_labels': support_labels, 'task': file_info['task']}

def collate_episodes(batch):
    if not batch: return {}
    support_images = [item['support_images'] for item in batch]
    support_labels = [item['support_labels'] for item in batch]
    return {
        'support_images': torch.stack(support_images),
        'support_labels': torch.stack(support_labels),
        'task': batch[0]['task']
    }

def get_naturalistic_ds(data_dir, ways, shots, test_shots):
    test_path = os.path.join(data_dir, 'test.h5')
    dataset = MetaNaturalisticDataset(h5_path=test_path)
    # The 'ways' and 'shots' are baked into the h5 file for naturalistic data,
    # so we don't need l2l.data.TaskDataset here. We just use a standard DataLoader.
    return DataLoader(dataset, batch_size=1) # batch_size=1 to get one episode at a time

def get_pb_ds(data_dir, ways, shots, test_shots):
    # For PB data, we adapt to one specific task. Let's use 'regular'.
    dataset = SameDifferentDataset(data_dir, tasks=['regular'], split='test', support_sizes=[shots])
    return DataLoader(dataset, batch_size=1, collate_fn=collate_episodes)

def get_model_weights(model):
    """Extracts and flattens model weights."""
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()

def adapt_and_collect_weights(model, task_data, adaptation_steps, inner_lr, dataset_type):
    """Adapts the model to a task and collects weights at each step."""
    # Use copy.deepcopy for standard nn.Module, as .clone() is for l2l models
    learner = copy.deepcopy(model)
    optimizer = torch.optim.SGD(learner.parameters(), lr=inner_lr)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    weights_trajectory = [get_model_weights(learner)]

    # Unpack data based on dataset type
    if dataset_type == 'naturalistic':
        # Batch size is 1, so we unsqueeze
        support_data, support_labels, _, _ = task_data
        support_data = support_data.squeeze(0)
        support_labels = support_labels.squeeze(0)
    elif dataset_type == 'pb':
        support_data = task_data['support_images'].squeeze(0)
        support_labels = task_data['support_labels'].squeeze(0)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    for step in range(adaptation_steps):
        optimizer.zero_grad()
        predictions = learner(support_data)
        loss = loss_fn(predictions, support_labels)
        
        # Check for unstable loss before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: Unstable loss ({loss.item()}) at step {step+1}. Halting adaptation for this model.")
            break

        loss.backward()
        optimizer.step()
        
        current_weights = get_model_weights(learner)
        # Check for unstable weights after optimizer step
        if np.isnan(current_weights).any():
            print(f"  WARNING: NaN values detected in weights at step {step+1}. Halting adaptation for this model.")
            break # Stop adapting if weights become NaN

        weights_trajectory.append(current_weights)

    return weights_trajectory

def plot_single_pca(pca_trajectory, title, output_path):
    """Plots the PCA results for a single model's trajectory."""
    plt.figure(figsize=(10, 8))
    
    # Plot trajectory
    plt.plot(pca_trajectory[:, 0], pca_trajectory[:, 1], 'b-', alpha=0.6, label='Adaptation Trajectory')
    # Plot start point
    plt.scatter(pca_trajectory[0, 0], pca_trajectory[0, 1], c='g', marker='o', s=150, edgecolors='k', zorder=5, label='Start Weights')
    # Plot end point
    plt.scatter(pca_trajectory[-1, 0], pca_trajectory[-1, 1], c='r', marker='X', s=200, edgecolors='k', zorder=5, label='End Weights')

    plt.title(title, fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"PCA plot saved to {output_path}")

def run_single_model_pca(model_path, model_type, model_class, data_loader, dataset_type, adaptation_steps, inner_lr, output_dir, plot_title_prefix):
    """Runs a PCA analysis for a single model's adaptation."""
    os.makedirs(output_dir, exist_ok=True)

    model = model_class()
    state_dict = load_model_checkpoint(model_path, model_type)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    task_data = next(iter(data_loader))
    
    trajectory_weights = adapt_and_collect_weights(model, task_data, adaptation_steps, inner_lr, dataset_type)
    
    # Perform PCA on this model's trajectory only
    pca = PCA(n_components=2)
    pca_trajectory = pca.fit_transform(trajectory_weights)

    # Plotting
    plot_single_pca(
        pca_trajectory,
        f'{plot_title_prefix} Adaptation Weight Drift',
        os.path.join(output_dir, f'{plot_title_prefix.lower().replace(" ", "_").replace("->", "to")}_pca.png')
    )

def load_model_checkpoint(path, model_type):
    """Loads a model checkpoint, handling different formats."""
    # Reverting weights_only=True as some checkpoints contain argparse.Namespace,
    # which is safe to load in this trusted context.
    checkpoint = torch.load(path, map_location='cpu')
    
    # All provided checkpoint paths seem to store the actual model weights 
    # inside the 'model_state_dict' key. The other model types were just 
    # special cases of this. We can simplify this function.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    
    # Fallback for checkpoints that are just the state_dict
    return checkpoint

def main():
    parser = argparse.ArgumentParser(description='Run cross-domain adaptation PCA analysis.')
    # --- Paths ---
    parser.add_argument('--pb_data_dir', type=str, required=True, help='Path to PB dataset directory.')
    parser.add_argument('--naturalistic_data_dir', type=str, required=True, help='Path to naturalistic dataset directory.')
    parser.add_argument('--output_dir', type=str, default='./visualizations/cross_domain_pca', help='Directory to save plots.')
    # Model Weights
    parser.add_argument('--meta_pb_model', type=str, required=True, help='Path to meta-trained PB model.')
    parser.add_argument('--vanilla_pb_model', type=str, required=True, help='Path to vanilla-trained PB model.')
    parser.add_argument('--meta_nat_model', type=str, required=True, help='Path to meta-trained naturalistic model.')
    parser.add_argument('--vanilla_nat_model', type=str, required=True, help='Path to vanilla-trained naturalistic model.')
    # --- Hyperparameters ---
    parser.add_argument('--shots', type=int, default=10, help='Support set size for adaptation.')
    parser.add_argument('--adaptation_steps', type=int, default=15, help='Number of adaptation steps.')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='Learning rate for adaptation.')

    args = parser.parse_args()
    
    # --- Setup DataLoaders ---
    pb_loader = get_pb_ds(args.pb_data_dir, ways=2, shots=args.shots, test_shots=args.shots)
    naturalistic_loader = get_naturalistic_ds(args.naturalistic_data_dir, ways=2, shots=args.shots, test_shots=args.shots)

    # --- Run 4 Separate Analyses ---
    
    # 1. Meta PB -> Naturalistic
    print("Running Analysis 1: Meta PB model -> Naturalistic data")
    run_single_model_pca(
        model_path=args.meta_pb_model,
        model_type='meta_pb',
        model_class=Conv6LR,
        data_loader=naturalistic_loader,
        dataset_type='naturalistic',
        adaptation_steps=args.adaptation_steps,
        inner_lr=args.inner_lr,
        output_dir=args.output_dir,
        plot_title_prefix='Meta PB -> Naturalistic'
    )

    # 2. Vanilla PB -> Naturalistic
    print("\nRunning Analysis 2: Vanilla PB model -> Naturalistic data")
    run_single_model_pca(
        model_path=args.vanilla_pb_model,
        model_type='vanilla_pb',
        model_class=Conv6LR_Legacy,
        data_loader=naturalistic_loader,
        dataset_type='naturalistic',
        adaptation_steps=args.adaptation_steps,
        inner_lr=args.inner_lr,
        output_dir=args.output_dir,
        plot_title_prefix='Vanilla PB -> Naturalistic'
    )

    # 3. Meta Naturalistic -> PB
    print("\nRunning Analysis 3: Meta Naturalistic model -> PB data")
    run_single_model_pca(
        model_path=args.meta_nat_model,
        model_type='meta_naturalistic',
        model_class=Conv6LR,
        data_loader=pb_loader,
        dataset_type='pb',
        adaptation_steps=args.adaptation_steps,
        inner_lr=args.inner_lr,
        output_dir=args.output_dir,
        plot_title_prefix='Meta Naturalistic -> PB'
    )

    # 4. Vanilla Naturalistic -> PB
    print("\nRunning Analysis 4: Vanilla Naturalistic model -> PB data")
    run_single_model_pca(
        model_path=args.vanilla_nat_model,
        model_type='vanilla_naturalistic',
        model_class=Conv6LR,
        data_loader=pb_loader,
        dataset_type='pb',
        adaptation_steps=args.adaptation_steps,
        inner_lr=args.inner_lr,
        output_dir=args.output_dir,
        plot_title_prefix='Vanilla Naturalistic -> PB'
    )

if __name__ == '__main__':
    main() 