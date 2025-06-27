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
    learner = model.clone()
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
        loss.backward()
        optimizer.step()
        weights_trajectory.append(get_model_weights(learner))

    return weights_trajectory

def plot_pca(pca_results, title, output_path):
    """Plots the PCA results."""
    plt.figure(figsize=(12, 10))
    colors = {'meta': 'r', 'vanilla': 'b'}
    labels = {'meta': 'Meta-trained', 'vanilla': 'Vanilla-trained'}

    for model_type, trajectory in pca_results.items():
        plt.plot(trajectory[:, 0], trajectory[:, 1], f'{colors[model_type]}-', alpha=0.5, label=f'{labels[model_type]} Trajectory')
        plt.scatter(trajectory[0, 0], trajectory[0, 1], c=colors[model_type], marker='o', s=150, edgecolors='k', zorder=5, label=f'{labels[model_type]} Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c=colors[model_type], marker='X', s=200, edgecolors='k', zorder=5, label=f'{labels[model_type]} End')

    plt.title(title, fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"PCA plot saved to {output_path}")

def load_model_checkpoint(path, model_type):
    """Loads a model checkpoint, handling different formats."""
    # Add weights_only=True for security and to suppress the warning.
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    
    # All provided checkpoint paths seem to store the actual model weights 
    # inside the 'model_state_dict' key. The other model types were just 
    # special cases of this. We can simplify this function.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    
    # Fallback for checkpoints that are just the state_dict
    return checkpoint

def run_pca_analysis(meta_model_path, vanilla_model_path, meta_model_type, vanilla_model_type, meta_model_class, vanilla_model_class, data_loader, dataset_type, adaptation_steps, inner_lr, output_dir, plot_title_prefix):
    """Runs a full PCA analysis for one experiment."""
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate models using their specific classes
    meta_model = meta_model_class()
    vanilla_model = vanilla_model_class()

    # Load state dicts using the helper function
    meta_state_dict = load_model_checkpoint(meta_model_path, meta_model_type)
    vanilla_state_dict = load_model_checkpoint(vanilla_model_path, vanilla_model_type)
    
    # Set strict=False to ignore missing batch norm running stats, which can happen
    # if models were saved in eval() mode.
    meta_model.load_state_dict(meta_state_dict, strict=False)
    vanilla_model.load_state_dict(vanilla_state_dict, strict=False)
    
    meta_model.eval()
    vanilla_model.eval()

    task_data = next(iter(data_loader))

    meta_trajectory_weights = adapt_and_collect_weights(meta_model, task_data, adaptation_steps, inner_lr, dataset_type)
    vanilla_trajectory_weights = adapt_and_collect_weights(vanilla_model, task_data, adaptation_steps, inner_lr, dataset_type)

    all_weights = np.array(meta_trajectory_weights + vanilla_trajectory_weights)
    
    pca = PCA(n_components=2)
    pca.fit(all_weights)

    pca_meta_traj = pca.transform(meta_trajectory_weights)
    pca_vanilla_traj = pca.transform(vanilla_trajectory_weights)

    plot_pca(
        {'meta': pca_meta_traj, 'vanilla': pca_vanilla_traj},
        f'{plot_title_prefix} Adaptation Weight Drift',
        os.path.join(output_dir, f'{plot_title_prefix.lower().replace(" ", "_")}_pca.png')
    )

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

    # --- Experiment 1: PB -> Naturalistic ---
    print("Running Experiment 1: PB-trained models adapting to Naturalistic data")
    run_pca_analysis(
        meta_model_path=args.meta_pb_model,
        vanilla_model_path=args.vanilla_pb_model,
        meta_model_type='meta_pb',
        vanilla_model_type='vanilla_pb',
        meta_model_class=Conv6LR, # Meta PB model uses the NEW architecture
        vanilla_model_class=Conv6LR_Legacy, # Vanilla PB model uses the LEGACY architecture
        data_loader=naturalistic_loader,
        dataset_type='naturalistic',
        adaptation_steps=args.adaptation_steps,
        inner_lr=args.inner_lr,
        output_dir=args.output_dir,
        plot_title_prefix='PB to Naturalistic'
    )

    # --- Experiment 2: Naturalistic -> PB ---
    print("\nRunning Experiment 2: Naturalistic-trained models adapting to PB data")
    run_pca_analysis(
        meta_model_path=args.meta_nat_model,
        vanilla_model_path=args.vanilla_nat_model,
        meta_model_type='meta_naturalistic',
        vanilla_model_type='vanilla_naturalistic',
        meta_model_class=Conv6LR, # Both Naturalistic models use the NEW architecture
        vanilla_model_class=Conv6LR,
        data_loader=pb_loader,
        dataset_type='pb',
        adaptation_steps=args.adaptation_steps,
        inner_lr=args.inner_lr,
        output_dir=args.output_dir,
        plot_title_prefix='Naturalistic to PB'
    )
    
if __name__ == '__main__':
    main() 