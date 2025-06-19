import torch
import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import copy
import itertools
from torch.utils.data import Dataset, DataLoader
import h5py

# --- Setup Project Path ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Model Imports ---
# Updated to handle potential import errors more gracefully.
try:
    from baselines.models.conv2 import SameDifferentCNN as Conv2CNN
    from baselines.models.conv4 import SameDifferentCNN as Conv4CNN
    from baselines.models.conv6 import SameDifferentCNN as Conv6CNN
    MODEL_MAP = {'conv2lr': Conv2CNN, 'conv4lr': Conv4CNN, 'conv6lr': Conv6CNN}
    print("Successfully imported baseline model architectures.")
except ImportError as e:
    print(f"Warning: Could not import all baseline models: {e}. The script may fail if these models are needed.")
    MODEL_MAP = {}

# --- Constants & Paths ---
# Base directories for models on the cluster
BASE_DIR = Path("/scratch/gpfs/mg7411/samedifferent")
PB_MODELS_BASE = Path("/scratch/gpfs/mg7411")
NATURALISTIC_MODELS_BASE = BASE_DIR / "logs_naturalistic_vanilla"
NATURALISTIC_DATA_PATH = BASE_DIR / "data/naturalistic_new/meta/naturalistic_metaset.h5"

# Define paths for each architecture
ARCH_PATHS = {
    'conv2lr': {
        'pb': PB_MODELS_BASE / "conv2lr_runs_20250127_131933/seed_0/model_seed_0_pretesting.pt",
        'naturalistic': NATURALISTIC_MODELS_BASE / "conv2lr/seed_42/best_model.pt"
    },
    'conv4lr': {
        'pb': PB_MODELS_BASE / "exp1_(finished)conv4lr_runs_20250126_201548/seed_0/model_seed_0.pt",
        'naturalistic': NATURALISTIC_MODELS_BASE / "conv4lr/seed_42/best_model.pt"
    },
    'conv6lr': {
        'pb': PB_MODELS_BASE / "exp1_(untested)conv6lr_runs_20250127_110352/seed_0/model_seed_0.pt",
        'naturalistic': NATURALISTIC_MODELS_BASE / "conv6lr/seed_42/best_model.pt"
    }
}

# --- Dataset ---
class NaturalisticDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.images = f['train_images'][:]
            self.labels = f['train_labels'][:]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Add a channel dimension, convert to float, and normalize
        image = torch.from_numpy(self.images[idx]).unsqueeze(0).float() / 255.0
        label = torch.from_numpy(np.array(self.labels[idx])).long()
        return image, label

# --- Helper Functions ---
def flatten_state_dict(state_dict):
    """Flattens weights from a state_dict into a single numpy vector."""
    return np.concatenate([p.cpu().detach().numpy().flatten() for p in state_dict.values()])

def adapt_model(model, adaptation_data_loader, device, lr, steps):
    """Performs adaptation on a model and returns the adapted model's state_dict."""
    if model is None:
        return None
    learner = copy.deepcopy(model)
    learner.to(device)
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    learner.train()
    for _ in range(steps):
        try:
            batch = next(iter(adaptation_data_loader))
        except StopIteration:
            # Handle case where loader is exhausted
            return learner.state_dict()
            
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = learner(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
    
    learner.eval()
    return learner.state_dict()

def load_model_and_state(model_path, arch):
    """Loads a state_dict and optionally the model instance if the arch is known."""
    try:
        if not model_path.exists():
            print(f"Warning: Checkpoint not found at {model_path}")
            return None, None
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        model_instance = None
        if arch in MODEL_MAP:
            try:
                model_instance = MODEL_MAP[arch]()
                # Use strict=False to be lenient
                model_instance.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Loaded state_dict for {arch} but couldn't instantiate/load into model. Adaptation might fail. Error: {e}")
                model_instance = None # Can't adapt if this fails
        
        return state_dict, model_instance

    except Exception as e:
        print(f"ERROR: Failed to load model from {model_path}. Error: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Run multi-task adaptation and visualize PCA.")
    parser.add_argument('--output_dir', type=str, default='visualizations/adaptation_pca', help='Directory to save the PCA plot.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for adaptation.')
    parser.add_argument('--steps', type=int, default=1, help='Number of adaptation steps.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load adaptation data
    naturalistic_dataset = NaturalisticDataset(NATURALISTIC_DATA_PATH)
    naturalistic_loader = DataLoader(naturalistic_dataset, batch_size=32, shuffle=True)
    print("Naturalistic adaptation data loaded.")

    print("\nLoading weights and calculating adaptation vectors...")

    all_vectors = {}
    all_labels = {}
    all_styles = {}

    for arch, paths in ARCH_PATHS.items():
        print(f"\n--- Processing architecture: {arch} ---")
        
        # 1. Load Initial PB Weights
        pb_initial_state, pb_model_for_adaptation = load_model_and_state(paths['pb'], arch)
        if pb_initial_state is None:
            print(f"Skipping arch {arch} due to missing PB weights.")
            continue
        pb_initial_vec = flatten_state_dict(pb_initial_state)

        # 2. Load Initial Naturalistic Weights
        naturalistic_initial_state, naturalistic_model_for_adaptation = load_model_and_state(paths['naturalistic'], arch)
        if naturalistic_initial_state is None:
            print(f"Skipping arch {arch} due to missing Naturalistic weights.")
            continue
        naturalistic_initial_vec = flatten_state_dict(naturalistic_initial_state)

        # 3. Adapt PB model on Naturalistic data
        print(f"Adapting PB model for {arch} on naturalistic data...")
        adapted_pb_state = adapt_model(pb_model_for_adaptation, naturalistic_loader, device, args.lr, args.steps)
        if adapted_pb_state:
            adapted_pb_vec = flatten_state_dict(adapted_pb_state)
        else:
            print(f"Cannot adapt PB model for {arch}, using initial vector as placeholder.")
            adapted_pb_vec = pb_initial_vec

        # 4. Adapt Naturalistic model on Naturalistic data (control)
        print(f"Adapting Naturalistic model for {arch} on naturalistic data...")
        adapted_naturalistic_state = adapt_model(naturalistic_model_for_adaptation, naturalistic_loader, device, args.lr, args.steps)
        if adapted_naturalistic_state:
            adapted_naturalistic_vec = flatten_state_dict(adapted_naturalistic_state)
        else:
            print(f"Cannot adapt Naturalistic model for {arch}, using initial vector as placeholder.")
            adapted_naturalistic_vec = naturalistic_initial_vec
        
        all_vectors[arch] = [pb_initial_vec, adapted_pb_vec, naturalistic_initial_vec, adapted_naturalistic_vec]
        all_labels[arch] = [f'{arch}-PB Init', f'{arch}-PB Adapt', f'{arch}-Nat Init', f'{arch}-Nat Adapt']
        all_styles[arch] = [
            {'color': 'red', 'marker': 'o'},
            {'color': 'red', 'marker': 'x'},
            {'color': 'blue', 'marker': 'o'},
            {'color': 'blue', 'marker': 'x'}
        ]

    if not all_vectors:
        print("No vectors could be calculated. Exiting.")
        return

    # --- Pad vectors to the same length for PCA ---
    print("\nPadding weight vectors for combined PCA...")
    all_flattened_vectors = list(itertools.chain.from_iterable(all_vectors.values()))
    
    if not all_flattened_vectors:
        print("No vectors available for PCA. Exiting.")
        return

    max_len = max(len(v) for v in all_flattened_vectors)
    print(f"All vectors will be padded to length: {max_len}")

    padded_vectors = [np.pad(v, (0, max_len - len(v)), 'constant') for v in all_flattened_vectors]
    
    # --- Perform PCA ---
    print("Performing PCA on all processed weights...")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(np.array(padded_vectors))
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA Explained Variance: {explained_variance[0]:.2%} (PC1), {explained_variance[1]:.2%} (PC2)")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(16, 12))
    
    all_labels_flat = list(itertools.chain.from_iterable(all_labels.values()))
    all_styles_flat = list(itertools.chain.from_iterable(all_styles.values()))

    # Plot points
    for i in range(len(padded_vectors)):
        style = all_styles_flat[i]
        label = all_labels_flat[i]
        ax.scatter(principal_components[i, 0], principal_components[i, 1], 
                   c=style['color'], marker=style['marker'], s=150, alpha=0.8, label=label)

    # Plot arrows
    for arch in all_vectors.keys():
        # Get indices for this arch's vectors
        pb_init_idx = all_labels_flat.index(f'{arch}-PB Init')
        pb_adapt_idx = all_labels_flat.index(f'{arch}-PB Adapt')
        nat_init_idx = all_labels_flat.index(f'{arch}-Nat Init')
        nat_adapt_idx = all_labels_flat.index(f'{arch}-Nat Adapt')
        
        # Draw PB arrow
        p_start, p_end = principal_components[pb_init_idx], principal_components[pb_adapt_idx]
        ax.arrow(p_start[0], p_start[1], p_end[0] - p_start[0], p_end[1] - p_start[1],
                 color='red', linestyle='solid', head_width=0.05, lw=1.5, length_includes_head=True)

        # Draw Naturalistic arrow
        n_start, n_end = principal_components[nat_init_idx], principal_components[nat_adapt_idx]
        ax.arrow(n_start[0], n_start[1], n_end[0] - n_start[0], n_end[1] - n_start[1],
                 color='blue', linestyle='solid', head_width=0.05, lw=1.5, length_includes_head=True)

    ax.set_title('PCA of Multi-Task Adaptation Trajectories', fontsize=20)
    ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.2%})', fontsize=16)
    ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.2%})', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='PB-Trained'),
        Line2D([0], [0], color='blue', lw=2, label='Naturalistic-Trained'),
        Line2D([0], [0], marker='o', color='grey', label='Initial Weights', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='x', color='grey', label='Adapted Weights', markersize=10, linestyle='None')
    ]
    ax.legend(handles=legend_elements, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save the plot
    output_path = Path(args.output_dir) / 'multi_task_adaptation_pca.png'
    output_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_path)
    
    print(f"\nPCA plot saved to {output_path}")

if __name__ == '__main__':
    main() 