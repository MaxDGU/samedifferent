import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import glob
from PIL import Image
from torchvision import transforms
import json
import argparse
import sys
import pickle

# Add parent directory to path to import models
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Commenting out, prefer PYTHONPATH setup
from meta_baseline.models.conv2lr import SameDifferentCNN as Conv2CNN
from meta_baseline.models.conv4lr import SameDifferentCNN as Conv4CNN
from meta_baseline.models.utils_meta import load_model # Use the one from utils_meta

class NaturalisticDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load all image paths and labels
        self.same_dir = os.path.join(data_dir, 'same')
        self.diff_dir = os.path.join(data_dir, 'different')
        
        self.same_images = sorted(glob.glob(os.path.join(self.same_dir, '*.png')))
        self.diff_images = sorted(glob.glob(os.path.join(self.diff_dir, '*.png')))
        
        # Create image paths and labels
        self.images = []
        self.labels = []
        
        # Same pairs
        for i in range(0, len(self.same_images), 2):
            if i + 1 < len(self.same_images):
                self.images.append(self.same_images[i])  # Each image contains both objects
                self.labels.append(1)  # 1 for same
        
        # Different pairs
        for i in range(0, len(self.diff_images), 2):
            if i + 1 < len(self.diff_images):
                self.images.append(self.diff_images[i])  # Each image contains both objects
                self.labels.append(0)  # 0 for different
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load single image containing both objects
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['conv2', 'conv4'], help='Model architecture to use')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define model paths based on model type
    if args.model == 'conv2':
        model_dir = 'conv2lr_runs_20250127_131933'
        model_class = Conv2CNN
        load_model_fn = load_model # Use the imported load_model
    else:  # conv4
        model_dir = 'exp1_(finished)conv4lr_runs_20250126_201548'
        model_class = Conv4CNN
        load_model_fn = load_model # Use the imported load_model
    
    # Construct weight path
    weight_path = os.path.join(model_dir, f'seed_{args.seed}', f'model_seed_{args.seed}_pretesting.pt')
    print(f"Loading weights from: {weight_path}")

    # Initialize model
    model = model_class()
    
    # Load weights
    model = load_model_fn(model, weight_path, device)
    model.eval()

    # Load test data
    test_dataset = NaturalisticDataset(args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Test the model
    all_preds = []
    all_labels = []
    all_accuracies = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass with single image containing both objects
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate accuracy for this batch
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            all_accuracies.append(accuracy)
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Accuracy: {accuracy:.4f}')

    # Calculate overall accuracy
    overall_accuracy = np.mean(all_accuracies)
    print(f'\nOverall Accuracy: {overall_accuracy:.4f}')

    # Save results
    results = {
        'model': args.model,
        'seed': args.seed,
        'accuracy': overall_accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'per_batch_accuracies': all_accuracies
    }
    
    output_file = os.path.join(args.output_dir, f'{args.model}_seed_{args.seed}_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f'Results saved to {output_file}')

if __name__ == '__main__':
    main() 