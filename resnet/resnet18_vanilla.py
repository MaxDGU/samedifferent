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
import torchvision.models as models
import json
import argparse

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # Load pretrained ResNet18 and modify for our task
        self.model = models.resnet18(pretrained=False)
        
        # Modify first conv layer to accept our image size and channels
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fc layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        # Add batch norm with track_running_stats=False to match conv6.py
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
        
        # Add dropout to match conv6.py's regularization
        self.dropout2d = nn.Dropout2d(0.1)
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Apply spatial dropout after each major block
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.dropout2d(x)
        
        x = self.model.layer1(x)
        x = self.dropout2d(x)
        
        x = self.model.layer2(x)
        x = self.dropout2d(x)
        
        x = self.model.layer3(x)
        x = self.dropout2d(x)
        
        x = self.model.layer4(x)
        x = self.dropout2d(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.model.fc(x)
        
        return x

def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = F.cross_entropy(outputs, labels.long())
        loss.backward()
        
        optimizer.step()
        
        # Calculate accuracy
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        running_loss += loss.item()
        running_acc += acc.item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.4f}',
            'acc': f'{running_acc/len(train_loader):.4f}'
        })
    
    return running_loss / len(train_loader), running_acc / len(train_loader)

def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels.long())
            
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean()
            
            val_loss += loss.item()
            val_acc += acc.item()
    
    return val_loss / len(val_loader), val_acc / len(val_loader)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val = None
        self.should_stop = False
    
    def __call__(self, acc):
        if self.best_val is None:
            self.best_val = acc
        elif acc < self.best_val + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_val = acc
            self.counter = 0

# Add SameDifferentDataset class from conv6.py
class SameDifferentDataset(Dataset):
    def __init__(self, data_dir, problem_number, split):
        """Dataset for loading same-different task PNG data for a specific problem."""
        self.data_dir = data_dir
        self.problem_number = problem_number
        self.split = split
        
        # Find the correct problem directory (ignoring timestamps)
        problem_pattern = f'results_problem_{problem_number}_*'
        problem_dirs = glob.glob(os.path.join(data_dir, problem_pattern))
        if not problem_dirs:
            raise ValueError(f"No directory found for problem {problem_number}")
        
        # Use the first matching directory
        problem_dir = problem_dirs[0]
        split_dir = os.path.join(problem_dir, split)
        
        # Get all PNG files
        self.image_paths = glob.glob(os.path.join(split_dir, '*.png'))
        if not self.image_paths:
            raise ValueError(f"No PNG files found in {split_dir}")
        
        # Extract labels from filenames (sample_1_0009 -> 1)
        self.labels = []
        for path in self.image_paths:
            filename = os.path.basename(path)
            label = int(filename.split('_')[1])  # Get the middle number
            self.labels.append(label)
        
        self.labels = torch.tensor(self.labels)
        print(f"Loaded {len(self.image_paths)} images for {split} split")
        print(f"Label distribution: {torch.bincount(self.labels.long())}")
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        label = self.labels[idx]
        
        return {
            'image': image,
            'label': label
        }

def main(seed=None):
    # Set random seeds if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 100
    data_dir = 'svrt/data'
    problems = ['1', '7', '5', '15', '16', '19', '20', '21', '22']  # All 9 SVRT problems
    
    # Store results for all problems
    all_results = {}
    
    for problem_number in problems:
        print(f"\n{'='*50}")
        print(f"Training on SVRT problem {problem_number}")
        print(f"{'='*50}\n")
        
        # Create datasets and dataloaders
        train_dataset = SameDifferentDataset(data_dir, problem_number, 'train')
        val_dataset = SameDifferentDataset(data_dir, problem_number, 'test')  # Using test split as validation
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Create fresh model for each problem
        model = ResNet18().to(device)
        print(f"Model created on {device}")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=20)
        best_val_acc = 0.0
        best_train_acc = 0.0
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping check on validation accuracy
            early_stopping(val_acc)
            if early_stopping.should_stop:
                print("Early stopping triggered!")
                break
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc  # Store train acc when we get best val acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, f'best_model_resnet18_problem_{problem_number}.pt')
        
        # Load best model for final evaluation
        checkpoint = torch.load(f'best_model_resnet18_problem_{problem_number}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final validation pass
        final_val_loss, final_val_acc = validate(model, val_loader, device)
        print(f"\nResults for SVRT Problem {problem_number}:")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Best Train Accuracy: {best_train_acc:.4f}")
        
        # Store results
        all_results[problem_number] = {
            'val_loss': final_val_loss,
            'val_acc': final_val_acc,
            'best_val_acc': best_val_acc,
            'best_train_acc': best_train_acc,
            'epochs_trained': epoch + 1
        }
    
    # Print summary of all results
    print("\n" + "="*50)
    print("Summary of Results:")
    print("="*50)
    for problem, results in all_results.items():
        print(f"\nProblem {problem}:")
        print(f"Validation Accuracy: {results['val_acc']:.4f}")
        print(f"Validation Loss: {results['val_loss']:.4f}")
        print(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
        print(f"Best Train Accuracy: {results['best_train_acc']:.4f}")
        print(f"Epochs Trained: {results['epochs_trained']}")
    
    # Save all results with seed in filename
    json_filename = f'resnet18_all_problems_results_seed{seed}.json'
    print(f"\nSaving results to {json_filename}")
    print("Results to be saved:", all_results)
    
    with open(json_filename, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved successfully to {json_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    args = parser.parse_args()
    
    print(f"\nStarting training with seed {args.seed}")
    main(seed=args.seed) 