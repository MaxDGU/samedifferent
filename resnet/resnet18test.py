import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from learn2learn.data import MetaDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
from resnet18 import get_cosine_schedule_with_warmup, SameDifferentDataset

class SameDifferentResNet18(nn.Module):
    def __init__(self):
        super(SameDifferentResNet18, self).__init__()
        
        # Load pre-trained ResNet18 without classification layer
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Remove the final FC layer and replace with our own classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2)  # Binary classification
        )
        
        # Initialize learned learning rates
        self.lr_conv1 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_bn1 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_layer1 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_layer2 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_layer3 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_layer4 = nn.Parameter(torch.ones(1) * 0.01)
        self.lr_classifier = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.01) for _ in range(3)
        ])
    
    def get_layer_lrs(self):
        """Return a dictionary mapping parameters to their learning rates"""
        lrs = {}
        
        # ResNet main layers
        for name, param in self.resnet.conv1.named_parameters():
            lrs[f'resnet.conv1.{name}'] = self.lr_conv1.abs()
        
        for name, param in self.resnet.bn1.named_parameters():
            lrs[f'resnet.bn1.{name}'] = self.lr_bn1.abs()
        
        for i, layer in enumerate([self.resnet.layer1, self.resnet.layer2, 
                                 self.resnet.layer3, self.resnet.layer4]):
            lr = getattr(self, f'lr_layer{i+1}').abs()
            for name, param in layer.named_parameters():
                lrs[f'resnet.layer{i+1}.{name}'] = lr
        
        # Classifier layers - only get learning rates for linear layers
        linear_idx = 0
        for name, module in self.classifier.named_children():
            if isinstance(module, nn.Linear):
                for param_name, param in module.named_parameters():
                    lrs[f'classifier.{name}.{param_name}'] = self.lr_classifier[linear_idx].abs()
                linear_idx += 1
        
        return lrs
    
    def forward(self, x):
        # ResNet feature extraction (up to global average pooling)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Global average pooling
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        return self.classifier(x)

def accuracy(predictions, targets):
    predictions = (torch.sigmoid(predictions[:, 1]) > 0.5).float()
    return (predictions == targets.squeeze(1)).float().mean()

def validate(maml, val_dataset, device, meta_batch_size=8, num_adaptation_steps=5, max_episodes=200):
    """Validation with proper model mode and loss handling"""
    maml.module.eval()  # Ensure model is in eval mode
    val_loss = 0.0
    val_acc = 0.0
    
    # Calculate number of batches
    num_tasks = len(val_dataset.tasks)
    episodes_per_task = max_episodes // num_tasks
    num_batches = max_episodes // meta_batch_size
    
    task_metrics = {task: {'acc': [], 'loss': []} for task in val_dataset.tasks}
    pbar = tqdm(range(num_batches), desc="Validating")
    
    for _ in pbar:
        batch_loss = 0.0
        batch_acc = 0.0
        
        # Get balanced batch like in training
        episodes = val_dataset.get_balanced_batch(meta_batch_size)
        
        for episode in episodes:
            task = episode['task']
            learner = maml.clone()
            
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].unsqueeze(1).to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].unsqueeze(1).to(device)
            
            # Inner loop adaptation
            layer_lrs = learner.module.get_layer_lrs()
            for _ in range(num_adaptation_steps):
                support_preds = learner(support_images)
                # Use BCE loss for support set
                support_loss = F.binary_cross_entropy_with_logits(
                    support_preds[:, 1], support_labels.squeeze(1).float())
                
                grads = torch.autograd.grad(support_loss, learner.parameters(),
                                          create_graph=True,
                                          allow_unused=True)
                
                for (name, param), grad in zip(learner.named_parameters(), grads):
                    if grad is not None:
                        lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                        param.data = param.data - lr.abs() * grad
            
            with torch.no_grad():
                query_preds = learner(query_images)
                # Use BCE loss for query set
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                query_acc = accuracy(query_preds, query_labels)
            
            batch_loss += query_loss.item()
            batch_acc += query_acc.item()
            
            task_metrics[task]['acc'].append(query_acc.item())
            task_metrics[task]['loss'].append(query_loss.item())
        
        batch_loss /= len(episodes)
        batch_acc /= len(episodes)
        val_loss += batch_loss
        val_acc += batch_acc
        
        # Calculate running task accuracies
        task_accs = {task: np.mean(metrics['acc']) if metrics['acc'] else 0.0
                    for task, metrics in task_metrics.items()}
        
        # Show progress with task-specific accuracies
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'acc': f'{batch_acc:.4f}',
            'task_accs': {t: f'{acc:.2f}' for t, acc in task_accs.items()}
        })
    
    print("\nValidation Results by Task:")
    for task in task_metrics:
        task_acc = np.mean(task_metrics[task]['acc']) if task_metrics[task]['acc'] else 0.0
        task_loss = np.mean(task_metrics[task]['loss']) if task_metrics[task]['loss'] else 0.0
        print(f"{task}: Acc = {task_acc:.4f}, Loss = {task_loss:.4f}")
    
    return val_loss / num_batches, val_acc / num_batches

def main():
    # Configuration for quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_batch_size = 8  # Reduced batch size
    num_epochs = 5 # Very short training
    adaptation_steps = 4  # Fewer adaptation steps
    max_batches_per_epoch = 20  # Limit batches per epoch
    
    # Split tasks
    all_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled', 
                 'random_color', 'arrows', 'irregular', 'filled', 'original']
    test_task = 'original'
    train_tasks = [task for task in all_tasks if task != test_task]
    
    # Create datasets
    train_dataset = SameDifferentDataset('data/meta_h5', train_tasks, 'train')
    val_dataset = SameDifferentDataset('data/meta_h5', train_tasks, 'val')
    test_dataset = SameDifferentDataset('data/meta_h5', [test_task], 'test')
    
    # Create model
    model = SameDifferentResNet18()
    model = model.to(device)
    
    maml = l2l.algorithms.MAML(model, 
                              lr=None,  # We're using per-parameter learning rates
                              first_order=True,
                              allow_unused=True)
    
    opt = torch.optim.AdamW(maml.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training loop with validation every epoch
    print("Starting mini training run...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        pbar = tqdm(range(max_batches_per_epoch), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx in pbar:
            batch_loss = 0.0
            batch_acc = 0.0
            
            # Get balanced batch
            episodes = train_dataset.get_balanced_batch(meta_batch_size)
            
            for episode in episodes:
                learner = maml.clone()
                
                # Get episode data
                support_images = episode['support_images'].to(device)
                support_labels = episode['support_labels'].unsqueeze(1).to(device)
                query_images = episode['query_images'].to(device)
                query_labels = episode['query_labels'].unsqueeze(1).to(device)
                
                # Inner loop adaptation
                layer_lrs = learner.module.get_layer_lrs()
                for _ in range(adaptation_steps):
                    support_preds = learner(support_images)
                    support_loss = F.binary_cross_entropy_with_logits(
                        support_preds[:, 1], support_labels.squeeze(1).float())
                    
                    grads = torch.autograd.grad(support_loss, learner.parameters(),
                                              create_graph=True, allow_unused=True)
                    
                    for (name, param), grad in zip(learner.named_parameters(), grads):
                        if grad is not None:
                            lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                            param.data = param.data - lr.abs() * grad
                
                # Compute query loss
                query_preds = learner(query_images)
                query_loss = F.binary_cross_entropy_with_logits(
                    query_preds[:, 1], query_labels.squeeze(1).float())
                query_acc = accuracy(query_preds, query_labels)
                
                batch_loss += query_loss
                batch_acc += query_acc.item()
            
            # Average over episodes
            batch_loss = batch_loss / len(episodes)
            batch_acc = batch_acc / len(episodes)
            
            # Optimization step
            opt.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=1.0)
            opt.step()
            
            running_loss += batch_loss.item()
            running_acc += batch_acc
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'acc': f'{batch_acc:.4f}'
            })
        
        # Validation after each epoch
        print(f"\nRunning validation after epoch {epoch+1}...")
        val_loss, val_acc = validate(maml, val_dataset, device, 
                                   meta_batch_size=4,  # Smaller batch size for quick validation
                                   num_adaptation_steps=3,  # Fewer adaptation steps
                                   max_episodes=20)  # Fewer validation episodes
        
        print(f"Epoch {epoch+1} Results:")
        print(f"Train Loss: {running_loss/max_batches_per_epoch:.4f}")
        print(f"Train Acc: {running_acc/max_batches_per_epoch:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f}")

if __name__ == '__main__':
    main() 