import os
import torch
import torch.nn.functional as F
from conv6lr import SameDifferentCNN, SameDifferentDataset, accuracy, collate_episodes
import argparse
from torch.utils.data import DataLoader
import json
import learn2learn as l2l
from tqdm import tqdm
import gc

# PB tasks used in leave-one-out training
PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

def test_checkpoint(checkpoint_path, pb_data_dir, held_out_task, batch_size=16):
    try:
        # Setup device
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Running on CPU, but this will be slow and may not work correctly.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        
        # Verify paths exist
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not os.path.exists(pb_data_dir):
            raise FileNotFoundError(f"PB data directory not found: {pb_data_dir}")
            
        # Load checkpoint with appropriate device mapping
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' not in checkpoint:
                raise KeyError("Checkpoint does not contain model_state_dict")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint: {str(e)}")
        
        # Create model and load state
        model = SameDifferentCNN().to(device)
        
        # First load the base model state dict
        # Remove learning rate parameters from state dict
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                     if not k.startswith('lr_')}
        
        # Load state dict with strict=False to ignore missing temperature parameter
        model.load_state_dict(state_dict, strict=False)
        
        # Create test dataset for held-out task
        try:
            test_dataset = SameDifferentDataset(pb_data_dir, [held_out_task], 'test')
            if len(test_dataset) == 0:
                raise ValueError(f"No test data found for task {held_out_task}")
        except Exception as e:
            raise RuntimeError(f"Error loading test dataset: {str(e)}")
            
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_episodes
        )
        
        # Test the model
        model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        print(f"\nTesting checkpoint: {checkpoint_path}")
        print(f"Testing held-out task: {held_out_task}")
        print(f"No adaptation - directly evaluating on query sets")
        print(f"Dataset size: {len(test_dataset)} episodes")
        
        # Use tqdm for progress tracking
        for episodes in tqdm(test_loader, desc="Testing episodes"):
            batch_loss = 0
            batch_acc = 0
            
            for episode in episodes:
                try:
                    # Move data to GPU - we only need query data since we're not adapting
                    query_images = episode['query_images'].to(device, non_blocking=True)
                    query_labels = episode['query_labels'].unsqueeze(1).to(device, non_blocking=True)
                    
                    # Directly evaluate on query set without adaptation
                    with torch.no_grad():
                        query_preds = model(query_images)
                        query_loss = F.binary_cross_entropy_with_logits(
                            query_preds[:, 1], query_labels.squeeze(1).float())
                        query_acc = accuracy(query_preds, query_labels)
                        
                        batch_loss += query_loss.item()
                        batch_acc += query_acc.item()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: GPU OOM error. Trying to recover...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        continue
                    else:
                        raise e
            
            # Average over episodes
            avg_batch_loss = batch_loss / len(episodes)
            avg_batch_acc = batch_acc / len(episodes)
            
            total_loss += avg_batch_loss
            total_acc += avg_batch_acc
            num_batches += 1
            
            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        final_loss = total_loss / num_batches
        final_acc = total_acc / num_batches
        
        print(f"Test Loss: {final_loss:.4f}")
        print(f"Test Accuracy: {final_acc:.4f}")
        
        return {
            'loss': final_loss,
            'accuracy': final_acc,
            'adaptation_steps': 0,  # Indicate no adaptation was used
            'checkpoint': checkpoint_path,
            'held_out_task': held_out_task,
            'num_episodes': len(test_dataset),
            'best_val_acc': checkpoint.get('val_acc', None),
            'best_epoch': checkpoint.get('epoch', None)
        }
    
    except Exception as e:
        print(f"Error in test_checkpoint: {str(e)}")
        return None

def main(args):
    # Validate task
    if args.held_out_task not in PB_TASKS:
        raise ValueError(f"Invalid task {args.held_out_task}. Must be one of {PB_TASKS}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get checkpoint path for this held-out task and seed
    checkpoint_path = os.path.join(
        "results/leave_one_out_seeds",  # Base directory from experiment1
        args.held_out_task,            # Task-specific subdirectory
        f"seed_{args.seed}",           # Seed-specific subdirectory
        "best_model.pt"                # Best model checkpoint
    )
    
    # Test the checkpoint
    results = test_checkpoint(
        checkpoint_path,
        args.pb_data_dir,
        args.held_out_task,
        args.batch_size
    )
    
    if results is None:
        raise RuntimeError(f"Testing failed for task {args.held_out_task}")
    
    # Save results - include seed in filename
    output_file = os.path.join(args.output_dir, f'test_results_no_adapt_{args.held_out_task}_seed_{args.seed}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    if results['best_val_acc'] is not None:
        print(f"Best Validation Accuracy: {results['best_val_acc']:.4f}")
    if results['best_epoch'] is not None:
        print(f"Best Epoch: {results['best_epoch']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_data_dir', type=str, required=True,
                       help='Path to PB dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--held_out_task', type=str, required=True,
                       help='The PB task that was held out during training')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, required=True,
                       help='Which seed model to test')
    
    args = parser.parse_args()
    main(args) 