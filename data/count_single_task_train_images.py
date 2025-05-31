import h5py
import numpy as np
from pathlib import Path
import random
import os
import argparse
from PIL import Image, ImageDraw # Using Pillow for image manipulation

# TODO: Potentially add more robust image transformation libraries if needed, e.g., OpenCV for complex ops

def load_object_paths(objects_root_dir, allowed_extensions=('.jpg', '.jpeg', '.png')):
    """
    Scans the objects_root_dir for all unique object image files.
    """
    object_paths = []
    for root, _, files in os.walk(objects_root_dir):
        for f_name in files:
            if f_name.lower().endswith(allowed_extensions):
                object_paths.append(Path(root) / f_name)
    if not object_paths:
        raise FileNotFoundError(f"No object images found in {objects_root_dir}")
    random.shuffle(object_paths) # Shuffle once at the beginning
    print(f"Found {len(object_paths)} object image files.")
    return object_paths

def create_single_pair_image(obj1_path, obj2_path=None, canvas_size=(128, 128), resize_dim=(50,50)):
    """
    Creates an image with one or two objects placed on a canvas.
    If obj2_path is None, it's a "same" pair (two instances of obj1).
    Otherwise, it's a "different" pair (obj1 and obj2).

    Returns:
        np.array: Image as a NumPy array (H, W, C) uint8, or None if creation fails.
    """
    try:
        img1 = Image.open(obj1_path).convert("RGBA") # Use RGBA for transparency handling during rotation
        img1 = img1.resize(resize_dim)

        if obj2_path: # Different pair
            img2 = Image.open(obj2_path).convert("RGBA")
            img2 = img2.resize(resize_dim)
        else: # Same pair
            img2 = img1.copy() # Second instance of the same object

        # Create a white background canvas
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))

        # --- Placeholder for robust object placement and transformation ---
        # This is a simplified placement. Real implementation needs:
        # - Random rotation for each object instance.
        # - Random scaling (within limits) for each object instance.
        # - Random placement on the canvas, ensuring they are mostly visible.
        # - Logic to avoid excessive overlap or ensure distinctness.

        # Example: Simple rotation and placement
        angle1 = random.uniform(0, 360)
        img1_rotated = img1.rotate(angle1, expand=True, fillcolor=(0,0,0,0)) # Transparent fill for rotation
        
        angle2 = random.uniform(0, 360)
        img2_rotated = img2.rotate(angle2, expand=True, fillcolor=(0,0,0,0))

        # Simplified placement (top-left and bottom-right corners, needs improvement)
        # Position 1 (ensure it's within bounds after rotation)
        w1, h1 = img1_rotated.size
        x1 = random.randint(0, max(0, canvas_size[0] // 2 - w1))
        y1 = random.randint(0, max(0, canvas_size[1] - h1))
        
        # Position 2 (ensure it's within bounds and somewhat separated)
        w2, h2 = img2_rotated.size
        x2 = random.randint(canvas_size[0] // 2, max(canvas_size[0] // 2, canvas_size[0] - w2))
        y2 = random.randint(0, max(0, canvas_size[1] - h2))

        # Paste using the alpha channel as a mask for transparency
        canvas.paste(img1_rotated, (x1, y1), img1_rotated)
        canvas.paste(img2_rotated, (x2, y2), img2_rotated)
        # --- End Placeholder ---

        return np.array(canvas)

    except Exception as e:
        print(f"Error creating single pair image: obj1={obj1_path}, obj2={obj2_path}, Error: {e}")
        return None


def generate_episodes_for_split(split_name, num_episodes_target, all_object_paths_split,
                                support_sizes_pool, query_size_fixed,
                                canvas_size, hdf5_file_path):
    """
    Generates and saves all episodes for a given split (train, val, test).
    """
    print(f"\nGenerating {split_name} split with {num_episodes_target} episodes...")
    
    episodes_data = []
    num_objects_in_split = len(all_object_paths_split)
    if num_objects_in_split == 0:
        print(f"Warning: No objects provided for {split_name} split. Skipping generation.")
        return

    for ep_idx in range(num_episodes_target):
        if (ep_idx + 1) % 100 == 0:
            print(f"  Generating {split_name} episode {ep_idx + 1}/{num_episodes_target}...")

        # 1. Select a Reference Object for this episode
        # Cycle through objects to ensure variety, or pick randomly if many objects
        ref_obj_path = all_object_paths_split[ep_idx % num_objects_in_split]

        # 2. Determine Support Set Size for this episode
        current_support_size = random.choice(support_sizes_pool)
        
        episode_support_images = []
        episode_support_labels = []
        episode_query_images = []
        episode_query_labels = []

        # 3. Generate Support Set (balanced same/different)
        num_same_support = current_support_size // 2
        num_diff_support = current_support_size - num_same_support # Handles odd sizes if any

        # Create "same" support examples
        for _ in range(num_same_support):
            img_arr = create_single_pair_image(ref_obj_path, None, canvas_size)
            if img_arr is not None:
                episode_support_images.append(img_arr)
                episode_support_labels.append(1)
        
        # Create "different" support examples
        for _ in range(num_diff_support):
            other_obj_path = ref_obj_path
            attempts = 0
            while other_obj_path == ref_obj_path and attempts < num_objects_in_split * 2 : # Ensure 'other' is actually different
                other_obj_path = random.choice(all_object_paths_split)
                attempts+=1
            if other_obj_path == ref_obj_path and num_objects_in_split > 1: # Still same after many tries (e.g. only 1 object in split)
                 print(f"Warning: Could not find a different object for support set in episode {ep_idx} for {split_name}. Using same obj.")
                 # Fallback: create another "same" or skip, depending on strictness. Here, we'll make it 'different' by label.

            img_arr = create_single_pair_image(ref_obj_path, other_obj_path, canvas_size)
            if img_arr is not None:
                episode_support_images.append(img_arr)
                episode_support_labels.append(0)
        
        # Shuffle support set
        support_combined = list(zip(episode_support_images, episode_support_labels))
        random.shuffle(support_combined)
        if support_combined:
            episode_support_images, episode_support_labels = zip(*support_combined)
        else: # Handle cases where no support images were generated
             episode_support_images, episode_support_labels = [], []


        # 4. Generate Query Set (Size Q_fixed - BALANCED)
        # For Q=3, 8 patterns: (000, 001, 010, 011, 100, 101, 110, 111)
        # We will aim for a random selection from these patterns for balance.
        
        # Ensure we have enough images to form a query set based on generated support
        if not episode_support_images or len(episode_support_images) < current_support_size:
            print(f"Warning: Episode {ep_idx} for {split_name} has insufficient support images ({len(episode_support_images)}/{current_support_size}). Skipping query generation for this episode.")
        else:
            # Simplified: For each query item, decide label randomly, then generate.
            # A more robust way for perfect balance across dataset would be to cycle through patterns.
            for _ in range(query_size_fixed):
                query_label = random.choice([0, 1]) # 50/50 chance for 0 or 1
                if query_label == 1: # "same"
                    img_arr = create_single_pair_image(ref_obj_path, None, canvas_size)
                else: # "different"
                    other_obj_path = ref_obj_path
                    attempts = 0
                    while other_obj_path == ref_obj_path and attempts < num_objects_in_split *2 :
                        other_obj_path = random.choice(all_object_paths_split)
                        attempts +=1
                    if other_obj_path == ref_obj_path and num_objects_in_split > 1 :
                         print(f"Warning: Could not find a different object for query set in episode {ep_idx} for {split_name}. Using same obj.")

                    img_arr = create_single_pair_image(ref_obj_path, other_obj_path, canvas_size)
                
                if img_arr is not None:
                    episode_query_images.append(img_arr)
                    episode_query_labels.append(query_label)
            
        # Check if enough images were generated for the episode
        if len(episode_support_images) == current_support_size and \
           len(episode_query_images) == query_size_fixed:
            episodes_data.append({
                'support_images': np.array(episode_support_images, dtype=np.uint8),
                'support_labels': np.array(episode_support_labels, dtype=np.int32),
                'query_images': np.array(episode_query_images, dtype=np.uint8),
                'query_labels': np.array(episode_query_labels, dtype=np.int32),
            })
        else:
            print(f"Skipping episode {ep_idx} for {split_name} due to insufficient images generated."
                  f" Support: {len(episode_support_images)}/{current_support_size},"
                  f" Query: {len(episode_query_images)}/{query_size_fixed}")


    # 5. Save to HDF5
    if not episodes_data:
        print(f"No valid episodes generated for {split_name}. HDF5 file will not be created.")
        return

    print(f"Saving {len(episodes_data)} episodes to {hdf5_file_path}...")
    with h5py.File(hdf5_file_path, 'w') as f:
        for idx, episode_dict in enumerate(episodes_data):
            ep_group = f.create_group(f'episode_{idx:06d}')
            ep_group.create_dataset('support_images', data=episode_dict['support_images'], compression="gzip")
            ep_group.create_dataset('support_labels', data=episode_dict['support_labels'])
            ep_group.create_dataset('query_images', data=episode_dict['query_images'], compression="gzip")
            ep_group.create_dataset('query_labels', data=episode_dict['query_labels'])
    print(f"Successfully saved {hdf5_file_path}")


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    objects_root = Path(args.objects_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_objects = load_object_paths(objects_root)
    if not all_objects:
        return

    # Split objects for train/val/test to ensure no overlap
    # Example split: 70% train, 15% val, 15% test
    num_total_objects = len(all_objects)
    num_train = int(args.train_split_ratio * num_total_objects)
    num_val = int(args.val_split_ratio * num_total_objects)
    
    # Ensure all objects are used if ratios don't sum perfectly due to rounding
    obj_train = all_objects[:num_train]
    obj_val = all_objects[num_train : num_train + num_val]
    obj_test = all_objects[num_train + num_val:]

    print(f"Object splits: Train={len(obj_train)}, Val={len(obj_val)}, Test={len(obj_test)}")

    # Configuration based on typical values observed
    support_sizes = [4, 6, 8, 10]
    query_size = 3 
    canvas_dim = (args.canvas_height, args.canvas_width)

    # Generate Train set
    if args.num_train_episodes > 0 and obj_train:
        generate_episodes_for_split(
            split_name='train',
            num_episodes_target=args.num_train_episodes,
            all_object_paths_split=obj_train,
            support_sizes_pool=support_sizes,
            query_size_fixed=query_size,
            canvas_size=canvas_dim,
            hdf5_file_path=output_root / 'train.h5'
        )

    # Generate Validation set
    if args.num_val_episodes > 0 and obj_val:
        generate_episodes_for_split(
            split_name='val',
            num_episodes_target=args.num_val_episodes,
            all_object_paths_split=obj_val,
            support_sizes_pool=support_sizes,
            query_size_fixed=query_size,
            canvas_size=canvas_dim,
            hdf5_file_path=output_root / 'val.h5'
        )

    # Generate Test set
    if args.num_test_episodes > 0 and obj_test:
        generate_episodes_for_split(
            split_name='test',
            num_episodes_target=args.num_test_episodes,
            all_object_paths_split=obj_test,
            support_sizes_pool=support_sizes,
            query_size_fixed=query_size,
            canvas_size=canvas_dim,
            hdf5_file_path=output_root / 'test.h5'
        )
    
    print("\nDataset generation process complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Naturalistic Same-Different Meta-Learning Episodes.")
    parser.add_argument('--objects_dir', type=str, default='naturalistic/objectsall',
                        help='Root directory containing individual object JPG/PNG images.')
    parser.add_argument('--output_dir', type=str, default='data/naturalistic_new/meta',
                        help='Directory to save the generated HDF5 files (train.h5, val.h5, test.h5).')
    
    parser.add_argument('--num_train_episodes', type=int, default=16800, help='Number of episodes for train.h5.')
    parser.add_argument('--num_val_episodes', type=int, default=3600, help='Number of episodes for val.h5.')
    parser.add_argument('--num_test_episodes', type=int, default=3600, help='Number of episodes for test.h5.')

    parser.add_argument('--train_split_ratio', type=float, default=0.7, help='Fraction of objects for training.')
    parser.add_argument('--val_split_ratio', type=float, default=0.15, help='Fraction of objects for validation.')
    # Test split ratio is inferred (1 - train - val)

    parser.add_argument('--canvas_width', type=int, default=128, help='Width of the canvas for generated images.')
    parser.add_argument('--canvas_height', type=int, default=128, help='Height of the canvas for generated images.')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    parsed_args = parser.parse_args()
    main(parsed_args)
