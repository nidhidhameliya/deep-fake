"""
Utility functions for Deepfake Detection project
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def create_directory_structure():
    """Create necessary directories if they don't exist"""
    directories = [
        'models',
        'logs',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Directory '{directory}' ready")


def load_json(file_path):
    """
    Load JSON file
    
    Args:
        file_path (str): Path to JSON file
    
    Returns:
        dict: Loaded JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """
    Save data to JSON file
    
    Args:
        data (dict): Data to save
        file_path (str): Path to save file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def get_image_files(directory):
    """
    Get all image files in directory
    
    Args:
        directory (str): Directory path
    
    Returns:
        list: List of image file paths
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    
    image_files = []
    for file in Path(directory).rglob('*'):
        if file.suffix.lower() in valid_extensions:
            image_files.append(str(file))
    
    return sorted(image_files)


def count_images_in_dataset(dataset_path):
    """
    Count images in dataset (supports both old and new structure)
    
    Args:
        dataset_path (str): Path to dataset
    
    Returns:
        dict: Image counts by category
    """
    return update_count_images_in_dataset(dataset_path)
    
    return counts


def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history (dict): Training history with keys: train_loss, val_loss, train_accuracy, val_accuracy
        save_path (str): Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_accuracy'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    return fig


def create_sample_batch_visualization(images, labels, class_names, save_path=None):
    """
    Create visualization of a batch of images
    
    Args:
        images (Tensor): Batch of images [B, 3, H, W]
        labels (Tensor): Batch of labels [B]
        class_names (list): List of class names
        save_path (str): Path to save visualization (optional)
    """
    batch_size = min(9, images.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(batch_size):
        # Denormalize image
        image = images[i].numpy()
        image = np.transpose(image, (1, 2, 0))
        
        # Denormalize (ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std[:, np.newaxis, np.newaxis] + mean[:, np.newaxis, np.newaxis]
        image = np.clip(image, 0, 1)
        
        # Plot
        axes[i].imshow(image)
        label_idx = labels[i].item()
        axes[i].set_title(f"Label: {class_names[label_idx]}")
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(batch_size, 9):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Batch visualization saved to {save_path}")
    
    return fig


def get_class_distribution(dataset_path, mode='binary'):
    """
    Get class distribution statistics
    
    Args:
        dataset_path (str): Path to dataset
        mode (str): 'binary' or 'generator'
    
    Returns:
        dict: Class distribution information
    """
    counts = count_images_in_dataset(dataset_path)
    
    if mode == 'binary':
        total = counts['real'] + counts['fake_total']
        return {
            'Real': counts['real'],
            'Fake': counts['fake_total'],
            'Total': total,
            'Real %': f"{100 * counts['real'] / total:.2f}%",
            'Fake %': f"{100 * counts['fake_total'] / total:.2f}%"
        }
    
    elif mode == 'generator':
        total = counts['fake_total']
        distribution = {}
        for gen_name, count in counts['generators'].items():
            distribution[gen_name] = {
                'count': count,
                'percentage': f"{100 * count / total:.2f}%" if total > 0 else "0%"
            }
        distribution['Total'] = total
        return distribution


def print_dataset_statistics(dataset_path):
    """Print dataset statistics"""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    # Binary distribution
    binary_dist = get_class_distribution(dataset_path, mode='binary')
    print("\nBinary Classification (Real vs Fake):")
    print(f"  Real images:  {binary_dist['Real']:>6} ({binary_dist['Real %']})")
    print(f"  Fake images:  {binary_dist['Fake']:>6} ({binary_dist['Fake %']})")
    print(f"  Total:        {binary_dist['Total']:>6}")
    
    # Generator distribution
    gen_dist = get_class_distribution(dataset_path, mode='generator')
    print(f"\nGenerator Distribution ({gen_dist['Total']} fake images):")
    
    # Sort by count
    sorted_gens = sorted(
        [(name, info) for name, info in gen_dist.items() if name != 'Total'],
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    for gen_name, info in sorted_gens:
        print(f"  {gen_name:30} {info['count']:>5} ({info['percentage']})")
    
    print("=" * 70 + "\n")


def setup_logging(log_file='training.log'):
    """
    Setup logging configuration
    
    Args:
        log_file (str): Path to log file
    """
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def delete_old_models(model_paths=None):
    """
    Delete old model files
    
    Args:
        model_paths (list): List of model file paths to delete. 
                           Defaults to ['binary_model.pth', 'generator_model.pth']
    
    Returns:
        dict: Summary of deleted files
    """
    if model_paths is None:
        model_paths = ['binary_model.pth', 'generator_model.pth']
    
    deleted = []
    not_found = []
    
    for model_path in model_paths:
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
                deleted.append(model_path)
                print(f"✓ Deleted: {model_path}")
            else:
                not_found.append(model_path)
                print(f"- File not found: {model_path}")
        except Exception as e:
            print(f"✗ Error deleting {model_path}: {e}")
    
    return {
        'deleted': deleted,
        'not_found': not_found,
        'total': len(deleted)
    }


def check_dataset_structure(dataset_path):
    """
    Check if dataset has proper structure (Real/ and Fake/ folders)
    
    Args:
        dataset_path (str): Path to dataset root
    
    Returns:
        dict: Structure information
    """
    dataset_path = Path(dataset_path)
    structure = {
        'valid': False,
        'real_folder': None,
        'fake_folder': None,
        'generators': [],
        'messages': []
    }
    
    # Check for new structure
    real_new = dataset_path / 'Real'
    fake_new = dataset_path / 'Fake'
    
    # Check for old structure
    real_old = dataset_path / 'real_images'
    fake_old = dataset_path / 'fake_images'
    
    if real_new.exists() and fake_new.exists():
        structure['real_folder'] = 'Real'
        structure['fake_folder'] = 'Fake'
        structure['valid'] = True
        structure['messages'].append("✓ Using new dataset structure (Real/ and Fake/)")
    elif real_old.exists() and fake_old.exists():
        structure['real_folder'] = 'real_images'
        structure['fake_folder'] = 'fake_images'
        structure['valid'] = True
        structure['messages'].append("⚠ Using legacy dataset structure (real_images/ and fake_images/)")
        structure['messages'].append("  Please consider reorganizing to new structure (Real/ and Fake/)")
    else:
        structure['messages'].append("✗ Dataset structure is incomplete or missing")
        if real_new.exists() or real_old.exists():
            structure['messages'].append("  Real folder found but Fake folder missing")
        if fake_new.exists() or fake_old.exists():
            structure['messages'].append("  Fake folder found but Real folder missing")
        return structure
    
    # Get generator list
    fake_path = dataset_path / structure['fake_folder']
    if fake_path.exists():
        generators = sorted([
            folder.name for folder in fake_path.iterdir() 
            if folder.is_dir()
        ])
        structure['generators'] = generators
        structure['messages'].append(f"Found {len(generators)} generator(s)")
    
    return structure


def update_count_images_in_dataset(dataset_path):
    """
    Count images in dataset (supports both old and new structure)
    
    Args:
        dataset_path (str): Path to dataset
    
    Returns:
        dict: Image counts by category
    """
    dataset_path = Path(dataset_path)
    counts = {
        'real': 0,
        'fake_total': 0,
        'generators': {},
        'structure': 'unknown'
    }
    
    # Check structure
    structure_info = check_dataset_structure(dataset_path)
    
    if not structure_info['valid']:
        print("Warning: Dataset structure is not valid")
        return counts
    
    counts['structure'] = structure_info['real_folder']
    
    # Count real images
    real_path = dataset_path / structure_info['real_folder']
    if real_path.exists():
        counts['real'] = len(get_image_files(str(real_path)))
    
    # Count fake images
    fake_path = dataset_path / structure_info['fake_folder']
    if fake_path.exists():
        for gen_folder in fake_path.iterdir():
            if gen_folder.is_dir():
                gen_images = len(get_image_files(str(gen_folder)))
                counts['generators'][gen_folder.name] = gen_images
                counts['fake_total'] += gen_images
    
    return counts


if __name__ == '__main__':
    # Example usage
    dataset_path = 'Real vs Fake(AI) Image Dataset'
    
    if os.path.exists(dataset_path):
        print_dataset_statistics(dataset_path)
    else:
        print(f"Dataset not found at {dataset_path}")
