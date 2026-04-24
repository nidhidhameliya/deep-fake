"""
Dataset handling for Deepfake Detection
Loads images from folder structure and creates PyTorch DataLoaders
"""

import os
from pathlib import Path
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from collections import defaultdict


class DeepfakeDataset(Dataset):
    """
    Custom Dataset for loading real and fake images
    
    Args:
        root_dir (str): Path to dataset root (contains Real/ and Fake/ folders)
        mode (str): 'binary' for real/fake or 'generator' for generator classification
        transform: Image transformations to apply
    """
    
    def __init__(self, root_dir, mode='binary', transform=None):
        """
        Initialize the dataset
        
        Args:
            root_dir: Path to dataset root
            mode: 'binary' (real/fake) or 'generator' (which generator)
            transform: Torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []
        
        if mode == 'binary':
            self._load_binary_dataset()
        elif mode == 'generator':
            self._load_generator_dataset()
        else:
            raise ValueError("Mode must be 'binary' or 'generator'")
    
    def _load_binary_dataset(self):
        """Load dataset for binary classification (Real vs Fake)"""
        
        # Real images (label = 0)
        real_path = self.root_dir / 'Real'
        if real_path.exists():
            for img_file in real_path.glob('*'):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_file))
                    self.labels.append(0)
        else:
            # Fallback to old structure if new structure doesn't exist
            real_path_old = self.root_dir / 'real_images'
            if real_path_old.exists():
                print(f"Warning: Using legacy 'real_images' folder structure")
                for img_file in real_path_old.glob('*'):
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.images.append(str(img_file))
                        self.labels.append(0)
        
        # Fake images (label = 1) - automatically include all subfolders
        fake_path = self.root_dir / 'Fake'
        if fake_path.exists():
            for generator_folder in fake_path.iterdir():
                if generator_folder.is_dir():
                    for img_file in generator_folder.glob('*'):
                        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            self.images.append(str(img_file))
                            self.labels.append(1)
        else:
            # Fallback to old structure if new structure doesn't exist
            fake_path_old = self.root_dir / 'fake_images'
            if fake_path_old.exists():
                print(f"Warning: Using legacy 'fake_images' folder structure")
                for generator_folder in fake_path_old.iterdir():
                    if generator_folder.is_dir():
                        for img_file in generator_folder.glob('*'):
                            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                self.images.append(str(img_file))
                                self.labels.append(1)
        
        self.class_names = ['Real', 'Fake']
        print(f"Loaded {len(self.images)} images for binary classification")
        print(f"  Real images: {sum(1 for l in self.labels if l == 0)}")
        print(f"  Fake images: {sum(1 for l in self.labels if l == 1)}")
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.root_dir}. Expected 'Real/' and 'Fake/' folders.")
    
    def _load_generator_dataset(self):
        """Load dataset for generator classification (which model generated the fake)"""
        
        fake_path = self.root_dir / 'Fake'
        generator_names = []
        
        if fake_path.exists():
            # Get all generator folders (automatically includes subfolders)
            for generator_folder in sorted(fake_path.iterdir()):
                if generator_folder.is_dir():
                    generator_names.append(generator_folder.name)
        else:
            # Fallback to old structure if new structure doesn't exist
            fake_path_old = self.root_dir / 'fake_images'
            if fake_path_old.exists():
                print(f"Warning: Using legacy 'fake_images' folder structure")
                for generator_folder in sorted(fake_path_old.iterdir()):
                    if generator_folder.is_dir():
                        generator_names.append(generator_folder.name)
        
        self.class_names = sorted(generator_names)
        
        if len(self.class_names) == 0:
            raise ValueError(f"No generator folders found in {fake_path}. Expected subfolders in 'Fake/' directory.")
        
        # Load images from each generator folder
        for class_idx, gen_name in enumerate(self.class_names):
            gen_path = fake_path / gen_name if fake_path.exists() else self.root_dir / 'fake_images' / gen_name
            for img_file in gen_path.glob('*'):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_file))
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images for generator classification")
        print(f"Number of generators: {len(self.class_names)}")
        print(f"Generators: {', '.join(self.class_names)}")
    
    def __len__(self):
        """Return total number of images"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get image and label at index
        
        Returns:
            img (Tensor): Transformed image
            label (int): Class label
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Open image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 224, 224), label


def get_data_transforms():
    """
    Define train and validation transforms
    
    Returns:
        dict: Dictionary with 'train' and 'val' transforms
    """
    
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    }
    
    return data_transforms


def create_train_val_loaders(root_dir, mode='binary', batch_size=32, val_split=0.2, num_workers=0):
    """
    Create training and validation DataLoaders
    
    Args:
        root_dir (str): Path to dataset root
        mode (str): 'binary' or 'generator'
        batch_size (int): Batch size for data loading
        val_split (float): Validation split percentage (0.0-1.0)
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    
    # Load full dataset
    transforms_dict = get_data_transforms()
    full_dataset = DeepfakeDataset(root_dir, mode=mode, transform=transforms_dict['train'])
    
    # Get class names
    class_names = full_dataset.class_names
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Apply validation transforms to validation dataset
    val_dataset.dataset.transform = transforms_dict['val']
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"\nDataLoader Summary:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of classes: {len(class_names)}")
    
    return train_loader, val_loader, class_names


if __name__ == '__main__':
    # Example usage
    dataset_path = Path(__file__).parent / 'Real vs Fake(AI) Image Dataset'
    
    # Test binary dataset
    print("=" * 60)
    print("Testing Binary Dataset")
    print("=" * 60)
    train_loader, val_loader, classes = create_train_val_loaders(
        str(dataset_path),
        mode='binary',
        batch_size=32,
        val_split=0.2
    )
    print(f"Classes: {classes}\n")
    
    # Test generator dataset
    print("=" * 60)
    print("Testing Generator Dataset")
    print("=" * 60)
    train_loader, val_loader, classes = create_train_val_loaders(
        str(dataset_path),
        mode='generator',
        batch_size=32,
        val_split=0.2
    )
    print(f"Number of generators: {len(classes)}")
    print(f"Generators: {classes}\n")
    
    # Show sample batch
    print("Sample batch shape:", next(iter(train_loader))[0].shape)
