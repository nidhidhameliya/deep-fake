"""
Training script for Generator Classifier (Which AI model generated the fake)
Trains a multi-class model to identify which generator created the fake image
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
import time

from dataset import create_train_val_loaders
from model import create_model, print_model_summary


class GeneratorTrainer:
    """Trainer class for generator classification model"""
    
    def __init__(self, model, device, num_classes, learning_rate=1e-4, model_save_path='generator_model.pth'):
        """
        Initialize trainer
        
        Args:
            model (nn.Module): PyTorch model
            device (torch.device): Device (cpu or cuda)
            num_classes (int): Number of generator classes
            learning_rate (float): Learning rate
            model_save_path (str): Path to save best model
        """
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.model_save_path = model_save_path
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=20):
        """
        Train the model
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of epochs to train
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Number of generator classes: {self.num_classes}")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f"\nEpoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch + 1
                self.save_model()
                print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")
            
            print("=" * 70)
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f} (Epoch {self.best_epoch})")
    
    def save_model(self):
        """Save the best model"""
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"  Model saved to {self.model_save_path}")
    
    def save_history(self, save_path='training_history_generator.json'):
        """Save training history to JSON file"""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {save_path}")


def main():
    """Main training function"""
    
    # Configuration
    CONFIG = {
        'dataset_path': 'Real vs Fake(AI) Image Dataset',
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'model_type': 'resnet50',  # 'resnet50' or 'efficientnet_b0'
        'model_save_path': 'generator_model.pth',
        'history_save_path': 'training_history_generator.json'
    }
    
    # Check if dataset exists
    if not os.path.exists(CONFIG['dataset_path']):
        print(f"Error: Dataset not found at {CONFIG['dataset_path']}")
        print("Please make sure the dataset folder is in the correct location")
        print(f"Expected location: {os.path.abspath(CONFIG['dataset_path'])}")
        return
    
    # Check for required folder structure
    dataset_path = Path(CONFIG['dataset_path'])
    fake_path = dataset_path / 'Fake'
    
    # Check for new structure, fallback to old structure
    if not (fake_path.exists() or (dataset_path / 'fake_images').exists()):
        print(f"Error: No 'Fake/' or 'fake_images/' folder found in {dataset_path}")
        print("Please reorganize your dataset to include:")
        print("  - Real vs Fake(AI) Image Dataset/Fake/  (contains generator subfolders)")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, class_names = create_train_val_loaders(
        root_dir=CONFIG['dataset_path'],
        mode='generator',
        batch_size=CONFIG['batch_size'],
        val_split=0.2,
        num_workers=0
    )
    
    print(f"Number of generator classes: {len(class_names)}")
    print(f"Generators: {', '.join(class_names)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        'generator',
        num_classes=len(class_names),
        model_type=CONFIG['model_type'],
        pretrained=True
    )
    print_model_summary(model, num_classes=len(class_names))
    
    # Create trainer
    trainer = GeneratorTrainer(
        model=model,
        device=device,
        num_classes=len(class_names),
        learning_rate=CONFIG['learning_rate'],
        model_save_path=CONFIG['model_save_path']
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs']
    )
    
    # Save history
    trainer.save_history(CONFIG['history_save_path'])
    
    # Save class names for inference
    with open('generator_classes.json', 'w') as f:
        json.dump(class_names, f, indent=4)
    print("Generator class names saved to generator_classes.json")
    
    print("\n" + "=" * 70)
    print("Training finished successfully!")
    print(f"Model saved to: {CONFIG['model_save_path']}")
    print(f"History saved to: {CONFIG['history_save_path']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
