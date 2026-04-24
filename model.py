"""
Model architectures for Deepfake Detection
Uses transfer learning with EfficientNet or ResNet50
"""

import torch
import torch.nn as nn
from torchvision import models


class BinaryClassifier(nn.Module):
    """
    Binary classifier for Real vs Fake detection
    Uses pretrained ResNet50 backbone with custom classification head
    """
    
    def __init__(self, num_classes=2, model_type='resnet50', pretrained=True):
        """
        Initialize Binary Classifier
        
        Args:
            num_classes (int): Number of classes (should be 2 for binary)
            model_type (str): 'resnet50' or 'efficientnet_b0'
            pretrained (bool): Whether to use pretrained weights
        """
        super(BinaryClassifier, self).__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        
        if model_type == 'resnet50':
            # Load pretrained ResNet50
            self.backbone = models.resnet50(pretrained=pretrained)
            # Get number of input features for the final layer
            num_features = self.backbone.fc.in_features
            # Remove the original classification layer
            self.backbone.fc = nn.Identity()
            
        elif model_type == 'efficientnet_b0':
            # Load pretrained EfficientNet-B0
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Get number of input features
            num_features = self.backbone.classifier[1].in_features
            # Remove the original classification layer
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): Input image tensor [B, 3, 224, 224]
        
        Returns:
            Tensor: Logits [B, num_classes]
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    def get_features(self, x):
        """
        Get feature map from backbone (useful for Grad-CAM)
        
        Args:
            x (Tensor): Input image tensor
        
        Returns:
            Tensor: Feature map from backbone
        """
        return self.backbone(x)


class GeneratorClassifier(nn.Module):
    """
    Multi-class classifier for identifying which generator created the fake image
    Uses pretrained ResNet50 backbone with custom classification head
    """
    
    def __init__(self, num_classes, model_type='resnet50', pretrained=True):
        """
        Initialize Generator Classifier
        
        Args:
            num_classes (int): Number of generator classes
            model_type (str): 'resnet50' or 'efficientnet_b0'
            pretrained (bool): Whether to use pretrained weights
        """
        super(GeneratorClassifier, self).__init__()
        
        self.model_type = model_type
        self.num_classes = num_classes
        
        if model_type == 'resnet50':
            # Load pretrained ResNet50
            self.backbone = models.resnet50(pretrained=pretrained)
            # Get number of input features for the final layer
            num_features = self.backbone.fc.in_features
            # Remove the original classification layer
            self.backbone.fc = nn.Identity()
            
        elif model_type == 'efficientnet_b0':
            # Load pretrained EfficientNet-B0
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Get number of input features
            num_features = self.backbone.classifier[1].in_features
            # Remove the original classification layer
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Custom classification head (slightly larger for multi-class)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): Input image tensor [B, 3, 224, 224]
        
        Returns:
            Tensor: Logits [B, num_classes]
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits
    
    def get_features(self, x):
        """
        Get feature map from backbone (useful for Grad-CAM)
        
        Args:
            x (Tensor): Input image tensor
        
        Returns:
            Tensor: Feature map from backbone
        """
        return self.backbone(x)


def create_model(model_name, num_classes, model_type='resnet50', pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_name (str): 'binary' or 'generator'
        num_classes (int): Number of classes
        model_type (str): 'resnet50' or 'efficientnet_b0'
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        nn.Module: Model instance
    """
    if model_name == 'binary':
        return BinaryClassifier(num_classes=num_classes, model_type=model_type, pretrained=pretrained)
    elif model_name == 'generator':
        return GeneratorClassifier(num_classes=num_classes, model_type=model_type, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model):
    """
    Count total number of trainable parameters
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, num_classes):
    """
    Print model summary
    
    Args:
        model (nn.Module): PyTorch model
        num_classes (int): Number of classes
    """
    total_params = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of classes: {num_classes}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model structure:\n{model}")


if __name__ == '__main__':
    # Test model creation
    print("=" * 60)
    print("Testing Binary Classifier")
    print("=" * 60)
    binary_model = create_model('binary', num_classes=2, model_type='resnet50', pretrained=True)
    print_model_summary(binary_model, num_classes=2)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 224, 224)
    output = binary_model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}\n")
    
    print("=" * 60)
    print("Testing Generator Classifier")
    print("=" * 60)
    num_generators = 25  # Example: 25 different generators
    generator_model = create_model('generator', num_classes=num_generators, model_type='resnet50', pretrained=True)
    print_model_summary(generator_model, num_classes=num_generators)
    
    # Test with dummy input
    output = generator_model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
