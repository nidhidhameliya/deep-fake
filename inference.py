"""
Inference pipeline for Deepfake Detection
Handles image prediction using both binary and generator models
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
from pathlib import Path

from model import create_model
from dataset import get_data_transforms


class DeepfakeDetector:
    """End-to-end inference pipeline for deepfake detection"""
    
    def __init__(self, binary_model_path='binary_model.pth', 
                 generator_model_path='generator_model.pth',
                 device=None):
        """
        Initialize detector with pretrained models
        
        Args:
            binary_model_path (str): Path to binary model weights
            generator_model_path (str): Path to generator model weights
            device (torch.device): Device to use (defaults to cuda if available)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load binary model
        self.binary_model = create_model('binary', num_classes=2, model_type='resnet50', pretrained=False)
        self.binary_model.load_state_dict(torch.load(binary_model_path, map_location=self.device))
        self.binary_model.to(self.device)
        self.binary_model.eval()
        self.class_names_binary = ['Real', 'Fake']
        
        # Load generator model and class names
        self.generator_model = None
        self.generator_classes = []
        self.load_generator_model(generator_model_path)
        
        # Get transforms
        self.transforms = get_data_transforms()['val']
        
        print(f"Detector initialized on device: {self.device}")
    
    def load_generator_model(self, model_path):
        """
        Load generator model and class names
        
        Args:
            model_path (str): Path to generator model weights
        """
        # Try to load generator class names from JSON
        class_names_path = 'generator_classes.json'
        if Path(class_names_path).exists():
            try:
                with open(class_names_path, 'r') as f:
                    self.generator_classes = json.load(f)
                print(f"Loaded {len(self.generator_classes)} generator classes from {class_names_path}")
            except Exception as e:
                print(f"Error loading {class_names_path}: {e}")
                self.generator_classes = []
        else:
            print(f"Warning: {class_names_path} not found")
            self.generator_classes = []
        
        # If no classes loaded from file, we need to load the model first to get the number of classes
        if not self.generator_classes:
            print("Attempting to infer number of classes from model checkpoint...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                # Look for the last layer's weight shape to determine number of classes
                for key in checkpoint.keys():
                    if 'head' in key and 'weight' in key and 'linear' in key:
                        num_classes = checkpoint[key].shape[0]
                        print(f"Inferred {num_classes} classes from model checkpoint")
                        self.generator_classes = [f'generator_{i}' for i in range(num_classes)]
                        break
            except Exception as e:
                print(f"Error inferring number of classes: {e}")
                self.generator_classes = []
        
        if not self.generator_classes:
            raise ValueError("Could not determine number of generator classes. Please provide generator_classes.json")
        
        # Load generator model
        self.generator_model = create_model('generator', num_classes=len(self.generator_classes), 
                                           model_type='resnet50', pretrained=False)
        self.generator_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.generator_model.to(self.device)
        self.generator_model.eval()
        
        print(f"Generator model loaded successfully with {len(self.generator_classes)} classes")
    
    def preprocess_image(self, image_path_or_pil):
        """
        Preprocess image for model input
        
        Args:
            image_path_or_pil: Either file path (str) or PIL Image
        
        Returns:
            Tensor: Preprocessed image [1, 3, 224, 224]
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transforms(image)
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device), image
    
    def predict_binary(self, image_tensor):
        """
        Predict using binary model (Real vs Fake)
        
        Args:
            image_tensor (Tensor): Preprocessed image [1, 3, 224, 224]
        
        Returns:
            dict: Prediction results with label, confidence, and logits
        """
        with torch.no_grad():
            outputs = self.binary_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        result = {
            'class_idx': predicted_class.item(),
            'class_name': self.class_names_binary[predicted_class.item()],
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0],
            'logits': outputs.cpu().numpy()[0]
        }
        
        return result
    
    def predict_generator(self, image_tensor):
        """
        Predict using generator model (which AI model generated the fake)
        
        Args:
            image_tensor (Tensor): Preprocessed image [1, 3, 224, 224]
        
        Returns:
            dict: Prediction results with generator name and confidence
        """
        with torch.no_grad():
            outputs = self.generator_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        result = {
            'class_idx': predicted_class.item(),
            'class_name': self.generator_classes[predicted_class.item()],
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0],
            'logits': outputs.cpu().numpy()[0]
        }
        
        return result
    
    def detect(self, image_path_or_pil):
        """
        Full deepfake detection pipeline
        
        Args:
            image_path_or_pil: Either file path (str) or PIL Image
        
        Returns:
            dict: Complete detection results
        """
        # Preprocess
        image_tensor, pil_image = self.preprocess_image(image_path_or_pil)
        
        # Binary prediction
        binary_result = self.predict_binary(image_tensor)
        
        # Generator prediction (only if fake)
        generator_result = None
        if binary_result['class_name'] == 'Fake':
            generator_result = self.predict_generator(image_tensor)
        
        # Compile results
        results = {
            'binary': binary_result,
            'generator': generator_result,
            'pil_image': pil_image,
            'image_tensor': image_tensor
        }
        
        return results
    
    def format_results(self, results):
        """
        Format results for display
        
        Args:
            results (dict): Detection results
        
        Returns:
            str: Formatted string for display
        """
        binary_result = results['binary']
        generator_result = results['generator']
        
        output = f"Prediction: {binary_result['class_name']}\n"
        output += f"Confidence: {binary_result['confidence']:.4f}\n"
        
        if binary_result['class_name'] == 'Fake' and generator_result:
            output += f"\nGenerator: {generator_result['class_name'].replace('_', ' ').title()}\n"
            output += f"Generator Confidence: {generator_result['confidence']:.4f}\n"
        
        return output


def main():
    """Test inference pipeline"""
    
    # Initialize detector
    detector = DeepfakeDetector(
        binary_model_path='binary_model.pth',
        generator_model_path='generator_model.pth'
    )
    
    # Example usage (if you have a test image)
    test_images = list(Path('Real vs Fake(AI) Image Dataset/real_images').glob('*.jpg'))[:1]
    test_images += list(Path('Real vs Fake(AI) Image Dataset/fake_images/stable_diffusion').glob('*.jpg'))[:1]
    
    for img_path in test_images:
        if img_path.exists():
            print(f"\nTesting on: {img_path.name}")
            results = detector.detect(str(img_path))
            print(detector.format_results(results))


if __name__ == '__main__':
    main()
