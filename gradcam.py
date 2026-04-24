"""
Grad-CAM implementation for model explainability
Generates and visualizes attention heatmaps for neural networks
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path


class GradCAM:
    """Grad-CAM: Gradient-based Class Activation Map"""
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM
        
        Args:
            model (nn.Module): PyTorch model
            target_layer (str): Name of the layer to compute gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        
        def forward_hook(module, input, output):
            # Store activations
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            # Store gradients
            self.gradients = grad_output[0].detach()
        
        # Find and hook the target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                print(f"Hooks registered on layer: {name}")
                return
        
        raise ValueError(f"Layer {self.target_layer} not found in model")
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor (Tensor): Input image [1, 3, H, W]
            target_class (int): Target class index (uses max if None)
        
        Returns:
            np.ndarray: Heatmap [H, W]
        """
        self.model.eval()
        
        # Forward pass
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            
            # Select target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Backward pass
            target_score = output[0, target_class]
            self.model.zero_grad()
            target_score.backward()
        
        # Compute Grad-CAM
        # Gradients shape: [1, C, H, W]
        # Activations shape: [1, C, H, W]
        
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU to get positive activations
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class
    
    def visualize(self, image_np, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image
        
        Args:
            image_np (np.ndarray): Original image [H, W, 3] in RGB, values [0, 255]
            heatmap (np.ndarray): Heatmap [H, W], values [0, 1]
            alpha (float): Blending coefficient
        
        Returns:
            np.ndarray: Overlaid image [H, W, 3]
        """
        # Handle RGBA images - convert to RGB
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Ensure image is RGB with 3 channels
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match image size if needed
        if heatmap.shape != image_np.shape[:2]:
            heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        
        # Ensure image_np is correct type
        image_np = image_np.astype(np.float32)
        
        # Convert heatmap to color (jet colormap)
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        # OpenCV uses BGR, convert to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored = heatmap_colored.astype(np.float32)
        
        # Blend
        overlaid = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlaid.astype(np.uint8)


def get_target_layer_name(model_name, model_type):
    """
    Get the appropriate target layer for Grad-CAM
    
    Args:
        model_name (str): 'binary' or 'generator'
        model_type (str): 'resnet50' or 'efficientnet_b0'
    
    Returns:
        str: Target layer name
    """
    if model_type == 'resnet50':
        # Use the last residual block
        return 'backbone.layer4.2.conv3'
    elif model_type == 'efficientnet_b0':
        # Use the last MBConv block
        return 'backbone.features.8'
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def pil_to_np(pil_image):
    """Convert PIL image to numpy array [H, W, 3] with values [0, 255]"""
    return np.array(pil_image)


def np_to_pil(image_np):
    """Convert numpy array [H, W, 3] with values [0, 255] to PIL image"""
    from PIL import Image
    return Image.fromarray(image_np.astype(np.uint8))


def generate_gradcam(model, image_tensor, pil_image, target_class=None, model_type='resnet50', alpha=0.5):
    """
    Convenient wrapper to generate Grad-CAM visualization
    
    Args:
        model: PyTorch model
        image_tensor (Tensor): Preprocessed image [1, 3, 224, 224]
        pil_image: Original PIL image
        target_class (int): Target class for Grad-CAM
        model_type (str): Type of model backbone
        alpha (float): Blending coefficient
    
    Returns:
        tuple: (heatmap_np, overlaid_image_pil, target_class)
    """
    # Get target layer
    target_layer = get_target_layer_name('binary', model_type)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap, class_idx = gradcam.generate(image_tensor, target_class)
    
    # Convert image to numpy
    image_np = pil_to_np(pil_image)
    
    # Visualize
    overlaid = gradcam.visualize(image_np, heatmap, alpha=alpha)
    overlaid_pil = np_to_pil(overlaid)
    
    return heatmap, overlaid_pil, class_idx


if __name__ == '__main__':
    """Test Grad-CAM"""
    from model import create_model
    from dataset import get_data_transforms
    from PIL import Image
    
    print("Grad-CAM test would run here with actual model and image")
