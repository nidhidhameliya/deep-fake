# PROJECT UPDATE GUIDE - Retrain with New Dataset Structure

## Overview

This guide explains the changes made to support the updated dataset structure and retraining process for the Deepfake Detection project.

---

## 📋 What Changed?

### 1. **Dataset Structure**
The project now supports a simplified dataset organization:

**New Structure (Recommended):**
```
Real vs Fake(AI) Image Dataset/
├── Real/                    (Real images)
└── Fake/                    (Fake images organized by generator)
    ├── Nano Banana 2/
    ├── OpenAI DALL-E/
    └── [other generators]/
```

**Old Structure (Still Supported):**
```
Real vs Fake(AI) Image Dataset/
├── real_images/
└── fake_images/
    ├── big_gan/
    ├── stylegan/
    └── [other generators]/
```

### 2. **Model Files**
- **Old files to delete:**
  - `binary_model.pth` (will be retrained)
  - `generator_model.pth` (will be retrained)

### 3. **New Files & Updates**
- ✅ `dataset.py` - Updated to handle both old and new structures with fallback support
- ✅ `train_binary.py` - Enhanced with better error handling
- ✅ `train_generator.py` - Saves `generator_classes.json` for inference
- ✅ `inference.py` - Updated to load models and class names properly
- ✅ `utils.py` - New helper functions for migration
- ✅ `setup_project.py` - NEW: Automated setup and migration script

---

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

Run the automated setup script:

```bash
python setup_project.py
```

This script will:
1. ✅ Migrate dataset from old structure to new structure (if needed)
2. ✅ Delete old model files (`binary_model.pth`, `generator_model.pth`)
3. ✅ Display dataset statistics
4. ✅ Save project status

### Option 2: Manual Setup

#### Step 1: Delete Old Models
```bash
# Delete old models manually
del binary_model.pth
del generator_model.pth
```

#### Step 2: Reorganize Dataset (if using new structure)
```
Rename/move folders:
- real_images/ → Real/
- fake_images/ → Fake/
```

#### Step 3: Verify Structure
```python
from utils import check_dataset_structure
structure = check_dataset_structure('Real vs Fake(AI) Image Dataset')
for msg in structure['messages']:
    print(msg)
```

---

## 📚 Training Models

### Training Binary Classifier (Real vs Fake)

```bash
python train_binary.py
```

**Expected Output:**
- Creates: `binary_model.pth`
- Creates: `training_history_binary.json`
- Training time: ~30-60 minutes (depends on dataset size)
- GPU: Recommended for faster training

### Training Generator Classifier (Which AI model generated the fake)

```bash
python train_generator.py
```

**Expected Output:**
- Creates: `generator_model.pth`
- Creates: `generator_classes.json` (automatically saved)
- Creates: `training_history_generator.json`
- Training time: ~60-120 minutes (depends on number of generators)
- GPU: Recommended for faster training

---

## 📊 Dataset Structure Details

### Binary Classification Labeling
| Folder | Label | Description |
|--------|-------|-------------|
| `Real/` | 0 | Real photographs |
| `Fake/` | 1 | AI-generated images (all generators) |

### Generator Classification Labeling
Automatically numbered based on alphabetical order of folders in `Fake/`:

```
Fake/
├── Nano Banana 2/        → Label 0
└── OpenAI DALL-E/        → Label 1
(Numbering continues for additional generators)
```

The mapping is saved in `generator_classes.json`:
```json
["Nano Banana 2", "OpenAI DALL-E", ...]
```

---

## 🔧 Configuration

### Key Parameters in Training Scripts

#### train_binary.py
```python
CONFIG = {
    'dataset_path': 'Real vs Fake(AI) Image Dataset',
    'batch_size': 32,              # Images per batch
    'num_epochs': 20,              # Training epochs
    'learning_rate': 1e-4,         # Optimizer learning rate
    'model_type': 'resnet50',      # Model architecture
    'model_save_path': 'binary_model.pth',
    'history_save_path': 'training_history_binary.json'
}
```

#### train_generator.py
```python
CONFIG = {
    'dataset_path': 'Real vs Fake(AI) Image Dataset',
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'model_type': 'resnet50',      # ResNet50 or EfficientNet-B0
    'model_save_path': 'generator_model.pth',
    'history_save_path': 'training_history_generator.json'
}
```

### Model Architectures Available
- **ResNet50** (Default) - Better accuracy, larger
- **EfficientNet-B0** - Faster, smaller footprint

---

## 📈 Monitoring Training

### View Training Progress
Training scripts print real-time progress:

```
Epoch [1/20] (45.2s)
  Train Loss: 0.4532 | Train Acc: 0.8234
  Val Loss:   0.3892 | Val Acc:   0.8756
  ✓ Best model saved (Val Acc: 0.8756)
```

### Load Training History
```python
import json

# Binary training history
with open('training_history_binary.json', 'r') as f:
    history_binary = json.load(f)

# Generator training history
with open('training_history_generator.json', 'r') as f:
    history_generator = json.load(f)

# Plot accuracy
import matplotlib.pyplot as plt
plt.plot(history_binary['train_accuracy'], label='Train')
plt.plot(history_binary['val_accuracy'], label='Validation')
plt.legend()
plt.show()
```

---

## 🔍 Running Inference

### Using Updated Inference Pipeline

```python
from inference import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector(
    binary_model_path='binary_model.pth',
    generator_model_path='generator_model.pth'
)

# Detect image
image_path = 'test_image.jpg'
results = detector.detect(image_path)

# Display results
print(detector.format_results(results))
```

### Expected Output
```
Prediction: Fake
Confidence: 0.9823

Generator: OpenAI DALL-E
Generator Confidence: 0.8765
```

---

## 📁 Dataset Organization Guide

### Creating Real/ Folder
```bash
# Option 1: Move existing real_images
mv "Real vs Fake(AI) Image Dataset/real_images" "Real vs Fake(AI) Image Dataset/Real"

# Option 2: Using Python
import shutil
shutil.move('Real vs Fake(AI) Image Dataset/real_images', 
            'Real vs Fake(AI) Image Dataset/Real')
```

### Creating Fake/ Folder
```bash
# Option 1: Move existing fake_images
mv "Real vs Fake(AI) Image Dataset/fake_images" "Real vs Fake(AI) Image Dataset/Fake"

# Option 2: Using Python
import shutil
shutil.move('Real vs Fake(AI) Image Dataset/fake_images',
            'Real vs Fake(AI) Image Dataset/Fake')
```

### Verifying Structure
```python
from pathlib import Path
from utils import check_dataset_structure

dataset_path = 'Real vs Fake(AI) Image Dataset'
structure = check_dataset_structure(dataset_path)

print("Structure valid:", structure['valid'])
print("Real folder:", structure['real_folder'])
print("Fake folder:", structure['fake_folder'])
print("Generators:", structure['generators'])
```

---

## ⚠️ Troubleshooting

### Issue: "No images found in dataset"
**Solution:**
- Check folder names are exactly "Real" and "Fake" (case-sensitive on Linux/Mac)
- Verify image file extensions are: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Run `python setup_project.py` to diagnose

### Issue: "generator_classes.json not found"
**Solution:**
- Train the generator model first: `python train_generator.py`
- This automatically saves the generator classes

### Issue: CUDA out of memory
**Solution:**
- Reduce batch size in CONFIG: `'batch_size': 16` (instead of 32)
- Use CPU instead: Models will auto-detect or use torch.device('cpu')

### Issue: Models not found during inference
**Solution:**
- Train both models first:
  ```bash
  python train_binary.py
  python train_generator.py
  ```
- Verify files exist: `binary_model.pth`, `generator_model.pth`, `generator_classes.json`

---

## 📝 Key Code Changes

### dataset.py - New Structure Support
```python
# Now checks for new structure with fallback
real_path = self.root_dir / 'Real'      # New
real_path_old = self.root_dir / 'real_images'  # Fallback

fake_path = self.root_dir / 'Fake'      # New
fake_path_old = self.root_dir / 'fake_images'  # Fallback
```

### train_generator.py - Auto-save Classes
```python
# Automatically saves generator class names
with open('generator_classes.json', 'w') as f:
    json.dump(class_names, f, indent=4)
```

### inference.py - Better Error Handling
```python
# Improved model loading with error handling and fallback
self.load_generator_model(generator_model_path)
```

### utils.py - New Helper Functions
```python
delete_old_models()                  # Clean up old models
check_dataset_structure()            # Verify dataset
update_count_images_in_dataset()    # Count images
```

---

## 🎯 Next Steps

1. **Run setup script:**
   ```bash
   python setup_project.py
   ```

2. **Train models:**
   ```bash
   python train_binary.py
   python train_generator.py
   ```

3. **Run inference (Streamlit app):**
   ```bash
   streamlit run app.py
   ```

4. **View results:** Open http://localhost:8501 in your browser

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review training logs in `training_history_*.json`
3. Verify dataset structure with `setup_project.py`
4. Check GPU availability with `torch.cuda.is_available()`

---

## ✅ Verification Checklist

Before starting training, verify:
- [ ] Old model files deleted
- [ ] Dataset structure is correct (Real/ and Fake/)
- [ ] Images have correct extensions (.jpg, .png, etc.)
- [ ] GPU is available (if using): `torch.cuda.is_available()`
- [ ] Python requirements installed: `pip install -r requirements.txt`
- [ ] Both training scripts are executable

---

**Last Updated:** 2026-04-22
**Version:** 2.0 (Updated for new dataset structure)
