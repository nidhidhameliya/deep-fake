# PROJECT UPDATE TASK: Retrain models after dataset change

# I have a Deep Fake / AI Image Detection project using PyTorch.

# Updated Dataset Structure:
# Real_vs_Fake(AI)_Image_Dataset/
# ├── Real/
# └── Fake/
#     ├── Nano Banana 2/
#     └── OpenAI DALL-E/

# TASK REQUIREMENTS:

# 1. DELETE OLD MODELS
# Remove:
# - binary_model.pth
# - generator_model.pth

# 2. DATASET LOADING
# - Update dataset loader to:
#   - Read images from Real/ and Fake/ subfolders
#   - Automatically include subfolders inside Fake/
#   - Use torchvision.datasets.ImageFolder if possible
#   - Apply transforms (resize, normalize)

# 3. LABELING
# Binary Model:
#   Real = 0
#   Fake = 1
#
# Generator Model (multi-class):
#   Nano Banana 2 = 0
#   OpenAI DALL-E = 1

# 4. TRAINING UPDATE
# - Update training loop to:
#   - Load updated dataset
#   - Use GPU if available
#   - Use proper loss:
#       Binary: BCEWithLogitsLoss or CrossEntropyLoss
#       Generator: CrossEntropyLoss
#   - Add validation split (optional but preferred)
#   - Print accuracy and loss

# 5. SAVE MODELS
# Save new models as:
# - binary_model.pth
# - generator_model.pth

# 6. INFERENCE UPDATE
# - Update prediction code to:
#   - Load new models
#   - Apply same transforms
#   - Return correct class labels

# 7. CODE QUALITY
# - Keep code modular (dataset, train, test functions)
# - Avoid hardcoding paths
# - Handle errors properly

# Generate complete updated code for:
# - dataset loader
# - training scripts (both models)
# - inference script

# Do not skip any required change.