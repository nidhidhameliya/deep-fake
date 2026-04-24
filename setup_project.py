"""
Setup and Migration Script for Deepfake Detection Project
Helps reorganize dataset and clean up old files
"""

import os
import shutil
import json
from pathlib import Path
from utils import (
    delete_old_models, 
    check_dataset_structure, 
    get_image_files,
    count_images_in_dataset
)


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step_num, title):
    """Print formatted step"""
    print(f"\n[Step {step_num}] {title}")
    print("-" * 70)


def migrate_dataset_structure(source_dataset_path='Real vs Fake(AI) Image Dataset'):
    """
    Migrate dataset from old structure to new structure
    (real_images/fake_images -> Real/Fake)
    
    Args:
        source_dataset_path (str): Path to dataset root
    """
    print_step(1, "DATASET MIGRATION: Old Structure → New Structure")
    
    dataset_path = Path(source_dataset_path)
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found at: {dataset_path}")
        return False
    
    # Check current structure
    structure = check_dataset_structure(dataset_path)
    
    if structure['valid'] and structure['real_folder'] == 'Real':
        print("✓ Dataset is already in new structure (Real/ and Fake/)")
        return True
    
    # Create new structure
    real_new = dataset_path / 'Real'
    fake_new = dataset_path / 'Fake'
    
    # Handle real images migration
    real_old = dataset_path / 'real_images'
    if real_old.exists() and not real_new.exists():
        print(f"\nMigrating real images...")
        try:
            shutil.copytree(real_old, real_new)
            print(f"✓ Created: {real_new}")
            print(f"  Copied {len(get_image_files(str(real_new)))} images")
        except Exception as e:
            print(f"✗ Error migrating real images: {e}")
            return False
    elif not real_old.exists():
        print(f"⚠ real_images/ folder not found, skipping...")
    elif real_new.exists():
        print(f"✓ Real/ folder already exists")
    
    # Handle fake images migration
    fake_old = dataset_path / 'fake_images'
    if fake_old.exists() and not fake_new.exists():
        print(f"\nMigrating fake images (generators)...")
        try:
            shutil.copytree(fake_old, fake_new)
            print(f"✓ Created: {fake_new}")
            
            # Count generators
            gen_count = len([
                f for f in fake_new.iterdir() 
                if f.is_dir()
            ])
            print(f"  Migrated {gen_count} generator folders")
            
            total_fakes = len(get_image_files(str(fake_new)))
            print(f"  Total fake images: {total_fakes}")
        except Exception as e:
            print(f"✗ Error migrating fake images: {e}")
            return False
    elif not fake_old.exists():
        print(f"⚠ fake_images/ folder not found, skipping...")
    elif fake_new.exists():
        print(f"✓ Fake/ folder already exists")
    
    # Verify new structure
    print(f"\nVerifying new structure...")
    final_structure = check_dataset_structure(dataset_path)
    
    if final_structure['valid'] and final_structure['real_folder'] == 'Real':
        print("✓ Dataset structure verification successful!")
        return True
    else:
        print("✗ Dataset structure verification failed")
        for msg in final_structure['messages']:
            print(f"  {msg}")
        return False


def cleanup_old_models():
    """Delete old model files"""
    print_step(2, "CLEANUP: Remove Old Model Files")
    
    model_files = ['binary_model.pth', 'generator_model.pth']
    
    print(f"Checking for old model files to delete:")
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"  - {model_file} (found)")
        else:
            print(f"  - {model_file} (not found)")
    
    print("\nDeleting old models...")
    result = delete_old_models(model_files)
    
    print(f"\nCleanup Summary:")
    print(f"  Deleted: {result['total']} file(s)")
    if result['deleted']:
        for f in result['deleted']:
            print(f"    ✓ {f}")
    if result['not_found']:
        for f in result['not_found']:
            print(f"    - {f} (not found)")
    
    return True


def dataset_info(dataset_path='Real vs Fake(AI) Image Dataset'):
    """Display dataset information"""
    print_step(3, "DATASET INFORMATION")
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"✗ Dataset not found at: {dataset_path}")
        return
    
    # Check structure
    structure = check_dataset_structure(dataset_path)
    print("\nDataset Structure:")
    for msg in structure['messages']:
        print(f"  {msg}")
    
    if not structure['valid']:
        print("\n✗ Cannot provide information: Dataset structure is invalid")
        return
    
    # Count images
    counts = count_images_in_dataset(str(dataset_path))
    
    print(f"\nImage Statistics:")
    print(f"  Real images:   {counts['real']:>6}")
    print(f"  Fake images:   {counts['fake_total']:>6}")
    print(f"  Total images:  {counts['real'] + counts['fake_total']:>6}")
    
    if structure['generators']:
        print(f"\nGenerators ({len(structure['generators'])} total):")
        
        # Get counts per generator
        gen_counts = counts.get('generators', {})
        sorted_gens = sorted(
            gen_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for gen_name, count in sorted_gens:
            percentage = 100 * count / counts['fake_total'] if counts['fake_total'] > 0 else 0
            print(f"    {gen_name:30} {count:>5} ({percentage:>5.1f}%)")


def save_project_status(status_file='project_status.json'):
    """Save project status to file"""
    status = {
        'timestamp': str(Path.cwd()),
        'dataset_structure': check_dataset_structure('Real vs Fake(AI) Image Dataset'),
        'models': {
            'binary_model': os.path.exists('binary_model.pth'),
            'generator_model': os.path.exists('generator_model.pth')
        },
        'configs': {
            'generator_classes': os.path.exists('generator_classes.json'),
            'training_history_binary': os.path.exists('training_history_binary.json'),
            'training_history_generator': os.path.exists('training_history_generator.json')
        }
    }
    
    # Don't include the message list as it may not be JSON serializable
    if 'messages' in status['dataset_structure']:
        del status['dataset_structure']['messages']
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=4)
    
    print(f"✓ Project status saved to: {status_file}")


def main():
    """Main setup function"""
    
    print_header("DEEPFAKE DETECTION PROJECT - SETUP & MIGRATION")
    
    print("\nThis script will help you:")
    print("  1. Migrate dataset from old structure to new structure")
    print("  2. Delete old model files")
    print("  3. Display dataset information")
    
    # Step 1: Migrate dataset
    print("\n")
    migrate_dataset_structure()
    
    # Step 2: Cleanup old models
    print("\n")
    cleanup_old_models()
    
    # Step 3: Show dataset info
    print("\n")
    dataset_info()
    
    # Save status
    print("\n")
    print_step(4, "SAVE PROJECT STATUS")
    save_project_status()
    
    # Final summary
    print_header("SETUP COMPLETE")
    print("\nNext Steps:")
    print("  1. Train binary model:     python train_binary.py")
    print("  2. Train generator model:  python train_generator.py")
    print("  3. Run inference:          python app.py")
    print("\nYour project is ready for training!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
