"""
Verification script to check if everything is set up correctly.
Run this before starting training to ensure all dependencies and data are ready.
"""
import sys
import os
from pathlib import Path
import json


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60)


def check_python_version():
    """Check Python version"""
    print("\n🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro}")
        print("   ⚠️  Python 3.8+ required")
        return False


def check_pytorch():
    """Check PyTorch installation"""
    print("\n🔥 Checking PyTorch...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"   ✓ CUDA {torch.version.cuda}")
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️  CUDA not available (CPU only)")
            print("   ⚠️  Training will be very slow without GPU")
        
        return True
    except ImportError:
        print("   ✗ PyTorch not installed")
        print("   Install with: pip install torch torchvision")
        return False


def check_dependencies():
    """Check required dependencies"""
    print("\n📦 Checking dependencies...")
    
    required = {
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'albumentations': 'albumentations',
        'pycocotools': 'pycocotools',
        'transformers': 'transformers',
        'peft': 'peft',
        'optuna': 'optuna',
        'tqdm': 'tqdm',
        'segment_anything': 'segment-anything'
    }
    
    all_installed = True
    
    for module_name, package_name in required.items():
        try:
            if module_name == 'segment_anything':
                __import__(module_name)
            else:
                __import__(module_name)
            print(f"   ✓ {package_name}")
        except ImportError:
            print(f"   ✗ {package_name} not installed")
            all_installed = False
    
    if not all_installed:
        print("\n   Install missing packages with:")
        print("   pip install -r requirements.txt")
    
    return all_installed


def check_sam_checkpoint():
    """Check if SAM checkpoint exists"""
    print("\n🎯 Checking SAM checkpoint...")
    
    checkpoint_paths = [
        "checkpoints/sam_vit_b_01ec64.pth",
        "../checkpoints/sam_vit_b_01ec64.pth",
        "./sam_vit_b_01ec64.pth"
    ]
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   ✓ Found checkpoint: {path}")
            print(f"   ✓ Size: {size_mb:.1f} MB")
            return True
    
    print("   ✗ SAM checkpoint not found")
    print("\n   Download with:")
    print("   mkdir -p checkpoints")
    print("   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O checkpoints/sam_vit_b_01ec64.pth")
    print("\n   Or in PowerShell:")
    print('   Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "checkpoints/sam_vit_b_01ec64.pth"')
    
    return False


def check_data():
    """Check if COCO data exists"""
    print("\n📊 Checking COCO dataset...")
    
    data_root = "../Cataract COCO Segmentation/Cataract COCO Segmentation"
    
    if not os.path.exists(data_root):
        print(f"   ✗ Data directory not found: {data_root}")
        return False
    
    splits = ['train', 'valid', 'test']
    all_ok = True
    
    for split in splits:
        ann_file = os.path.join(data_root, split, '_annotations.coco.json')
        img_dir = os.path.join(data_root, split)
        
        if not os.path.exists(ann_file):
            print(f"   ✗ {split}: annotation file not found")
            all_ok = False
            continue
        
        # Check annotation file
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            n_images = len(data.get('images', []))
            n_annotations = len(data.get('annotations', []))
            n_categories = len(data.get('categories', []))
            
            # Count actual image files
            img_files = [f for f in os.listdir(img_dir) 
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"   ✓ {split}:")
            print(f"      - {n_images} images in annotations")
            print(f"      - {len(img_files)} image files")
            print(f"      - {n_annotations} annotations")
            print(f"      - {n_categories} categories")
            
            if n_images != len(img_files):
                print(f"      ⚠️  Mismatch: {n_images} annotations vs {len(img_files)} files")
            
        except Exception as e:
            print(f"   ✗ {split}: Error reading annotations: {e}")
            all_ok = False
    
    return all_ok


def check_disk_space():
    """Check available disk space"""
    print("\n💾 Checking disk space...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Free: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print("   ⚠️  Less than 5 GB free space available")
            print("   ⚠️  Training may fail due to insufficient space")
            return False
        else:
            print("   ✓ Sufficient disk space")
            return True
    except:
        print("   ⚠️  Could not check disk space")
        return True


def check_file_structure():
    """Check if all required files are present"""
    print("\n📁 Checking file structure...")
    
    # Get parent directory (project root)
    project_root = Path(__file__).parent.parent
    
    required_structure = {
        'Core Modules': ['config.py', 'dataset.py', 'model.py', 'trainer.py', 'utils.py'],
        'Main Scripts': ['train.py', 'inference.py', 'optuna_tuning.py'],
        'Utility Scripts': ['quickstart.py', 'verify_setup.py', 'export_to_huggingface.py'],
        'Documentation': ['README.md', 'INSTALLATION.md']
    }
    
    all_ok = True
    for folder, files in required_structure.items():
        folder_path = project_root / folder
        if folder_path.exists():
            print(f"   ✓ {folder}/")
            for file in files:
                file_path = folder_path / file
                if file_path.exists():
                    print(f"      ✓ {file}")
                else:
                    print(f"      ✗ {file} not found")
                    all_ok = False
        else:
            print(f"   ✗ {folder}/ not found")
            all_ok = False
    
    # Check requirements.txt in root
    if (project_root / 'requirements.txt').exists():
        print(f"   ✓ requirements.txt")
    else:
        print(f"   ✗ requirements.txt not found")
        all_ok = False
    
    return all_ok


def test_imports():
    """Test if modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    try:
        # Add Core Modules to path
        project_root = Path(__file__).parent.parent
        core_modules = str(project_root / "Core Modules")
        if core_modules not in sys.path:
            sys.path.insert(0, core_modules)
        
        from config import get_default_config
        print("   ✓ config module")
        
        from dataset import COCOSegmentationDataset
        print("   ✓ dataset module")
        
        from model import create_sam_lora_model
        print("   ✓ model module")
        
        from trainer import SAMTrainer
        print("   ✓ trainer module")
        
        from utils import set_seed, get_device
        print("   ✓ utils module")
        
        return True
    except Exception as e:
        print(f"   ✗ Import error: {e}")
        return False


def main():
    """Run all verification checks"""
    print_header("SAM LoRA Setup Verification")
    print("\nThis script will verify that everything is set up correctly.")
    
    results = {
        'Python version': check_python_version(),
        'PyTorch': check_pytorch(),
        'Dependencies': check_dependencies(),
        'SAM checkpoint': check_sam_checkpoint(),
        'COCO data': check_data(),
        'Disk space': check_disk_space(),
        'File structure': check_file_structure(),
        'Module imports': test_imports()
    }
    
    # Summary
    print_header("Verification Summary")
    
    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} - {check}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. Run quickstart: python quickstart.py")
        print("  2. Or start training: python train.py --checkpoint checkpoints/sam_vit_b_01ec64.pth")
        print("  3. Or run with menu: .\\run_training.ps1")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before training.")
        print("\nCommon solutions:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Download SAM checkpoint (see message above)")
        print("  - Verify data directory path")
    
    print()
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
