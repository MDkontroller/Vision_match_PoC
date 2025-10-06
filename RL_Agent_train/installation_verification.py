# Installation Test Script for Drone Localization RL Agent
# Run this after installing dependencies to verify everything works

import sys
import os
from pathlib import Path

def test_installation():
    """
    Comprehensive test of all required dependencies
    """
    
    print("🧪" + "="*60 + "🧪")
    print("    INSTALLATION TEST - DRONE LOCALIZATION AGENT")
    print("🧪" + "="*60 + "🧪")
    
    print(f"\n💻 System Information:")
    print(f"   Python Version: {sys.version}")
    print(f"   Platform: {sys.platform}")
    print(f"   Working Directory: {os.getcwd()}")
    
    # Test results tracking
    tests_passed = 0
    total_tests = 10
    
    print(f"\n🔍 Testing Dependencies...")
    
    # Test 1: PyTorch
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ PyTorch: Not installed ({e})")
    
    # Test 2: CUDA Support
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"   ✅ CUDA: Available (Version {torch.version.cuda})")
            print(f"      Device: {torch.cuda.get_device_name(0)}")
            print(f"      Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            tests_passed += 1
        else:
            print(f"   ⚠️ CUDA: Not available (will use CPU)")
            print(f"      This will work but training will be slower")
    except Exception as e:
        print(f"   ❌ CUDA: Error checking CUDA ({e})")
    
    # Test 3: TorchVision
    try:
        import torchvision
        print(f"   ✅ TorchVision: {torchvision.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ TorchVision: Not installed ({e})")
    
    # Test 4: OpenCV
    try:
        import cv2
        print(f"   ✅ OpenCV: {cv2.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ OpenCV: Not installed ({e})")
    
    # Test 5: NumPy
    try:
        import numpy as np
        print(f"   ✅ NumPy: {np.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ NumPy: Not installed ({e})")
    
    # Test 6: Scikit-learn
    try:
        import sklearn
        print(f"   ✅ Scikit-learn: {sklearn.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Scikit-learn: Not installed ({e})")
    
    # Test 7: Scikit-image
    try:
        import skimage
        print(f"   ✅ Scikit-image: {skimage.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Scikit-image: Not installed ({e})")
    
    # Test 8: Matplotlib
    try:
        import matplotlib
        print(f"   ✅ Matplotlib: {matplotlib.__version__}")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Matplotlib: Not installed ({e})")
    
    # Test 9: PIL/Pillow
    try:
        from PIL import Image
        print(f"   ✅ Pillow (PIL): Available")
        tests_passed += 1
    except ImportError as e:
        print(f"   ❌ Pillow (PIL): Not installed ({e})")
    
    # Test 10: File structure
    try:
        tif_found = False
        crops_found = False
        
        # Check for TIF file
        tif_extensions = ['.TIF', '.tif', '.TIFF', '.tiff']
        for ext in tif_extensions:
            tif_files = list(Path('.').glob(f'*sentinel2_ukraine_10km*{ext}'))
            if tif_files:
                print(f"   ✅ TIF File: Found {tif_files[0]}")
                tif_found = True
                break
        
        if not tif_found:
            print(f"   ⚠️ TIF File: sentinel2_ukraine_10km not found in current directory")
        
        # Check for crops directory
        crops_dir = Path('realistic_drone_crops')
        if crops_dir.exists():
            jpg_files = list(crops_dir.glob('*.jpg')) + list(crops_dir.glob('*.jpeg'))
            json_files = list(crops_dir.glob('*.json'))
            
            if jpg_files and json_files:
                print(f"   ✅ Crops Directory: {len(jpg_files)} JPG files, {len(json_files)} metadata files")
                crops_found = True
            else:
                print(f"   ⚠️ Crops Directory: Missing JPG files or metadata")
        else:
            print(f"   ⚠️ Crops Directory: realistic_drone_crops/ not found")
        
        if tif_found and crops_found:
            tests_passed += 1
        
    except Exception as e:
        print(f"   ❌ File Structure: Error checking files ({e})")
    
    # Test summary
    print(f"\n📊 Test Results:")
    print(f"   Passed: {tests_passed}/{total_tests}")
    
    if tests_passed >= 8:
        print(f"   🎉 Excellent! Your setup is ready for training")
        status = "READY"
    elif tests_passed >= 6:
        print(f"   ⚠️ Good setup, but some optional components missing")
        status = "MOSTLY_READY"
    else:
        print(f"   ❌ Several issues found, please install missing dependencies")
        status = "NOT_READY"
    
    return status, tests_passed, total_tests

def test_gpu_training():
    """
    Test GPU training capability
    """
    print(f"\n🚀 Testing GPU Training Capability...")
    
    try:
        import torch
        import torch.nn as nn
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Create a small test model
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10)
        ).to(device)
        
        # Test forward pass
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(test_input)
        
        print(f"   ✅ GPU Training Test: Passed")
        print(f"      Input shape: {test_input.shape}")
        print(f"      Output shape: {output.shape}")
        print(f"      Device: {output.device}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GPU Training Test: Failed ({e})")
        return False

def provide_installation_help():
    """
    Provide specific installation commands based on test results
    """
    print(f"\n🔧 Installation Help:")
    print(f"   If tests failed, run these commands:")
    print(f"")
    print(f"   # For PyTorch + CUDA 12.4:")
    print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print(f"")
    print(f"   # For other dependencies:")
    print(f"   pip install opencv-python scikit-image scikit-learn matplotlib Pillow")
    print(f"")
    print(f"   # If CUDA not working:")
    print(f"   1. Check nvidia-smi command shows CUDA 12.4+")
    print(f"   2. Reinstall PyTorch with correct CUDA version")
    print(f"   3. Restart Python/Jupyter after installation")

def check_memory_requirements():
    """
    Check if system has enough memory for training
    """
    print(f"\n💾 Memory Requirements Check:")
    
    try:
        import psutil
        
        # System RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   System RAM: {ram_gb:.1f} GB")
        
        if ram_gb >= 16:
            print(f"   ✅ Excellent RAM for training")
        elif ram_gb >= 8:
            print(f"   ⚠️ Minimum RAM available, reduce batch size if needed")
        else:
            print(f"   ❌ Low RAM, training may be slow or fail")
        
        # GPU Memory
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 8:
                print(f"   ✅ Excellent GPU memory")
            elif gpu_memory >= 4:
                print(f"   ⚠️ Good GPU memory")
            else:
                print(f"   ❌ Low GPU memory, use smaller batch sizes")
        
    except ImportError:
        print(f"   Install psutil for detailed memory info: pip install psutil")
    except Exception as e:
        print(f"   Could not check memory: {e}")

if __name__ == "__main__":
    # Run comprehensive tests
    status, passed, total = test_installation()
    
    # Test GPU training
    if status in ["READY", "MOSTLY_READY"]:
        gpu_ok = test_gpu_training()
        check_memory_requirements()
    
    # Provide help if needed
    if status == "NOT_READY":
        provide_installation_help()
    
    print(f"\n🎯 Next Steps:")
    if status == "READY":
        print(f"   1. Run: list_available_crops()")
        print(f"   2. Run: test_setup()")  
        print(f"   3. Run: trainer, results = train_drone_localization_agent()")
    elif status == "MOSTLY_READY":
        print(f"   1. Install missing optional dependencies")
        print(f"   2. Run training with reduced settings if needed")
    else:
        print(f"   1. Install missing critical dependencies")
        print(f"   2. Re-run this test script")
    
    print(f"\n🚀 Installation test complete!")