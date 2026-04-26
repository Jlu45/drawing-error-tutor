#!/usr/bin/env python3
import os
import sys
import subprocess

def check_python_version():
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9+ is required")
        sys.exit(1)
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}")

def install_dependencies():
    print("\n[1/4] Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Dependencies installed")
    except subprocess.CalledProcessError:
        print("[FAIL] Failed to install dependencies")
        sys.exit(1)

def create_config():
    print("\n[2/4] Checking configuration...")
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.example.py")
    
    if os.path.exists(config_path):
        print("[OK] config.py already exists")
        return True
    
    if os.path.exists(example_path):
        import shutil
        shutil.copy(example_path, config_path)
        print("[CREATED] config.py created from config.example.py")
        print("")
        print("  *** IMPORTANT: Edit config.py and set your API credentials ***")
        print("  *** Required fields: MULTIMODAL_API_URL, MULTIMODAL_API_KEY ***")
        print("")
        return True
    else:
        print("[FAIL] config.example.py not found")
        return False

def create_directories():
    print("\n[3/4] Creating data directories...")
    dirs = [
        "uploads",
        "data/drawings",
        "data/standard_drawings",
        "data/error_drawings",
        "data/error_labels",
        "data/gb_standards",
        "data/knowledge_base",
        "data/rl_experience",
    ]
    base = os.path.dirname(os.path.abspath(__file__))
    for d in dirs:
        path = os.path.join(base, d)
        os.makedirs(path, exist_ok=True)
    print(f"[OK] {len(dirs)} directories ready")

def verify_setup():
    print("\n[4/4] Verifying setup...")
    errors = []
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    if not os.path.exists(config_path):
        errors.append("config.py not found - run setup first")
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from config_loader import MULTIMODAL_API_URL, MULTIMODAL_API_KEY
        if not MULTIMODAL_API_URL:
            errors.append("MULTIMODAL_API_URL is empty - set it in config.py or environment")
        if not MULTIMODAL_API_KEY:
            errors.append("MULTIMODAL_API_KEY is empty - set it in config.py or environment")
    
    try:
        import flask
        print(f"  [OK] flask {flask.__version__}")
    except ImportError:
        errors.append("flask not installed")
    
    try:
        import cv2
        print(f"  [OK] opencv {cv2.__version__}")
    except ImportError:
        errors.append("opencv-python not installed")
    
    try:
        import numpy
        print(f"  [OK] numpy {numpy.__version__}")
    except ImportError:
        errors.append("numpy not installed")
    
    if errors:
        print("\n[WARN] Issues found:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print("\n[OK] All checks passed!")
    return True

def main():
    print("=" * 60)
    print("  Engineering Drawing Error Correction - Setup")
    print("=" * 60)
    
    check_python_version()
    install_dependencies()
    create_config()
    create_directories()
    verify_setup()
    
    print("\n" + "=" * 60)
    print("  Setup complete! Run the application:")
    print("    python app.py")
    print("")
    print("  Then open http://localhost:5000 in your browser")
    print("=" * 60)

if __name__ == "__main__":
    main()
