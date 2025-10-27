#!/usr/bin/env python3
"""
Pre-deployment verification script for Streamlit Cloud
Verifies all necessary files are present and configured correctly
"""

import sys
from pathlib import Path
import yaml

def check_file_exists(file_path, description):
    """Verifies if a file exists"""
    path = Path(file_path)
    if path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} not found: {file_path}")
        return False

def check_directory_exists(dir_path, description):
    """Verifies if a directory exists"""
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        files = list(path.iterdir())
        print(f"✅ {description}: {dir_path} ({len(files)} files)")
        return True
    else:
        print(f"❌ {description} not found: {dir_path}")
        return False

def check_requirements():
    """Checks requirements.txt"""
    required_packages = [
        'streamlit',
        'ultralytics',
        'torch',
        'torchvision',
        'opencv-python-headless',
        'pillow',
        'pandas',
        'plotly',
        'pyyaml',
        'streamlit-option-menu',
        'streamlit-image-select'
    ]
    
    req_file = Path('requirements.txt')
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    with open(req_file) as f:
        content = f.read().lower()
    
    missing = []
    for pkg in required_packages:
        if pkg.lower() not in content:
            missing.append(pkg)
    
    if missing:
        print(f"❌ Missing packages in requirements.txt: {', '.join(missing)}")
        return False
    else:
        print(f"✅ requirements.txt has all required packages")
        return True

def check_git_lfs():
    """Verifies if Git LFS is configured"""
    gitattributes = Path('.gitattributes')
    if gitattributes.exists():
        with open(gitattributes) as f:
            content = f.read()
        if '*.pt' in content and 'lfs' in content:
            print("✅ Git LFS configured for .pt files")
            return True
        else:
            print("❌ Git LFS not configured correctly in .gitattributes")
            return False
    else:
        print("⚠️  .gitattributes not found (Git LFS may not be configured)")
        return False

def check_config_yaml():
    """Checks config.yaml"""
    config_file = Path('config.yaml')
    if not config_file.exists():
        print("⚠️  config.yaml not found (will use defaults)")
        return True
    
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        # Check if device is configured for CPU
        device = config.get('performance', {}).get('device', 'cpu')
        if device == 'cpu':
            print("✅ config.yaml configured for CPU (correct for Streamlit Cloud)")
        else:
            print(f"⚠️  config.yaml with device='{device}' (recommended: 'cpu' for Streamlit Cloud)")
        
        return True
    except Exception as e:
        print(f"❌ Error reading config.yaml: {e}")
        return False

def main():
    print("🔍 Streamlit Cloud Deploy Verification\n")
    print("="*60)
    
    all_ok = True
    
    # Essential files
    print("\n📁 Essential Files:")
    all_ok &= check_file_exists('app.py', 'Main app')
    all_ok &= check_file_exists('requirements.txt', 'Python dependencies')
    all_ok &= check_file_exists('packages.txt', 'System dependencies')
    all_ok &= check_file_exists('.streamlit/config.toml', 'Streamlit config')
    
    # Model
    print("\n🤖 Model:")
    all_ok &= check_file_exists('weights/best.pt', 'YOLO model')
    
    # Data and results
    print("\n📊 Data and Results:")
    check_directory_exists('images', 'Test images')
    check_directory_exists('results', 'Training results')
    check_file_exists('results/results.csv', 'Results CSV')
    check_file_exists('args/args.yaml', 'Training args')
    
    # Configuration
    print("\n⚙️  Configuration:")
    all_ok &= check_requirements()
    check_git_lfs()
    check_config_yaml()
    
    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✅ All ready for Streamlit Cloud deployment!")
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'Prepare for deployment'")
        print("3. git push origin main")
        print("4. Deploy at https://share.streamlit.io")
        return 0
    else:
        print("❌ There are issues that need to be fixed")
        print("\nCheck the items marked with ❌ above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
