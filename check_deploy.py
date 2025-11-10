#!/usr/bin/env python3
"""
Pre-deployment verification script tailored for Hugging Face Spaces (Docker mode).
Ensures that the API dependencies, configuration files, and model weights are present.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import yaml


def _check_path(path: Path, description: str) -> bool:
    exists = path.exists()
    prefix = "✅" if exists else "❌"
    print(f"{prefix} {description}: {path}")
    return exists


def _check_requirements(required_packages: Iterable[str]) -> bool:
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False

    content = req_file.read_text(encoding="utf-8").lower()
    missing = [pkg for pkg in required_packages if pkg.lower() not in content]

    if missing:
        print(f"❌ Missing packages in requirements.txt: {', '.join(missing)}")
        return False

    print("✅ requirements.txt includes the expected packages")
    return True


def _check_config_device() -> bool:
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("⚠️  config.yaml not found (defaults will be applied at runtime)")
        return True

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        print(f"❌ Unable to parse config.yaml: {exc}")
        return False

    device = config.get("performance", {}).get("device", "cpu")
    if str(device).lower() == "cpu":
        print("✅ config.yaml is configured for CPU inference")
    else:
        print(f"⚠️  config.yaml uses device='{device}'. Hugging Face CPU spaces require 'cpu'.")
    return True


def _check_git_lfs() -> None:
    gitattributes = Path(".gitattributes")
    if not gitattributes.exists():
        print("⚠️  .gitattributes not found (Git LFS for large weights is recommended)")
        return

    content = gitattributes.read_text(encoding="utf-8")
    if "*.pt" in content and "lfs" in content.lower():
        print("✅ Git LFS is configured for PyTorch weights")
    else:
        print("⚠️  .gitattributes does not appear to configure LFS for .pt files")


def main() -> int:
    print("🔍 Hugging Face Spaces Deployment Check\n" + "=" * 60)

    success = True

    print("\n📁 Essential files")
    success &= _check_path(Path("app.py"), "FastAPI application")
    success &= _check_path(Path("Dockerfile"), "Dockerfile")
    success &= _check_path(Path("requirements.txt"), "Python dependencies")
    success &= _check_path(Path("weights/best.pt"), "YOLO model weights")

    print("\n🛠️  Optional assets")
    _check_path(Path("config.yaml"), "Config file")
    _check_path(Path("args/args.yaml"), "Training arguments")
    _check_path(Path("results/results.csv"), "Training metrics CSV")

    print("\n📦 Dependencies")
    required_packages = [
        "fastapi",
        "uvicorn",
        "ultralytics",
        "torch",
        "torchvision",
        "opencv-python-headless",
        "pillow",
        "pyyaml",
        "python-multipart",
    ]
    success &= _check_requirements(required_packages)

    print("\n⚙️  Configuration")
    success &= _check_config_device()
    _check_git_lfs()

    print("\n" + "=" * 60)
    if success:
        print("✅ Ready for Hugging Face Spaces deployment!")
        print("Next steps:")
        print("1. git add .")
        print("2. git commit -m 'Prepare API deployment'")
        print("3. git push")
        print("4. Create/Update your Space pointing to this repository")
        return 0

    print("❌ Resolve the issues above before deploying.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
