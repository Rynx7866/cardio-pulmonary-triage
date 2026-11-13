import sys
import importlib.util

print("=" * 45)
print("MULTIMODAL HEART & LUNG AI SYSTEM")
print("ENVIRONMENT DIAGNOSTICS")
print("=" * 45)

print("\nPython version:", sys.version)
print("Python executable:", sys.executable)

packages = [
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "sklearn",
    "cv2",
    "fastapi",
    "streamlit",
    "xgboost",
    "shap",
    "wfdb",
    "seaborn",
    "pytest",
]

print("\nPACKAGE STATUS")

for package in packages:
    installed = importlib.util.find_spec(package) is not None
    status = "INSTALLED" if installed else "MISSING"
    print(f"{package:15} : {status}")

print("\nGPU STATUS")

import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("\nEnvironment diagnostic completed.")