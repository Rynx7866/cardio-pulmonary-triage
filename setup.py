"""
Cardio-Pulmonary Triage System - Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="cardio_pulmonary_triage",
    version="0.1.0",
    description=(
        "Multimodal AI-powered cardiac and respiratory disease "
        "screening and triage-support system (research prototype)"
    ),
    author="Student Researcher",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "torchvision",
        "transformers",
        "xgboost",
        "opencv-python",
        "Pillow",
        "wfdb",
        "shap",
        "matplotlib",
        "seaborn",
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "streamlit",
        "pydantic",
        "pyyaml",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.12",
    ],
)
