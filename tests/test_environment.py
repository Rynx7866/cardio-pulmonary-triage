"""
Environment Verification Tests
===============================
Run these tests after setting up the environment to verify that
all dependencies are correctly installed and importable.

HOW TO RUN:
    python -m pytest tests/test_environment.py -v

WHAT THIS TESTS:
    - All required Python packages are importable.
    - Package versions meet minimum requirements.
    - Device detection works (CPU at minimum).
    - Configuration files are loadable.
    - Project directory structure is intact.
"""

import sys
import importlib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Project root (two levels up from this test file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Test 1: Python version
# ---------------------------------------------------------------------------
class TestPythonVersion:
    """Verify the Python interpreter version."""

    def test_python_minimum_version(self):
        """Python must be 3.11 or newer."""
        assert sys.version_info >= (3, 11), (
            f"Python 3.11+ required, got {sys.version_info.major}.{sys.version_info.minor}"
        )

    def test_python_not_too_old(self):
        """Python should be 3.11 or 3.12."""
        assert sys.version_info.major == 3
        assert sys.version_info.minor in (11, 12, 13)


# ---------------------------------------------------------------------------
# Test 2: Core package imports
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
]

# Packages installed during Phase 1 setup
PHASE1_PACKAGES = [
    ("xgboost", "XGBoost"),
    ("shap", "SHAP"),
    ("wfdb", "wfdb"),
    ("seaborn", "seaborn"),
]

WEB_PACKAGES = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "uvicorn"),
    ("streamlit", "streamlit"),
    ("sqlalchemy", "SQLAlchemy"),
    ("pydantic", "pydantic"),
]

NLP_PACKAGES = [
    ("transformers", "Transformers"),
]


class TestCoreImports:
    """Verify all required packages can be imported."""

    @pytest.mark.parametrize("module_name,display_name", REQUIRED_PACKAGES)
    def test_core_package_import(self, module_name, display_name):
        """Core scientific packages must be importable."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"{display_name} ({module_name}) failed to import"

    @pytest.mark.parametrize("module_name,display_name", PHASE1_PACKAGES)
    def test_phase1_package_import(self, module_name, display_name):
        """Phase 1 installed packages must be importable."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"{display_name} ({module_name}) failed to import"

    @pytest.mark.parametrize("module_name,display_name", WEB_PACKAGES)
    def test_web_package_import(self, module_name, display_name):
        """Web framework packages must be importable."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"{display_name} ({module_name}) failed to import"

    @pytest.mark.parametrize("module_name,display_name", NLP_PACKAGES)
    def test_nlp_package_import(self, module_name, display_name):
        """NLP packages must be importable."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"{display_name} ({module_name}) failed to import"


# ---------------------------------------------------------------------------
# Test 3: PyTorch device detection
# ---------------------------------------------------------------------------
class TestDeviceDetection:
    """Verify PyTorch device detection works."""

    def test_torch_cpu_available(self):
        """CPU should always be available."""
        import torch
        device = torch.device("cpu")
        assert device.type == "cpu"

    def test_torch_tensor_creation(self):
        """Should be able to create and operate on tensors."""
        import torch
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x @ y  # Matrix multiplication
        assert z.shape == (3, 3)

    def test_device_utility(self):
        """Our custom device utility should work."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.device import get_device, get_device_info

        device = get_device()
        assert device.type in ("cpu", "cuda")

        info = get_device_info()
        assert "pytorch_version" in info
        assert "cuda_available" in info


# ---------------------------------------------------------------------------
# Test 4: Configuration files
# ---------------------------------------------------------------------------
class TestConfigFiles:
    """Verify configuration files exist and are valid YAML."""

    @pytest.mark.parametrize("config_name", [
        "model_config.yaml",
        "triage_rules.yaml",
        "data_config.yaml",
    ])
    def test_config_file_exists(self, config_name):
        """Config files must exist."""
        config_path = PROJECT_ROOT / "configs" / config_name
        assert config_path.exists(), f"Missing config: {config_path}"

    @pytest.mark.parametrize("config_name", [
        "model_config.yaml",
        "triage_rules.yaml",
        "data_config.yaml",
    ])
    def test_config_file_valid_yaml(self, config_name):
        """Config files must be valid YAML."""
        import yaml
        config_path = PROJECT_ROOT / "configs" / config_name
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{config_name} did not parse as dict"

    def test_model_config_has_all_models(self):
        """model_config.yaml must define all six models."""
        import yaml
        config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        expected_keys = ["xray", "ecg", "cardiac", "respiratory", "nlp", "fusion"]
        for key in expected_keys:
            assert key in config, f"Missing model config section: '{key}'"

    def test_triage_rules_has_categories(self):
        """triage_rules.yaml must define three urgency categories."""
        import yaml
        config_path = PROJECT_ROOT / "configs" / "triage_rules.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "categories" in config
        categories = config["categories"]
        for cat in ["emergency", "urgent", "routine"]:
            assert cat in categories, f"Missing triage category: '{cat}'"

    def test_triage_rules_has_emergency_symptoms(self):
        """triage_rules.yaml must define emergency symptom overrides."""
        import yaml
        config_path = PROJECT_ROOT / "configs" / "triage_rules.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert "emergency_symptoms" in config
        symptoms = config["emergency_symptoms"]
        assert len(symptoms) > 0, "Emergency symptoms list is empty"


# ---------------------------------------------------------------------------
# Test 5: Project directory structure
# ---------------------------------------------------------------------------
class TestProjectStructure:
    """Verify all required directories exist."""

    REQUIRED_DIRS = [
        "configs",
        "datasets",
        "datasets/ecg",
        "datasets/chest_xray",
        "datasets/clinical",
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/explainability",
        "src/triage",
        "src/utils",
        "notebooks",
        "trained_models",
        "backend",
        "backend/api",
        "backend/services",
        "backend/database",
        "frontend",
        "frontend/pages",
        "frontend/components",
        "tests",
        "reports",
        "colab",
    ]

    @pytest.mark.parametrize("dir_path", REQUIRED_DIRS)
    def test_directory_exists(self, dir_path):
        """Required directories must exist."""
        full_path = PROJECT_ROOT / dir_path
        assert full_path.is_dir(), f"Missing directory: {full_path}"

    def test_src_is_package(self):
        """src/ must have __init__.py to be a Python package."""
        assert (PROJECT_ROOT / "src" / "__init__.py").exists()

    def test_requirements_file_exists(self):
        """requirements.txt must exist."""
        assert (PROJECT_ROOT / "requirements.txt").exists()

    def test_gitignore_exists(self):
        """.gitignore must exist."""
        assert (PROJECT_ROOT / ".gitignore").exists()


# ---------------------------------------------------------------------------
# Test 6: Utility modules
# ---------------------------------------------------------------------------
class TestUtilityModules:
    """Verify our custom utility modules work correctly."""

    def test_logger_creation(self):
        """Logger utility should create a functioning logger."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.logger import get_logger

        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
        # Should not raise
        logger.info("Environment test: logger works")

    def test_io_utils_yaml(self):
        """IO utility should load YAML configs."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.io_utils import load_yaml

        config = load_yaml(PROJECT_ROOT / "configs" / "model_config.yaml")
        assert isinstance(config, dict)
        assert "xray" in config

    def test_io_utils_ensure_dir(self):
        """ensure_dir should create directories."""
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.io_utils import ensure_dir
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "test_subdir" / "nested"
            result = ensure_dir(test_path)
            assert result.is_dir()


# ---------------------------------------------------------------------------
# Test 7: Metrics module
# ---------------------------------------------------------------------------
class TestMetricsModule:
    """Verify the evaluation metrics module works."""

    def test_binary_classification_metrics(self):
        """Metrics should compute correctly for binary classification."""
        import numpy as np
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.evaluation.metrics import compute_classification_metrics

        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.6, 0.9, 0.8, 0.3, 0.2, 0.7, 0.15])

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "confusion_matrix" in metrics
        assert "false_negative_rate" in metrics

        # Sensitivity equals recall
        assert metrics["sensitivity"] == metrics["recall"]

        # Values should be between 0 and 1
        for key in ["accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_multiclass_classification_metrics(self):
        """Metrics should handle multiclass classification."""
        import numpy as np
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.evaluation.metrics import compute_classification_metrics

        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1])

        metrics = compute_classification_metrics(y_true, y_pred, average="weighted")

        assert "accuracy" in metrics
        assert "specificity" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
