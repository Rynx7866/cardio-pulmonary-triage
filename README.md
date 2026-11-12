# 🏥 Cardio-Pulmonary AI Triage System

> **Multimodal AI-powered cardiac and respiratory disease screening and triage-support system.**

⚠️ **DISCLAIMER**: This is an academic research prototype developed as a college project.
It is **NOT** a clinically validated diagnostic tool and must **NOT** be used for
real patient care, emergency decision-making, or clinical diagnosis.

---

## 📋 Overview

This system analyses multiple types of patient data using specialised AI/ML models to
identify potential cardiac and respiratory abnormalities and provide triage-support
categorisation.

### Input Modalities

| # | Modality | Model | Architecture |
|---|----------|-------|--------------|
| 1 | Chest X-ray images | X-ray Classifier | ResNet50 (Transfer Learning) |
| 2 | ECG recordings | ECG Classifier | 1D CNN |
| 3 | Structured clinical data | Cardiac Risk Predictor | XGBoost |
| 4 | Respiratory measurements | Respiratory Risk Assessor | Random Forest |
| 5 | Medical text / clinical notes | Clinical NLP | Bio_ClinicalBERT |
| 6 | Combined multimodal features | Fusion Network | Neural Network (Late Fusion) |

### Triage Categories

| Level | Category | Description |
|-------|----------|-------------|
| 🔴 1 | Emergency Assessment Required | Immediate clinical attention recommended |
| 🟡 2 | Urgent Clinical Assessment Recommended | Prompt clinical review recommended |
| 🟢 3 | Routine Clinical Assessment | No urgent findings, standard follow-up |

---

## 🏗️ Project Structure

```
cardio_pulmonary_triage/
├── configs/            # YAML configuration files
├── datasets/           # Dataset storage (gitignored)
├── src/                # Core source code
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # Model architectures
│   ├── training/       # Training scripts
│   ├── evaluation/     # Evaluation metrics
│   ├── explainability/ # Grad-CAM, SHAP
│   ├── triage/         # Triage engine & rules
│   └── utils/          # Shared utilities
├── notebooks/          # Jupyter notebooks
├── trained_models/     # Saved model weights (gitignored)
├── backend/            # FastAPI REST API
├── frontend/           # Streamlit dashboard
├── tests/              # Unit & integration tests
├── reports/            # Evaluation reports & figures
└── colab/              # Google Colab training notebooks
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| Language | Python 3.12 |
| Deep Learning | PyTorch, torchvision |
| Machine Learning | scikit-learn, XGBoost |
| NLP | Hugging Face Transformers |
| Computer Vision | OpenCV, Pillow |
| Signal Processing | wfdb, SciPy |
| Explainability | Grad-CAM, SHAP |
| Backend | FastAPI, SQLAlchemy, SQLite |
| Frontend | Streamlit |
| Visualisation | Matplotlib, Seaborn |
| Configuration | PyYAML |
| Testing | pytest |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11 or 3.12
- Git
- (Optional) NVIDIA GPU with CUDA for local training

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd cardio_pulmonary_triage

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run environment verification
python -m pytest tests/test_environment.py -v
```

### Run the Dashboard

```bash
# Start the FastAPI backend
cd backend
uvicorn main:app --reload --port 8000

# In a separate terminal, start the Streamlit frontend
cd frontend
streamlit run app.py
```

---

## 📊 Datasets

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| PTB-XL | ECG recordings | ~2.8 GB | [PhysioNet](https://physionet.org/content/ptb-xl/) |
| NIH ChestX-ray14 | Chest X-rays | ~45 GB | [NIH](https://nihcc.app.box.com/v/ChestXray-Nihcc) |

> Datasets are **NOT** included in this repository. See the dataset documentation
> for download instructions and licensing terms.

---

## 📈 Model Evaluation

Every model is evaluated independently using:

- Accuracy, Precision, Recall (Sensitivity), Specificity
- F1-Score, ROC-AUC
- Confusion Matrix
- Calibration Analysis
- Patient-level train/validation/test splits (no data leakage)

Special attention is paid to **false negatives** (missed emergencies).

---

## 🔒 Safety Design Principles

1. **Emergency symptom override**: Dangerous symptoms ALWAYS escalate to emergency
   regardless of AI risk scores.
2. **Missing data warnings**: The system explicitly warns when data is incomplete.
3. **Confidence reporting**: All predictions include uncertainty estimates.
4. **No silent failures**: Every model output includes confidence levels and limitations.
5. **Clinical disclaimers**: All outputs carry clear "not for clinical use" warnings.

---

## 📜 License

This project is developed for academic purposes. Dataset usage is governed by
their respective licenses (see dataset documentation).

---

## 🙏 Acknowledgements

- [PTB-XL Dataset](https://physionet.org/content/ptb-xl/) — PhysioNet
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-Nihcc) — National Institutes of Health
- PyTorch, scikit-learn, Hugging Face, and all open-source contributors
