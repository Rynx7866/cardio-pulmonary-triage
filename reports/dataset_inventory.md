# Dataset Inventory

## Multimodal Heart and Lung AI Triage System

Last updated: 15 November 2026

## 1\. ECG Dataset — PTB-XL

* Source: https://physionet.org/content/ptb-xl/1.0.3/
* Version: 1.0.3
* Records: 21,837
* Patients: 18,885
* Leads: 12
* Duration: 10 seconds
* Sampling rates: 100 Hz and 500 Hz
* License: CC BY 4.0
* Approximate uncompressed size: 3.0 GB
* Task: ECG diagnostic classification
* Planned approach: 100 Hz subset initially
* Split: Official patient-separated folds
* Model: 1D CNN

## 2\. Chest X-ray Dataset — NIH ChestX-ray14

* Source: https://nihcc.app.box.com/v/ChestXray-Nihcc
* Images: 112,120
* Patients: 30,805
* Labels: 14 thoracic findings
* Format: PNG
* Task: Multilabel chest X-ray classification
* Planned approach: Small pilot subset first
* Split: Patient-level separation
* Model: Pretrained ResNet50

Note: Labels are derived from radiology reports and can be noisy.
Check official download instructions and attribution requirements.

## 3\. Cardiac Clinical Dataset — UCI Heart Disease

* Source: https://archive.ics.uci.edu/dataset/45/heart+disease
* Planned subset: Cleveland
* Records: 303
* Features: 13
* Target: Presence or absence of heart disease
* Task: Binary classification
* Model: XGBoost
* Evaluation: Stratified cross-validation

Note: Small dataset; results must be interpreted cautiously.

## 4\. Respiratory Dataset — Exasens

* Source: https://archive.ics.uci.edu/dataset/566/exasens
* Records: 399
* Features: 4
* Groups: COPD, asthma, infected, healthy controls
* Data: Demographic and saliva biosensor measurements
* Task: Exploratory respiratory group classification
* Model: Random Forest
* Evaluation: Stratified cross-validation

Note: This is not a conventional hospital clinical dataset.
Its findings must not be presented as validated clinical diagnosis.

## 5\. Linked Multimodal Dataset — Symile-MIMIC

* Source: https://physionet.org/content/symile-mimic/
* Modalities: Chest X-rays, ECGs and blood laboratory measurements
* Access: Restricted
* Requirements:

  * PhysioNet credentialing
  * CITI research training
  * Signed data-use agreement
* Purpose: Patient-level multimodal research
* Status: Access application to be investigated

## 6\. Data Integrity Rules

1. Never combine unrelated patients across datasets.
2. Use patient-level train/validation/test splits.
3. Fit preprocessing transformations on training data only.
4. Keep raw data separate from processed data.
5. Do not upload restricted or patient-level data to GitHub.
6. Record dataset versions, licenses and citations.
7. Treat model predictions as research outputs, not clinical diagnoses.

## 7\. Download Status

\### PTB-XL

\- Status: Downloaded

\- Version: 1.0.3

\- Sampling rate used: 100 Hz

\- Metadata and ECG loader: Validated

\- Preprocessing: Validated

\- PyTorch Dataset and DataLoader: Validated

\- CNN architecture: Implemented and tested

\- Training pipeline: Smoke test passed

\- Smoke-test configuration: 32 training records, 16 validation records, 1 epoch (CPU)

\- Smoke-test result: Pipeline execution and checkpoint saving verified; not a performance evaluation

\- Full model training: Pending Google Colab GPU



\### Other datasets

\- NIH ChestX-ray14: Pending

\- UCI Heart Disease: Pending

\- Exasens: Pending

\- Symile-MIMIC: Access investigation pending
