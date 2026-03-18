# ECG Biometric Identification: GBDT vs. 1D-CNN

This project implements a biometric identification system using Electrocardiogram (ECG) signals. 
Unlike traditional biometrics like fingerprints or facial recognition, ECG-based identification relies on the unique physiological "signature" of an individual's cardiac morphology—the specific shape and timing of their heartbeats, which are influenced by heart size, orientation, and conduction pathways.

## The Dataset: PTB-XL

The foundation of this study is the PTB-XL dataset, a large-scale, clinical ECG database.

Scale: Contains 21,837 clinical ECG records from 18,885 patients.

High Fidelity: Signals are provided at a 500Hz sampling rate, offering the granular temporal resolution necessary to detect subtle, individual-specific variations in the QRS complex.

Multi-Lead Data: For this task, we utilize Leads I and II, providing a multi-vector view of the heart's electrical activity, which significantly increases the "entropy" of the biometric template.

The goal is to extract a unique "cardiac fingerprint" from 2-lead ECG signals (Lead I and II).

## The Task: Closed-Set Identification

The core challenge is a multi-class classification problem where the goal is to map a short segment of an ECG signal to a specific identity.

Input: A 400ms window centered on the R-peak of a filtered, normalized ECG signal (2 channels).
Target (Y): The unique Patient ID.

# Installation & Setup

This project uses Conda for environment management to ensure compatibility between CatBoost and PyTorch.

## Project Structure

```
ECG-Biometric-ID/
├── data/                   # (Excluded from Git) Raw PTB-XL files
├── notebooks/
│   ├── 01_data_loading.ipynb      # Was: ptb_xl_data_loader
│   ├── 02_preprocessing.ipynb     # Was: ptb_xl_data_processing
│   └── 03_model_benchmarking.ipynb # Was: train_ml_models
├── environment.yml         # Conda environment definition
└── README.md               # Project documentation
```

### Create the environment
conda env create -f environment.yml

### Activate the environment
conda activate ecg-biometrics

# Pipeline Stages

## 1. Data Acquisition (01_data_loading.ipynb)

Fetches the PTB-XL dataset (WFDB format).
Filters for high-quality clinical records with consistent lead placement.

## 2. Signal Processing (02_preprocessing.ipynb)

QRS Detection: Locates the heartbeats within the 10-second strips.
Segmentation: Extracts 200ms windows around the R-peak.
Normalization: Applies Z-score standardization to stabilize gradients for CNN training.

## 3. Model Benchmarking (03_model_benchmarking.ipynb)

We compare two distinct architectural paradigms:

| Model | Architecture | Performance (Test Accuracy) |
|-------|-------------|----------------------------|
| CatBoost | GBDT (Gradient Boosted Decision Trees) with symmetric trees | **90.0%** (evaluated on 300 patients) |
| 1D-CNN | PyTorch implementation with Conv1d layers and BatchNorm | 68.1% (evaluated on all 10,742 patients) |

# Key Observations:
- CatBoost demonstrates excellent accuracy but was tested on a smaller subset of patients
- CatBoost provides extreme precision on smaller subsets but encounters a computational bottleneck as class counts exceed 1,000
- The 1D-CNN model was evaluated on the complete dataset, providing a more comprehensive assessment
- The performance gap suggests that tree-based models may capture ECG morphology features more effectively than convolutional approaches for this specific task


