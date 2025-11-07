---
license: apache-2.0
language:
- en
pipeline_tag: tabular-classification
---

# RLHF Training System

This directory contains the Reinforcement Learning from Human Feedback (RLHF) training pipeline for the driver behavior classification model.

## Overview

The RLHF system enables continuous improvement of the driver behavior model by:
1. Loading human-labeled data from Firebase storage (`skyledge/labeled`)
2. Combining it with existing model predictions for reinforcement learning
3. Retraining the XGBoost model with the enhanced dataset
4. Saving new model checkpoints to Hugging Face Hub

## Files

### `loader.py`
- **Purpose**: Load labeled data from Firebase storage
- **Key Features**:
  - Lists available labeled datasets from `skyledge/labeled` path
  - Tracks already processed datasets in `trained.txt`
  - Downloads and loads datasets into pandas DataFrames
  - Prevents retraining on the same data

### `saver.py`
- **Purpose**: Save trained models to Hugging Face Hub and local storage
- **Key Features**:
  - Saves model components (XGBoost model, label encoder, scaler)
  - Creates model metadata and README files
  - Uploads to Hugging Face Hub with versioning
  - Maintains local model directory structure

### `rlhf.py`
- **Purpose**: Main RLHF training pipeline
- **Key Features**:
  - Loads new labeled datasets
  - Creates RLHF dataset by combining labeled data with model predictions
  - Trains XGBoost model with enhanced dataset
  - Evaluates model performance
  - Coordinates with loader and saver modules

## API Endpoints

The RLHF training system is integrated into the main FastAPI application with the following endpoints:

### `POST /rlhf/train`
Trigger RLHF training session.

**Request Body:**
```json
{
  "max_datasets": 10,
  "force_retrain": false
}
```

**Response:**
```json
{
  "status": "success",
  "model_version": "20241201_143022",
  "datasets_processed": 5,
  "samples_processed": 1250,
  "performance_metrics": {
    "accuracy": 0.892,
    "cv_mean": 0.885,
    "cv_std": 0.012
  },
  "timestamp": "2024-12-01T14:30:22"
}
```

### `GET /rlhf/status`
Get status of RLHF training system and available labeled data.

### `GET /rlhf/trained-datasets`
Get list of datasets that have already been used for training.

## Configuration

### Environment Variables
- `HF_TOKEN`: Hugging Face authentication token
- `HF_MODEL_REPO`: Hugging Face model repository (default: `BinKhoaLe1812/Driver_Behavior_OBD`)
- `MODEL_DIR`: Local model directory (default: `/app/models/ul`)
- `FIREBASE_ADMIN_JSON`: Firebase Admin SDK credentials
- `FIREBASE_SERVICE_ACCOUNT_JSON`: Firebase service account credentials

### Firebase Storage Structure
```
skyledge-36b56.firebasestorage.app/
├── skyledge/
│   ├── processed/          # Original processed data
│   ├── labeled/            # Human-labeled data for RLHF
│   │   ├── dataset1.csv
│   │   ├── dataset2.csv
│   │   └── trained.txt     # Tracks processed datasets
│   └── logs/               # Training logs (future)
```

## Usage

## Model Versioning

We follow semantic versioning for model artifacts: `vMAJOR.MINOR`.

- Minor versions increment on each successful RLHF training: `v1.0 → v1.1 → … → v1.9`
- When the minor version reaches 9, the next version rolls over the major: `v1.9 → v2.0`
- This guarantees strict, append-only history with no overwrites.

Models are saved to:
- **Local**: `/app/models/ul/v{version}/`
- **Hugging Face**: `BinKhoaLe1812/Driver_Behavior_OBD` under `v{version}/`

## Data Flow

1. **Data Collection**: Human-labeled data stored in `skyledge/labeled/`
2. **Training Trigger**: API endpoint or manual trigger
3. **Data Loading**: Load new labeled datasets (skip already processed)
4. **RLHF Dataset**: Combine labeled data with model predictions
5. **Model Training**: Train XGBoost with enhanced dataset
6. **Evaluation**: Calculate performance metrics
7. **Model Saving**: Save to local storage and Hugging Face Hub
8. **Tracking**: Update `trained.txt` with processed datasets

## Training Techniques (Accuracy-Focused)

This RLHF system is designed to maximize generalization and accuracy while preserving prior knowledge. The pipeline uses several complementary techniques:

### 1) Dataset Construction with RLHF
- **Human-labeled data** from `skyledge/labeled/` is treated as the source of ground truth.
- We optionally incorporate the current model’s predictions to form an enhanced training signal (RLHF-style preference signal) that emphasizes areas where the model disagrees with humans.
- This prioritizes learning from difficult or previously misclassified examples, improving convergence toward human-intended behavior.

Why it improves accuracy:
- Focuses learning on disagreement regions, which are the greatest contributors to error reduction.
- Continuously integrates fresh human feedback, preventing performance stagnation.

### 2) Preprocessing Alignment (Leak-free, Consistent Features)
- The production preprocessing (feature scaling) is aligned with training via `StandardScaler` to avoid train–serve skew.
- We ensure features presented to the model at train and inference time are consistent in names, order, and scaling.

Why it improves accuracy:
- Eliminates distribution mismatch between training and serving.
- Stabilizes loss landscape for tree learners with continuous features, enabling more reliable splits.

### 3) Fine-Tuning Strategies that Preserve Knowledge
To avoid catastrophic forgetting and to methodically improve the model, we apply a tiered fine-tuning strategy when an existing model is available:

- **Continuation Training (reduced LR, few estimators)**
  - Trains a small number of additional trees with a lower learning rate on an enhanced feature space (original features + existing model soft predictions).
  - Effectively performs incremental learning while leveraging prior decision boundaries.

- **Knowledge Distillation**
  - Uses the existing model’s soft probabilities as auxiliary signals alongside ground-truth labels.
  - The student (new model) learns both from data and teacher signals, smoothing decision boundaries and improving calibration.

- **Ensemble with the Existing Model**
  - Combines existing and newly trained models (weighted probabilities) when continuation/distillation are not suitable.
  - Ensures performance never regresses sharply and benefits from complementary strengths.

Why it improves accuracy and stability:
- Preserves previously learned behaviors while adapting to new data distributions.
- Distillation integrates teacher knowledge, often improving calibration and top-1 accuracy.
- Ensemble provides robustness when new data is limited or shifts are partial.

### 4) Model Evaluation and Early Feedback
- We compute held-out **accuracy** and **cross-validation** scores (`cv_mean ± cv_std`) on the enhanced dataset.
- The system logs per-version metrics and training metadata to support regression detection and A/B comparisons.

Why it improves accuracy:
- Cross-validation reduces variance in estimates and guards against overfitting to a single split.
- Per-version metrics enforce an iterative, measurable improvement cycle.

### 5) Versioned, Append-Only Artifacts
- Each successful training run produces a new semantic version directory (`v{version}/`) both locally and on the Hub.
- Older versions remain intact for rollback and benchmarking.

Why it improves accuracy long-term:
- Enables safe experimentation and rapid rollback if a version underperforms in production.
- Facilitates longitudinal evaluation and selection of best-performing checkpoints.

## Performance Monitoring

The system tracks:
- Number of datasets processed
- Total samples processed
- Model accuracy and cross-validation scores
- Training timestamps and metadata

## Error Handling

- Graceful handling of missing datasets
- Firebase connection failures
- Model loading/saving errors
- XGBoost compatibility issues
- Comprehensive logging throughout the pipeline
