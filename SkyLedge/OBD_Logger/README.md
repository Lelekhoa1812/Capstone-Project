---
title: OBD Logger
emoji: 🚗
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
short_description: OBD-logging FastAPI server with data processing pipelines
---

# OBD Logger

A comprehensive OBD-II data logging and processing system built with FastAPI, featuring advanced data cleaning, Google Drive integration, MongoDB storage capabilities, **Reinforcement Learning from Human Feedback (RLHF)** for driver behavior classification, and **fuel efficiency scoring** using machine learning models.

![System Architecture](diagram/diagram.svg)

## Features

- **Real-time OBD-II Data Ingestion**: Stream and process OBD sensor data in real-time
- **Advanced Data Cleaning**: Intelligent gap detection, KNN imputation, and outlier handling
- **Multi-Storage Architecture**: 
  - Google Drive integration for CSV storage
  - Firebase for structured data storage and querying
  - MongoDB Atlas for structured data storage and querying
- **Driver Behavior Classification**: XGBoost-based ML model for driving style prediction
- **Fuel Efficiency Scoring**: ML model for drive-level fuel efficiency prediction (0-100%)
- **RLHF Training System**: Continuous model improvement through human feedback
- **Data Visualization**: Automatic generation of correlation heatmaps and trend plots
- **RESTful API**: Comprehensive endpoints for data management and retrieval
- **Web Dashboard**: User-friendly interface for monitoring and control
- **Model Versioning**: Semantic versioning (1.0, 1.1, 1.2, etc.) with Hugging Face integration

## Architecture

The application is structured into modular components:

- **`app.py`**: Main FastAPI application with data processing pipeline and RLHF endpoints
- **`data/`**: Storage and persistence modules
  - **`drive_saver.py`**: Google Drive operations and file management
  - **`mongo_saver.py`**: MongoDB operations and data persistence
  - **`firebase_saver.py`**: Firebase operations and data persistence
- **`train/`**: RLHF training system
  - **`loader.py`**: Load labeled data from Firebase storage with original dataset tracking
  - **`saver.py`**: Save trained models to Hugging Face Hub with semantic versioning
  - **`rlhf.py`**: Main RLHF training pipeline for continuous model improvement
- **`OBD/`**: OBD-specific modules for data analysis and logging
- **`utils/`**: Utility modules for model management and data processing
- **`efficiency/`**: Fuel efficiency model training and evaluation
  - **`retrain.py`**: Train and upload fuel efficiency models to Hugging Face
  - **`eval.py`**: Evaluate fuel efficiency on OBD data

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   - `GDRIVE_CREDENTIALS_JSON`: Google Service Account credentials
   - `FIREBASE_SERVICE_ACCOUNT_JSON`: Firebase connection string
   - `FIREBASE_ADMIN_JSON`: Firebase Admin SDK credentials
   - `HF_TOKEN`: Hugging Face authentication token
   - `HF_MODEL_REPO`: Driver behavior model repository (default: `BinKhoaLe1812/Driver_Behavior_OBD`)
   - `HF_EFFICIENCY_MODEL_REPO`: Fuel efficiency model repository (default: `BinKhoaLe1812/Fuel_Efficiency_OBD`)
   - `MODEL_DIR`: Driver behavior model directory (default: `/app/models/ul`)
   - `EFFICIENCY_MODEL_DIR`: Fuel efficiency model directory (default: `/app/models/efficiency`)

3. **Run the Application**:
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the Dashboard**:
   - Web UI: `https://binkhoale1812-obd-logger.hf.space/ui`
   - API Docs: `https://binkhoale1812-obd-logger.hf.space/docs`

## Data Processing Pipeline

1. **Ingestion**: Real-time streaming or bulk CSV upload
2. **Cleaning**: Automatic gap detection and KNN imputation
3. **Feature Engineering**: Derived metrics and sensor combinations
4. **Storage**: Simultaneous save to Google Drive, Firebase, and MongoDB
5. **Driver Behavior Classification**: XGBoost model prediction on processed data
6. **Fuel Efficiency Scoring**: ML model prediction for drive-level efficiency (0-100%)
7. **RLHF Training**: Continuous model improvement through human feedback
8. **Visualization**: Correlation analysis and trend plots

## API Endpoints

### Data Ingestion
- `POST /ingest`: Stream OBD data
- `POST /upload-csv/`: Bulk CSV upload

### Data Retrieval
- `GET /download/{filename}`: Download cleaned CSV
- `GET /events`: Get processing status
- `GET /predictions/latest`: Get latest driver behavior and fuel efficiency predictions
- `GET /efficiency/{filename}`: Get fuel efficiency prediction for specific processed file

### MongoDB Operations
- `GET /mongo/status`: Check MongoDB connection
- `GET /mongo/sessions`: Get data session summaries
- `GET /mongo/query`: Query data with filters
- `POST /mongo/save-csv`: Direct CSV to MongoDB

### RLHF Training System
- `POST /rlhf/train`: Trigger RLHF training session
- `GET /rlhf/status`: Get RLHF system status and available labeled data
- `GET /rlhf/trained-datasets`: List datasets already used for training
- `GET /rlhf/pending-datasets`: List datasets available for training but not yet trained
- `GET /rlhf/latest-model`: Get latest model version information

## API Request/Response Formats

### Data Ingestion Request (`POST /ingest`)
```json
{
  "timestamp": "2024-12-01T10:30:00",
  "driving_style": "Normal",
  "data": {
    "SPEED": 65.5,
    "RPM": 2500,
    "MAF": 8.2,
    "ENGINE_LOAD": 45.0,
    "THROTTLE_POS": 25.0
  },
  "status": "start|end|null"
}
```

### Latest Predictions Response (`GET /predictions/latest`)
```json
{
  "driver_behavior": ["Normal", "Aggressive", "Conservative"],
  "fuel_efficiency": [85.2, 72.1, 91.5],
  "timestamp": "2024-12-01T14:30:22",
  "driver_behavior_count": 3,
  "fuel_efficiency_count": 3
}
```

### Model Status Response (`GET /models/status`)
```json
{
  "driver_behavior": {
    "status": "ready",
    "model_directory": "/app/models/ul",
    "available_files": ["label_encoder_ul.pkl", "scaler_ul.pkl", "xgb_drivestyle_ul.pkl"],
    "missing_files": [],
    "total_files": 3,
    "loaded_files": 3
  },
  "fuel_efficiency": {
    "status": "ready", 
    "model_directory": "/app/models/efficiency",
    "available_files": ["efficiency_model.joblib"],
    "missing_files": [],
    "total_files": 1,
    "loaded_files": 1
  },
  "overall_status": "ready"
}
```

### Efficiency Retrieval Response (`GET /efficiency/{filename}`)
```bash
curl -s "https://binkhoale1812-obd-logger.hf.space/efficiency/007_2025-10-27_processed.csv"
```
```json
{
  "filename": "007_2025-10-27_processed.csv",
  "efficiency_score": 87.62542148813739,
  "timestamp": "2025-10-27T10:34:00.046241",
  "status": "success"
}
```

### Firebase Storage
- Structured data storage with automatic versioning
- **`skyledge/raw/`**: Original OBD data files
- **`skyledge/processed/`**: Cleaned and processed data
- **`skyledge/processed/efficiency.json`**: Fuel efficiency predictions for each processed file
- **`skyledge/labeled/`**: Human-labeled data for RLHF training
- **`skyledge/labeled/trained.txt`**: Tracks processed datasets to avoid retraining

### Hugging Face Hub
- **Driver Behavior Model Repository**: `BinKhoaLe1812/Driver_Behavior_OBD`
- **Fuel Efficiency Model Repository**: `BinKhoaLe1812/Fuel_Efficiency_OBD`
- **Semantic Versioning**: v1.0, v1.1, v1.2, ..., v2.0, etc.
- **Model Components**: 
  - Driver Behavior: XGBoost model, label encoder, scaler
  - Fuel Efficiency: Joblib model with scaler and calibration
- **Metadata**: Training logs, performance metrics, dataset information

## RLHF Training System

### Overview
The Reinforcement Learning from Human Feedback (RLHF) system enables continuous improvement of the driver behavior classification model through human-labeled data.

### Key Features
- **Original Dataset Tracking**: Automatically links labeled data to original datasets
- **Preference Learning**: Learns from differences between model predictions and human labels
- **Semantic Versioning**: Automatic model versioning (1.0 → 1.1 → 1.2 → 2.0)
- **Hugging Face Integration**: Saves models to HF Hub with metadata
- **Training Tracking**: Prevents retraining on the same datasets

### Usage Examples

#### Trigger RLHF Training
```bash
curl -X POST "https://binkhoale1812-obd-logger.hf.space/rlhf/train" \
     -H "Content-Type: application/json" \
     -d '{
       "max_datasets": 5,
       "force_retrain": false
     }'
```

#### Check Training Status
```bash
curl -X GET "https://binkhoale1812-obd-logger.hf.space/rlhf/status"
```

#### List Trained Datasets
```bash
curl -X GET "https://binkhoale1812-obd-logger.hf.space/rlhf/trained-datasets"
```

#### List Pending Datasets
```bash
curl -X GET "https://binkhoale1812-obd-logger.hf.space/rlhf/pending-datasets"
```

#### Get Latest Model Version
```bash
curl -X GET "https://binkhoale1812-obd-logger.hf.space/rlhf/latest-model"
```

### API Response Examples

#### Training Status Response (`/rlhf/status`)
```json
{
  "status": "available",
  "labeled_datasets_count": 5,
  "datasets": [
    {
      "name": "labeled/dataset1.csv",
      "size": 1024,
      "created": "2024-12-01T10:30:00"
    }
  ],
  "firebase_bucket": "skyledge-36b56.firebasestorage.app",
  "labeled_path": "skyledge/labeled",
  "timestamp": "2024-12-01T14:30:22"
}
```

#### Pending Datasets Response (`/rlhf/pending-datasets`)
```json
{
  "pending_datasets_count": 3,
  "pending_datasets": [
    {
      "name": "labeled/new_dataset1.csv",
      "size": 2048,
      "created": "2024-12-01T11:00:00"
    }
  ],
  "total_available": 8,
  "already_trained": 5,
  "timestamp": "2024-12-01T14:30:22"
}
```

#### Trained Datasets Response (`/rlhf/trained-datasets`)
```json
{
  "trained_datasets_count": 5,
  "trained_datasets": [
    "2024-12-01T10:30:00:labeled/dataset1.csv",
    "2024-12-01T11:15:00:labeled/dataset2.csv",
    "2024-12-01T12:00:00:labeled/dataset3.csv"
  ],
  "timestamp": "2024-12-01T14:30:22"
}
```

#### Latest Model Version Response (`/rlhf/latest-model`)
```json
{
  "status": "available",
  "latest_version": "v1.2",
  "model_repository": "BinKhoaLe1812/Driver_Behavior_OBD",
  "version_format": "semantic (v1.0, v1.1, v2.0, etc.)",
  "timestamp": "2024-12-01T14:30:22"
}
```

### Data Flow
1. **Human Labeling**: Data labeled and stored in `skyledge/labeled/`
2. **Filename Convention**: `001_raw-002_2025-09-19-labelled.csv`
3. **Original Dataset**: Automatically loads `skyledge/raw/002_2025-09-19-raw.csv`
4. **RLHF Training**: Compares model predictions vs human labels
5. **Model Update**: Trains new model with preference learning
6. **Versioning**: Saves as v1.0, v1.1, etc. to Hugging Face Hub

## Documentation

- **MongoDB Setup**: See `MONGODB_SETUP.md` for detailed configuration
- **Google Drive Setup**: See `GOOGLE_DRIVE_SETUP.md` for configuration
- **RLHF Training**: See `train/README.md` for detailed RLHF documentation
- **API Reference**: Interactive docs at `/docs` endpoint
- **Code Structure**: Modular design for easy maintenance

## Development

The codebase follows clean architecture principles:
- **Separation of concerns**: Between storage, processing, API, and ML layers
- **Comprehensive error handling**: Graceful fallbacks for service unavailability
- **Type hints and documentation**: Full type annotations and docstrings
- **Modular design**: Easy to extend and maintain
- **RLHF Integration**: Seamless integration of machine learning with data processing
- **Version control**: Semantic versioning for model artifacts
- **Testing**: Comprehensive test coverage for all components

## Model Management

### Driver Behavior Classification
- **Model Type**: XGBoost Classifier
- **Labels**: Aggressive, Normal, Conservative
- **Features**: OBD sensor data (speed, RPM, throttle, etc.)
- **Training**: RLHF with human feedback integration

### Fuel Efficiency Scoring
- **Model Type**: HistGradientBoostingRegressor/RandomForestRegressor
- **Output**: Drive-level efficiency score (0-100%)
- **Features**: Drive-level aggregated features (duration, distance, acceleration patterns, etc.)
- **Calibration**: Quantile-mapping calibration for accurate scoring

### Model Artifacts
- **Driver Behavior**:
  - XGBoost Model: `xgb_drivestyle_ul.pkl`
  - Label Encoder: `label_encoder_ul.pkl`
  - Feature Scaler: `scaler_ul.pkl`
- **Fuel Efficiency**:
  - ML Model: `efficiency_model.joblib`
  - Metadata: `efficiency_meta.json`

### Versioning Strategy
- **Semantic Versioning**: 1.0 → 1.1 → 1.2 → 2.0
- **Automatic Detection**: Checks existing versions in HF repo
- **Fallback**: Timestamp-based versioning if HF unavailable
- **Local Backup**: Saves to local `/app/models/ul/v{version}/`

## License

Apache 2.0 License
