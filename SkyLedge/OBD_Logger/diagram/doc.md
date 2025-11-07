title OBD-II Data Logging & RLHF Training System Architecture

// Color scheme based on component roles
// Data Sources: yellow | UI|Frontend: lightblue | Backend|API: amber
// ML|AI Models: green | Storage Systems: purple | External Services: gray
// RLHF Training: red | Data Processing: teal | Security: rose

// 0. Data Sources & Collection
Data Sources [color: yellow, icon: database] {
  OBD-II Vehicle Sensors [icon: car, color: yellow] {
    RPM | Throttle Position | Speed [icon: gauge, color: yellow]
    Engine Load | Coolant Temp | Fuel Pressure [icon: wrench, color: yellow]
    MAF | Intake Pressure | O2 Sensors [icon: activity, color: yellow]
  }
  Real-time Data Stream [icon: wifi, color: yellow] {
    High-frequency PIDs [icon: zap, color: yellow]
    Low-frequency PIDs [icon: clock, color: yellow]
    Session Management [icon: play-circle, color: yellow]
  }
}

// 1. NextJS Frontend Interface
NextJS Frontend [color: lightblue, icon: monitor] {
  Web Dashboard [icon: layout, color: lightblue] {
    Real-time Monitoring [icon: activity, color: lightblue]
    Data Visualization [icon: bar-chart, color: lightblue]
    File Upload Interface [icon: upload-cloud, color: lightblue]
  }
  User Interactions [icon: users, color: lightblue] {
    Raw Data Upload [icon: file-text, color: lightblue]
    Labeled Data Upload [icon: tag, color: lightblue]
    Processing Triggers [icon: play, color: lightblue]
    RLHF Training Triggers [icon: brain, color: lightblue]
  }
}

// 2. FastAPI Backend Core
FastAPI Backend [color: amber, icon: server] {
  Data Processing Pipeline [icon: workflow, color: amber] {
    Data Ingestion [icon: download, color: amber]
    Data Cleaning [icon: filter, color: amber]
    Gap Detection & KNN Imputation [icon: search, color: amber]
    Outlier Handling [icon: alert-triangle, color: amber]
  }
  ML Model Integration [icon: brain, color: amber] {
    Driver Behavior Classification [icon: user-check, color: amber]
    XGBoost Model Prediction [icon: target, color: amber]
    Feature Engineering [icon: cog, color: amber]
  }
  API Endpoints [icon: api, color: amber] {
    Data Upload & Processing [icon: upload, color: amber]
    Real-time Streaming [icon: wifi, color: amber]
    Model Management [icon: settings, color: amber]
  }
}

// 3. RLHF Training System
RLHF Training System [color: red, icon: brain] {
  Data Loader [icon: download, color: red] {
    Labeled Data Loading [icon: tag, color: red]
    Original Dataset Tracking [icon: file-text, color: red]
    Version Management [icon: git-branch, color: red]
  }
  Training Pipeline [icon: cpu, color: red] {
    Model Retraining [icon: refresh-cw, color: red]
    Performance Evaluation [icon: bar-chart, color: red]
    Version Control [icon: git-commit, color: red]
  }
  Model Deployment [icon: rocket, color: red] {
    Hugging Face Integration [icon: share, color: red]
    Semantic Versioning [icon: hash, color: red]
    Model Metadata [icon: info, color: red]
  }
}

// 4. Storage Systems
Storage Systems [color: purple, icon: database] {
  Firebase Storage [icon: cloud, color: purple] {
    Raw Data Storage [icon: file-text, color: purple] {
      skyledge|raw [icon: folder, color: purple]
    }
    Processed Data Storage [icon: check-circle, color: purple] {
      skyledge|processed [icon: folder, color: purple]
    }
    Labeled Data Storage [icon: tag, color: purple] {
      skyledge|labeled [icon: folder, color: purple]
    }
    Training Tracking [icon: list, color: purple] {
      trained.txt [icon: file-text, color: purple]
    }
  }
  Google Drive Integration [icon: drive, color: purple] {
    CSV File Storage [icon: file-text, color: purple]
    Folder Management [icon: folder, color: purple]
    Backup & Archive [icon: archive, color: purple]
  }
  MongoDB Atlas [icon: database, color: purple] {
    Structured Data Storage [icon: table, color: purple]
    Session Management [icon: clock, color: purple]
    Query Interface [icon: search, color: purple]
  }
}

// 5. External Services
External Services [color: gray, icon: cloud] {
  Driver Behavior [icon: share, color: gray] {
    Model Repository [icon: github, color: gray]
    Version Control [icon: git-branch, color: gray]
    Model Distribution [icon: download, color: gray]
  }
  GCC [icon: cloud, color: gray] {
    Firebase Authentication [icon: key, color: gray]
    Google Drive API [icon: drive, color: gray]
    Cloud Storage [icon: database, color: gray]
  }
}

// 6. Data Processing & Analysis
Data Processing [color: teal, icon: cpu] {
  Data Cleaning Pipeline [icon: filter, color: teal] {
    Gap Detection [icon: search, color: teal]
    KNN Imputation [icon: calculator, color: teal]
    Outlier Detection [icon: alert-triangle, color: teal]
    Data Validation [icon: check-circle, color: teal]
  }
  Feature Engineering [icon: cog, color: teal] {
    Rolling Averages [icon: bar-chart, color: teal]
    Rate of Change [icon: trend-up, color: teal]
    Statistical Features [icon: calculator, color: teal]
  }
  Visualization [icon: image, color: teal] {
    Correlation Heatmaps [icon: grid, color: teal]
    Trend Analysis [icon: line-chart, color: teal]
    Performance Metrics [icon: gauge, color: teal]
  }
}

// 7. Security & Authentication
Security & Auth [color: rose, icon: shield] {
  API Authentication [icon: key, color: rose]
  Data Encryption [icon: lock, color: rose]
  Access Control [icon: user-check, color: rose]
  Audit Logging [icon: list, color: rose]
}

// Main Data Flow (End-to-End.

// Data Collection Flow
OBD-II Vehicle Sensors > Real-time Data Stream
Real-time Data Stream > Data Ingestion
Data Ingestion > Data Cleaning Pipeline

// Frontend Interactions
NextJS Frontend > Web Dashboard
Web Dashboard > User Interactions
User Interactions > Raw Data Upload
User Interactions > Labeled Data Upload
User Interactions > Processing Triggers
User Interactions > RLHF Training Triggers

// Data Processing Flow
Raw Data Upload > Data Ingestion
Data Ingestion > Data Cleaning Pipeline
Data Cleaning Pipeline > Gap Detection
Gap Detection > KNN Imputation
KNN Imputation > Outlier Detection
Outlier Detection > Data Validation
Data Validation > Feature Engineering

// Storage Operations
Data Validation > skyledge|raw
Feature Engineering > skyledge|processed
Labeled Data Upload > skyledge|labeled
Feature Engineering > Google Drive Integration
Feature Engineering > MongoDB Atlas

// ML Model Operations
Feature Engineering > Driver Behavior Classification
Driver Behavior Classification > XGBoost Model Prediction
XGBoost Model Prediction > Model Management

// RLHF Training Flow
Processing Triggers > RLHF Training System
RLHF Training System > Data Loader
Data Loader > Labeled Data Loading
Labeled Data Loading > Original Dataset Tracking
Original Dataset Tracking > Training Pipeline
Training Pipeline > Model Retraining
Model Retraining > Performance Evaluation
Performance Evaluation > Model Deployment
Model Deployment > Driver Behavior

// External Service Integration
Model Deployment > Model Repository
Model Deployment > Version Control
Firebase Storage > Firebase Authentication
Google Drive Integration > Google Drive API
MongoDB Atlas > Cloud Storage

// Data Visualization & Monitoring
Performance Evaluation > Visualization
Visualization > Correlation Heatmaps
Visualization > Trend Analysis
Visualization > Performance Metrics
Web Dashboard > Real-time Monitoring
Real-time Monitoring > Data Visualization

// Security & Access Control
API Endpoints > API Authentication
Firebase Storage > Data Encryption
Google Drive Integration > Access Control
MongoDB Atlas > Audit Logging

// Model Versioning & Deployment
Model Deployment > Model Distribution
Version Control > Model Management
Model Management > Driver Behavior Classification

// Continuous Learning Loop
Performance Metrics > RLHF Training System
RLHF Training System > Model Retraining
Model Retraining > Performance Evaluation
Performance Evaluation > Model Deployment
Model Deployment > Model Repository
 Model Distribution > Driver Behavior Classification
