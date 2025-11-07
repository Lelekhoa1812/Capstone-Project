# saver.py
# Model saving functions for RLHF training
import os
import json
import logging
import pickle
import joblib
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from huggingface_hub import HfApi, Repository
import pandas as pd
import numpy as np

logger = logging.getLogger("rlhf-saver")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(_h)

class ModelSaver:
    """
    Save trained models to Hugging Face Hub and local storage.
    Handles model artifacts, metadata, and versioning.
    """
    
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN environment variable not set")
        
        self.hf_api = HfApi(token=self.hf_token)
        self.repo_id = os.getenv("HF_MODEL_REPO", "BinKhoaLe1812/Driver_Behavior_OBD")
        
        # Local model directory
        self.local_model_dir = Path(os.getenv("MODEL_DIR", "/app/models/ul"))
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 ModelSaver ready | repo={self.repo_id}")
    
    def _get_next_version(self) -> str:
        """Get the next version number (1.0, 1.1, 1.2, ..., 1.9, 2.0, etc.)"""
        try:
            # List existing versions in HF repo
            repo_files = self.hf_api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="model"
            )
            
            # Find version directories (v1.0/, v1.1/, etc.)
            version_dirs = []
            for f in repo_files:
                if f.startswith('v') and '/' in f:
                    # Extract version directory from path like "v1.0/file.txt"
                    version_dir = f.split('/')[0]
                    if version_dir not in version_dirs:
                        version_dirs.append(version_dir)
            
            versions = []
            for v_dir in version_dirs:
                try:
                    version_str = v_dir[1:]  # Remove 'v' prefix
                    if '.' in version_str:
                        major, minor = version_str.split('.')
                        versions.append((int(major), int(minor)))
                except (ValueError, IndexError):
                    continue
            
            if not versions:
                return "1.0"
            
            # Sort versions and get the latest
            versions.sort()
            latest_major, latest_minor = versions[-1]
            
            # Increment version
            if latest_minor < 9:
                return f"{latest_major}.{latest_minor + 1}"
            else:
                return f"{latest_major + 1}.0"
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to get next version from HF repo: {e}")
            # Fallback to timestamp-based version
            return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_model_metadata(self, 
                             model_type: str,
                             training_data_info: Dict[str, Any],
                             performance_metrics: Dict[str, float],
                             model_version: str,
                             rlhf_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create metadata for the trained model"""
        metadata = {
            "model_type": model_type,
            "version": model_version,
            "created_at": datetime.now().isoformat(),
            "training_data": training_data_info,
            "performance_metrics": performance_metrics,
            "framework": "xgboost",
            "task": "driver_behavior_classification",
            "labels": ["aggressive", "normal", "conservative"],  # Based on dbehavior_labeler.py
            "features": "obd_sensor_data",
            "rlhf_metadata": rlhf_metadata or {}
        }
        return metadata
    
    def save_model_locally(self, 
                          model: Any,
                          label_encoder: Any,
                          scaler: Any,
                          model_version: str,
                          metadata: Dict[str, Any]) -> Dict[str, str]:
        """Save model components locally with versioning"""
        try:
            # Create versioned directory
            version_dir = self.local_model_dir / f"v{model_version}"
            version_dir.mkdir(exist_ok=True)
            
            # Save model components
            model_path = version_dir / "xgb_drivestyle_ul.pkl"
            le_path = version_dir / "label_encoder_ul.pkl"
            scaler_path = version_dir / "scaler_ul.pkl"
            metadata_path = version_dir / "metadata.json"
            
            # Save using joblib for better compatibility
            joblib.dump(model, model_path)
            joblib.dump(label_encoder, le_path)
            joblib.dump(scaler, scaler_path)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Also save to the main model directory (for current usage)
            joblib.dump(model, self.local_model_dir / "xgb_drivestyle_ul.pkl")
            joblib.dump(label_encoder, self.local_model_dir / "label_encoder_ul.pkl")
            joblib.dump(scaler, self.local_model_dir / "scaler_ul.pkl")
            
            logger.info(f"✅ Model saved locally to {version_dir}")
            
            return {
                "model_path": str(model_path),
                "label_encoder_path": str(le_path),
                "scaler_path": str(scaler_path),
                "metadata_path": str(metadata_path)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to save model locally: {e}")
            raise
    
    def save_model_to_hf(self, 
                        model: Any,
                        label_encoder: Any,
                        scaler: Any,
                        model_version: str,
                        metadata: Dict[str, Any],
                        training_data_info: Dict[str, Any]) -> str:
        """Save model to Hugging Face Hub"""
        try:
            # Create temporary directory for upload
            temp_dir = Path(f"/tmp/hf_upload_{model_version}")
            temp_dir.mkdir(exist_ok=True)
            
            # Save model components
            model_path = temp_dir / "xgb_drivestyle_ul.pkl"
            le_path = temp_dir / "label_encoder_ul.pkl"
            scaler_path = temp_dir / "scaler_ul.pkl"
            metadata_path = temp_dir / "metadata.json"
            readme_path = temp_dir / "README.md"
            
            # Save using joblib
            joblib.dump(model, model_path)
            joblib.dump(label_encoder, le_path)
            joblib.dump(scaler, scaler_path)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create README
            readme_content = self._create_readme(metadata, training_data_info)
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            # Upload to Hugging Face Hub with versioned directory
            # Create versioned subdirectory in the temp folder
            versioned_temp_dir = temp_dir / f"v{model_version}"
            versioned_temp_dir.mkdir(exist_ok=True)
            
            # Move files to versioned directory
            for file in temp_dir.glob("*"):
                if file.is_file() and not file.name.startswith("."):
                    shutil.move(str(file), str(versioned_temp_dir / file.name))
            
            # Upload the versioned directory
            self.hf_api.upload_folder(
                folder_path=str(versioned_temp_dir),
                repo_id=self.repo_id,
                repo_type="model",
                path_in_repo=f"v{model_version}",
                commit_message=f"RLHF training update v{model_version}",
                ignore_patterns=["*.tmp", "*.log"]
            )
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"✅ Model uploaded to Hugging Face Hub: {self.repo_id}")
            return f"https://huggingface.co/{self.repo_id}"
            
        except Exception as e:
            logger.error(f"❌ Failed to save model to HF: {e}")
            raise
    
    def _create_readme(self, metadata: Dict[str, Any], training_data_info: Dict[str, Any]) -> str:
        """Create README content for the model"""
        readme = f"""---
license: mit
tags:
- driver-behavior
- obd-data
- xgboost
- rlhf
- reinforcement-learning
---

# Driver Behavior Classification Model (RLHF v{metadata['version']})

This model classifies driver behavior based on OBD (On-Board Diagnostics) sensor data using XGBoost.

## Model Information

- **Model Type**: {metadata['model_type']}
- **Version**: {metadata['version']}
- **Created**: {metadata['created_at']}
- **Framework**: {metadata['framework']}
- **Task**: {metadata['task']}

## Performance Metrics

"""
        
        for metric, value in metadata['performance_metrics'].items():
            if isinstance(value, (list, tuple)):
                readme += f"- **{metric}**: {value}\n"
            else:
                readme += f"- **{metric}**: {value:.4f}\n"
        
        readme += f"""
## Training Data

- **Datasets Used**: {len(training_data_info.get('datasets', []))}
- **Total Samples**: {training_data_info.get('total_samples', 'N/A')}
- **Training Date**: {training_data_info.get('training_date', 'N/A')}

## Labels

The model predicts one of the following driver behavior categories:
"""
        
        for label in metadata['labels']:
            readme += f"- {label}\n"
        
        readme += """
## Usage

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('xgb_drivestyle_ul.pkl')
label_encoder = joblib.load('label_encoder_ul.pkl')
scaler = joblib.load('scaler_ul.pkl')

# Prepare your OBD data
# (Ensure features match the training data format)

# Make predictions
predictions = model.predict(scaled_data)
behavior_labels = label_encoder.inverse_transform(predictions)
```

## Files

- `xgb_drivestyle_ul.pkl`: Main XGBoost model
- `label_encoder_ul.pkl`: Label encoder for behavior categories
- `scaler_ul.pkl`: Feature scaler
- `metadata.json`: Model metadata and performance metrics

## RLHF Training

This model was trained using Reinforcement Learning from Human Feedback (RLHF) to improve performance based on human-labeled data and feedback.
"""
        
        return readme
    
    def save_training_log(self, 
                         training_log: Dict[str, Any],
                         model_version: str) -> str:
        """Save training log to Firebase storage"""
        try:
            # Import Firebase client
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from data.firebase_saver import FirebaseSaver
            
            # Create log entry
            log_entry = {
                "version": model_version,
                "timestamp": datetime.now().isoformat(),
                "log": training_log
            }
            
            # Save to Firebase
            saver = FirebaseSaver()
            # Note: We'll need to modify FirebaseSaver to support different prefixes
            # For now, we'll save to a logs subdirectory
            log_filename = f"training_log_v{model_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Create temporary file
            temp_path = f"/tmp/{log_filename}"
            with open(temp_path, 'w') as f:
                json.dump(log_entry, f, indent=2)
            
            # Upload to Firebase (we'll need to extend FirebaseSaver for this)
            # For now, just log locally
            logger.info(f"📝 Training log saved: v{model_version} with {len(training_log.get('datasets_used', []))} datasets")
            
            return temp_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save training log: {e}")
            return ""
    
    def save_complete_model(self,
                           model: Any,
                           label_encoder: Any,
                           scaler: Any,
                           model_version: str,
                           training_data_info: Dict[str, Any],
                           performance_metrics: Dict[str, float],
                           training_log: Dict[str, Any],
                           rlhf_metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """Complete model saving workflow"""
        try:
            # Create metadata
            metadata = self._create_model_metadata(
                model_type="xgboost_classifier",
                training_data_info=training_data_info,
                performance_metrics=performance_metrics,
                model_version=model_version,
                rlhf_metadata=rlhf_metadata
            )
            
            # Save locally
            local_paths = self.save_model_locally(
                model, label_encoder, scaler, model_version, metadata
            )
            
            # Save to Hugging Face Hub
            hf_url = self.save_model_to_hf(
                model, label_encoder, scaler, model_version, metadata, training_data_info
            )
            
            # Save training log
            log_path = self.save_training_log(training_log, model_version)
            
            result = {
                "local_paths": local_paths,
                "hf_url": hf_url,
                "log_path": log_path,
                "version": model_version,
                "metadata": metadata
            }
            
            logger.info(f"✅ Complete model save successful: v{model_version}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Complete model save failed: {e}")
            raise


def main():
    """Test the saver functionality"""
    try:
        saver = ModelSaver()
        print(f"ModelSaver initialized for repo: {saver.repo_id}")
        print(f"Local model directory: {saver.local_model_dir}")
    except Exception as e:
        print(f"Failed to initialize ModelSaver: {e}")


if __name__ == "__main__":
    main()
