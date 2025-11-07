#!/usr/bin/env python3
"""
Simple script to reorganize existing models in HF repo to versioned structure.
This will move the current 3 .pkl files from root to v1.0 folder.
"""

import os
import tempfile
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, upload_file

def load_env():
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print("⚠️ No .env file found")

def main():
    """Main function to reorganize models"""
    print("🔄 Reorganizing models in Hugging Face repository...")
    print("=" * 60)
    
    # Load environment variables
    load_env()
    
    # Check if HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in environment")
        return 1
    
    print(f"✅ HF_TOKEN loaded: {hf_token[:10]}...")
    
    # Configuration
    repo_id = "BinKhoaLe1812/Driver_Behavior_OBD"
    model_files = ["label_encoder_ul.pkl", "scaler_ul.pkl", "xgb_drivestyle_ul.pkl"]
    
    print(f"📦 Target repository: {repo_id}")
    print(f"📁 Model files to move: {model_files}")
    
    # Initialize HF API
    hf_api = HfApi(token=hf_token)
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            print(f"📁 Using temporary directory: {temp_path}")
            
            # Download existing model files
            downloaded_files = []
            for file in model_files:
                try:
                    print(f"📥 Downloading {file}...")
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file,
                        repo_type="model",
                        token=hf_token
                    )
                    downloaded_files.append((file, local_path))
                    print(f"✅ Downloaded {file}")
                except Exception as e:
                    print(f"⚠️ Could not download {file}: {e}")
            
            if not downloaded_files:
                print("⚠️ No model files found to move")
                return 1
            
            # Create v1.0 directory structure
            v1_dir = temp_path / "v1.0"
            v1_dir.mkdir(exist_ok=True)
            print(f"📁 Created v1.0 directory: {v1_dir}")
            
            # Copy files to v1.0 directory
            for filename, local_path in downloaded_files:
                dest_path = v1_dir / filename
                import shutil
                shutil.copy2(local_path, dest_path)
                print(f"📦 Prepared {filename} for v1.0/")
            
            # Create metadata.json for v1.0
            metadata = {
                "version": "1.0",
                "model_type": "xgboost_classifier",
                "created_at": "2024-12-01T00:00:00",
                "description": "Initial model version - moved from root directory",
                "framework": "xgboost",
                "task": "driver_behavior_classification",
                "labels": ["aggressive", "normal", "conservative"],
                "features": "obd_sensor_data",
                "files": [f[0] for f in downloaded_files]
            }
            
            metadata_path = v1_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print("📝 Created metadata.json for v1.0")
            
            # Create README.md for v1.0
            readme_content = """---
license: mit
tags:
- driver-behavior
- obd-data
- xgboost
- version-1.0
---

# Driver Behavior Classification Model v1.0

Initial version of the driver behavior classification model.

## Files

- `xgb_drivestyle_ul.pkl`: Main XGBoost model
- `label_encoder_ul.pkl`: Label encoder for behavior categories  
- `scaler_ul.pkl`: Feature scaler
- `metadata.json`: Model metadata

## Usage

```python
import joblib

# Load the model
model = joblib.load('xgb_drivestyle_ul.pkl')
label_encoder = joblib.load('label_encoder_ul.pkl')
scaler = joblib.load('scaler_ul.pkl')

# Make predictions
predictions = model.predict(scaled_data)
behavior_labels = label_encoder.inverse_transform(predictions)
```
"""
            
            readme_path = v1_dir / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            print("📖 Created README.md for v1.0")
            
            # Upload files to v1.0 directory in HF repo
            print("🚀 Uploading files to Hugging Face Hub...")
            for file_path in v1_dir.iterdir():
                if file_path.is_file():
                    hf_filename = f"v1.0/{file_path.name}"
                    print(f"📤 Uploading {file_path.name} to {hf_filename}...")
                    upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=hf_filename,
                        repo_id=repo_id,
                        repo_type="model",
                        token=hf_token,
                        commit_message=f"Add {file_path.name} to v1.0 directory"
                    )
                    print(f"✅ Uploaded {file_path.name} to v1.0/")
            
            print("\n✅ Successfully moved models to v1.0 structure!")
            print(f"📁 Models now located at: {repo_id}/v1.0/")
            print("\nNext steps:")
            print("1. Verify the models are in the v1.0 folder on Hugging Face")
            print("2. Test the RLHF training with: curl -X POST 'http://localhost:8000/rlhf/train'")
            
            return 0
            
    except Exception as e:
        print(f"❌ Reorganization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure HF_TOKEN is set correctly")
        print("2. Check that you have write access to the repository")
        print("3. Verify the repository name is correct")
        return 1

if __name__ == "__main__":
    exit(main())
