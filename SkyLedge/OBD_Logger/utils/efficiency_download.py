"""
Fuel Efficiency Model Downloader
Downloads the latest fuel efficiency model from Hugging Face Hub
Similar to utils/download.py but for fuel efficiency models
"""

import os
import pathlib
import logging
from typing import Optional, List
from huggingface_hub import HfApi, hf_hub_download
import joblib

logger = logging.getLogger("efficiency-downloader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)

def load_env_file():
    """Load .env file if it exists"""
    env_path = pathlib.Path(".env")
    if env_path.exists():
        logger.info("📄 Loading .env file...")
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        return True
    return False

# Load .env file first before setting any environment variables
load_env_file()

# Configuration
EFFICIENCY_REPO_ID = os.getenv("HF_EFFICIENCY_MODEL_REPO", "BinKhoaLe1812/Fuel_Efficiency_OBD")
EFFICIENCY_MODEL_DIR = pathlib.Path(os.getenv("EFFICIENCY_MODEL_DIR", "/home/user/app/models/efficiency")).resolve()
EFFICIENCY_FILES = ["efficiency_model.joblib", "efficiency_meta.json"]

EFFICIENCY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def get_latest_efficiency_version():
    """Get the latest fuel efficiency model version from Hugging Face repo"""
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("⚠️ HF_TOKEN not set, using default efficiency model files")
            return None
        
        api = HfApi(token=hf_token)
        repo_files = api.list_repo_files(
            repo_id=EFFICIENCY_REPO_ID,
            repo_type="model"
        )
        
        logger.info(f"🔍 Checking efficiency repository files...")
        logger.info(f"📁 Found {len(repo_files)} files in efficiency repository")
        
        # Find version directories (v1.0, v1.1, etc.)
        version_dirs = [f for f in repo_files if f.startswith('v') and '/' not in f]
        logger.info(f"📦 Found efficiency version directories: {version_dirs}")
        
        # Also check for version directories with files inside
        version_dirs_with_files = []
        for f in repo_files:
            if f.startswith('v') and '/' in f:
                version_dir = f.split('/')[0]
                if version_dir not in version_dirs_with_files:
                    version_dirs_with_files.append(version_dir)
        
        if version_dirs_with_files:
            logger.info(f"📦 Found efficiency version directories with files: {version_dirs_with_files}")
            version_dirs.extend(version_dirs_with_files)
        
        versions = []
        
        for v_dir in version_dirs:
            try:
                # Extract version number (e.g., "v1.0" -> 1.0)
                version_str = v_dir[1:]  # Remove 'v' prefix
                major, minor = map(int, version_str.split('.'))
                versions.append((major, minor, v_dir))
            except ValueError:
                logger.warning(f"⚠️ Could not parse version: {v_dir}")
                continue
        
        if not versions:
            logger.warning("⚠️ No valid efficiency versions found")
            return None
        
        # Sort by major.minor version
        versions.sort(key=lambda x: (x[0], x[1]))
        latest_version = versions[-1][2]  # Get the version string
        
        logger.info(f"✅ Latest efficiency model version: {latest_version}")
        return latest_version
        
    except Exception as e:
        logger.error(f"❌ Error getting latest efficiency version: {e}")
        return None

def download_efficiency_model(version: Optional[str] = None) -> bool:
    """Download the specified version of the fuel efficiency model"""
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("❌ HF_TOKEN not set")
            return False
        
        if version is None:
            version = get_latest_efficiency_version()
            if version is None:
                logger.error("❌ Could not determine latest efficiency version")
                return False
        
        logger.info(f"📥 Downloading efficiency model version: {version}")
        logger.info(f"📁 Target directory: {EFFICIENCY_MODEL_DIR}")
        
        # Download each required file
        for filename in EFFICIENCY_FILES:
            try:
                logger.info(f"📥 Downloading {filename}...")
                file_path = hf_hub_download(
                    repo_id=EFFICIENCY_REPO_ID,
                    filename=f"{version}/{filename}",
                    token=hf_token,
                    local_dir=EFFICIENCY_MODEL_DIR,
                    local_dir_use_symlinks=False
                )
                logger.info(f"✅ Downloaded: {filename} to {file_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {filename}: {e}")
                return False
        
        logger.info(f"✅ Efficiency model {version} downloaded successfully")
        logger.info(f"📂 Final directory contents: {list(EFFICIENCY_MODEL_DIR.iterdir())}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading efficiency model: {e}")
        return False

def download_latest_efficiency_models() -> bool:
    """Download the latest fuel efficiency model files"""
    try:
        logger.info("🚀 Starting efficiency model download...")
        
        # Get latest version
        latest_version = get_latest_efficiency_version()
        if latest_version is None:
            logger.error("❌ Could not determine latest efficiency version")
            return False
        
        # Download the model
        success = download_efficiency_model(latest_version)
        if success:
            logger.info("✅ Latest efficiency model downloaded successfully")
        else:
            logger.error("❌ Failed to download latest efficiency model")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error in download_latest_efficiency_models: {e}")
        return False

def load_efficiency_model():
    """Load the efficiency model from local storage"""
    try:
        # Check if we have a versioned model (v1.0 subdirectory)
        versioned_model_path = EFFICIENCY_MODEL_DIR / "v1.0" / "efficiency_model.joblib"
        versioned_meta_path = EFFICIENCY_MODEL_DIR / "v1.0" / "efficiency_meta.json"
        
        # Check if we have a non-versioned model (direct in directory)
        model_path = EFFICIENCY_MODEL_DIR / "efficiency_model.joblib"
        meta_path = EFFICIENCY_MODEL_DIR / "efficiency_meta.json"
        
        # Try versioned path first, then fallback to non-versioned
        if versioned_model_path.exists():
            logger.info(f"📁 Loading versioned model from: {versioned_model_path}")
            model_artifacts = joblib.load(versioned_model_path)
            metadata = None
            if versioned_meta_path.exists():
                import json
                with open(versioned_meta_path, 'r') as f:
                    metadata = json.load(f)
        elif model_path.exists():
            logger.info(f"📁 Loading non-versioned model from: {model_path}")
            model_artifacts = joblib.load(model_path)
            metadata = None
            if meta_path.exists():
                import json
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
        else:
            logger.error(f"❌ Efficiency model not found at {model_path} or {versioned_model_path}")
            return None, None
        
        logger.info("✅ Efficiency model loaded successfully")
        if metadata:
            logger.info(f"📊 Model kind: {metadata.get('model_kind', 'unknown')}")
            logger.info(f"📊 Model version: {metadata.get('version', 'unknown')}")
        return model_artifacts, metadata
        
    except Exception as e:
        logger.error(f"❌ Error loading efficiency model: {e}")
        return None, None

def check_efficiency_model_exists() -> bool:
    """Check if efficiency model files exist locally"""
    # Check both versioned and non-versioned paths
    versioned_model_path = EFFICIENCY_MODEL_DIR / "v1.0" / "efficiency_model.joblib"
    model_path = EFFICIENCY_MODEL_DIR / "efficiency_model.joblib"
    
    logger.info(f"🔍 Checking for efficiency model at: {model_path}")
    logger.info(f"🔍 Checking for versioned efficiency model at: {versioned_model_path}")
    logger.info(f"📁 Model directory exists: {EFFICIENCY_MODEL_DIR.exists()}")
    
    if EFFICIENCY_MODEL_DIR.exists():
        logger.info(f"📂 Files in model directory: {list(EFFICIENCY_MODEL_DIR.iterdir())}")
        if (EFFICIENCY_MODEL_DIR / "v1.0").exists():
            logger.info(f"📂 Files in v1.0 directory: {list((EFFICIENCY_MODEL_DIR / 'v1.0').iterdir())}")
    
    exists = versioned_model_path.exists() or model_path.exists()
    logger.info(f"✅ Model file exists: {exists}")
    return exists

if __name__ == "__main__":
    # Test the download functionality
    success = download_latest_efficiency_models()
    if success:
        print("✅ Efficiency model download test successful")
    else:
        print("❌ Efficiency model download test failed")
