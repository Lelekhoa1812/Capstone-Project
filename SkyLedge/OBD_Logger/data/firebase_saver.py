# firebase_saver.py
import os
import io
import re
import json
import logging
from datetime import datetime
from typing import Optional, Tuple, List

import pandas as pd

logger = logging.getLogger("firebase-saver")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(_h)

# ---------- Constants (fixed as requested) ----------
FIXED_BUCKET = "skyledge-36b56.firebasestorage.app"
FIXED_PREFIX = "skyledge/processed"  # no trailing slash

# Pattern: NNN_YYYY-MM-DD_processed.csv
FILENAME_RE = re.compile(r"^(?P<num>\d{3})_(?P<date>\d{4}-\d{2}-\d{2})_processed\.csv$")


def _parse_gs_uri(uri: Optional[str]):
    if not uri or not uri.startswith("gs://"):
        return None, None
    path = uri[len("gs://"):]
    parts = path.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def _maybe_default_firebase_bucket(name: Optional[str]) -> Optional[str]:
    # If user passed a project ID (no dot), convert to <project>.appspot.com
    if name and "." not in name:
        return f"{name}.appspot.com"
    return name


# -------------------- Low-level clients --------------------

class _AdminClient:
    """Firebase Admin SDK storage client."""
    def __init__(self, bucket: str):
        import firebase_admin
        from firebase_admin import credentials, storage as fb_storage

        raw = os.getenv("FIREBASE_ADMIN_JSON")
        if not raw:
            raise RuntimeError("FIREBASE_ADMIN_JSON not set")
        info = json.loads(raw)
        client_email = info.get("client_email")
        cred = credentials.Certificate(info)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"storageBucket": bucket})

        # fb_storage.bucket returns a google.cloud.storage.bucket.Bucket
        self.bucket = fb_storage.bucket(bucket)
        self._bucket_name = bucket
        logger.info(f"✅ Firebase Admin initialized | bucket={bucket} as {client_email}")

    # Uploads
    def upload_from_filename(self, local_path: str, dest_path: str, content_type: str):
        blob = self.bucket.blob(dest_path)
        blob.cache_control = "no-store"
        blob.upload_from_filename(local_path, content_type=content_type)

    def upload_from_bytes(self, data: bytes, dest_path: str, content_type: str):
        blob = self.bucket.blob(dest_path)
        blob.cache_control = "no-store"
        blob.upload_from_string(data, content_type=content_type)

    # Listing (needs storage.objects.list permission)
    def list_names(self, prefix: str) -> List[str]:
        # Bucket.list_blobs works via the underlying GCS client
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [b.name for b in blobs]

    # Existence check (for collision-safe retry)
    def blob_exists(self, path: str) -> bool:
        blob = self.bucket.blob(path)
        return blob.exists()


class _GCSClient:
    """google-cloud-storage client."""
    def __init__(self, bucket: str):
        from google.cloud import storage
        from google.oauth2 import service_account

        raw = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not raw:
            raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON not set")
        info = json.loads(raw)
        client_email = info.get("client_email")
        creds = service_account.Credentials.from_service_account_info(info)
        project_id = info.get("project_id")

        self.client = storage.Client(credentials=creds, project=project_id)
        self.bucket = self.client.bucket(bucket)
        self._bucket_name = bucket
        logger.info(f"✅ GCS client initialized | bucket={bucket} as {client_email}")

    def upload_from_filename(self, local_path: str, dest_path: str, content_type: str):
        blob = self.bucket.blob(dest_path)
        blob.cache_control = "no-store"
        blob.upload_from_filename(local_path, content_type=content_type)

    def upload_from_bytes(self, data: bytes, dest_path: str, content_type: str):
        blob = self.bucket.blob(dest_path)
        blob.cache_control = "no-store"
        blob.upload_from_string(data, content_type=content_type)

    def list_names(self, prefix: str) -> List[str]:
        blobs = self.client.list_blobs(self._bucket_name, prefix=prefix)
        return [b.name for b in blobs]

    def blob_exists(self, path: str) -> bool:
        blob = self.bucket.blob(path)
        return blob.exists(self.client)


# -------------------- Saver (high level) --------------------

class FirebaseSaver:
    """
    Fixed target:
      Bucket: skyledge-36b56.firebasestorage.app
      Prefix: skyledge/processed
    Filename convention: NNN_YYYY-MM-DD_processed.csv (NNN is 001-based, zero-padded).
    Auto-increments by listing current objects and picking max+1.
    """

    def __init__(self):
        # Force fixed location regardless of env (as requested)
        bucket_name = FIXED_BUCKET
        self.prefix = FIXED_PREFIX

        # Try Admin SDK first; fallback to GCS client
        self.client = None
        self.mode = None
        try:
            if os.getenv("FIREBASE_ADMIN_JSON"):
                self.client = _AdminClient(bucket_name)
                self.mode = "admin"
        except Exception as e:
            logger.warning(f"⚠️ Admin SDK init failed: {e}")

        if self.client is None:
            try:
                self.client = _GCSClient(bucket_name)
                self.mode = "gcs"
            except Exception as e:
                logger.error(f"❌ GCS client init failed: {e}")
                raise

        logger.info(f"📦 FirebaseSaver ready | mode={self.mode} bucket={bucket_name} prefix={self.prefix}")

    def is_available(self) -> bool:
        return self.client is not None

    # ---------- Incremental naming helpers ----------

    def _list_existing_filenames(self) -> List[str]:
        """List object names under the fixed prefix, return just basenames under that folder."""
        names = self.client.list_names(prefix=self.prefix + "/")
        # keep only items immediately under prefix (not subfolders) & matching our filename pattern
        base_names = []
        for full in names:
            # full looks like 'skyledge/processed/NNN_YYYY-MM-DD_processed.csv'
            if not full.startswith(self.prefix + "/"):
                continue
            base = full[len(self.prefix) + 1:]  # strip 'prefix/'
            if "/" in base:
                # skip nested items (none expected)
                continue
            if FILENAME_RE.match(base):
                base_names.append(base)
        return base_names

    def _max_existing_id(self) -> int:
        """Return max NNN found under prefix, or 0 if none."""
        try:
            base_names = self._list_existing_filenames()
        except Exception as e:
            logger.warning(f"⚠️ Unable to list existing objects; defaulting max_id=0: {e}")
            return 0

        max_id = 0
        for name in base_names:
            m = FILENAME_RE.match(name)
            if not m:
                continue
            try:
                num = int(m.group("num"))
                if num > max_id:
                    max_id = num
            except ValueError:
                continue
        return max_id

    @staticmethod
    def _format_id(n: int) -> str:
        return f"{n:03d}"

    @staticmethod
    def _today_au() -> str:
        # Use Australia/Melbourne local date; if zoneinfo unavailable, fall back to UTC date.
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.now(ZoneInfo("Australia/Melbourne"))
        except Exception:
            dt = datetime.utcnow()
        return dt.strftime("%Y-%m-%d")

    def _build_filename(self, n_int: int, date_str: Optional[str] = None) -> str:
        date_val = (date_str or self._today_au())
        return f"{self._format_id(n_int)}_{date_val}_processed.csv"

    def _dest_path(self, filename: str) -> str:
        return f"{self.prefix}/{filename}"

    def _next_available_name(self, date_str: Optional[str] = None, max_retries: int = 5) -> Tuple[str, str]:
        """
        Compute the next file name by listing existing ones and incrementing.
        Includes a collision check (exists) and retries if necessary.
        Returns: (filename, full_gcs_path)
        """
        start = self._max_existing_id() + 1
        n = start
        for _ in range(max_retries):
            candidate = self._build_filename(n, date_str=date_str)
            dest_path = self._dest_path(candidate)
            # collision check
            if not self.client.blob_exists(dest_path):
                return candidate, dest_path
            n += 1

        # As a final fallback, return the last tried (very unlikely to collide repeatedly)
        candidate = self._build_filename(n, date_str=date_str)
        return candidate, self._dest_path(candidate)

    # ---------- Public save methods (incremental) ----------

    def upload_file_with_increment(
        self,
        local_path: str,
        date_str: Optional[str] = None,
        content_type: str = "text/csv",
    ) -> str:
        """
        Upload a local file using the next incremental name.
        Returns the gs:// URL of the uploaded object (string) or "" on failure.
        """
        if not self.is_available():
            logger.warning("⚠️ Firebase saver unavailable")
            return ""
        try:
            filename, dest_path = self._next_available_name(date_str=date_str)
            self.client.upload_from_filename(local_path, dest_path, content_type)
            logger.info(f"✅ Uploaded file to gs://{FIXED_BUCKET}/{dest_path}")
            return f"gs://{FIXED_BUCKET}/{dest_path}"
        except Exception as e:
            logger.error(f"❌ Firebase upload failed: {e}")
            return ""

    def upload_dataframe_with_increment(
        self,
        df: pd.DataFrame,
        date_str: Optional[str] = None,
        content_type: str = "text/csv",
    ) -> str:
        """
        Upload a DataFrame (as CSV) using the next incremental name.
        Returns the gs:// URL of the uploaded object (string) or "" on failure.
        """
        if not self.is_available():
            logger.warning("⚠️ Firebase saver unavailable")
            return ""
        try:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            data = buf.getvalue().encode("utf-8")

            filename, dest_path = self._next_available_name(date_str=date_str)
            self.client.upload_from_bytes(data, dest_path, content_type)
            logger.info(f"✅ Uploaded DataFrame to gs://{FIXED_BUCKET}/{dest_path}")
            return f"gs://{FIXED_BUCKET}/{dest_path}"
        except Exception as e:
            logger.error(f"❌ Firebase DF upload failed: {e}")
            return ""
    
    def save_efficiency_data(self, filename: str, efficiency_score: float) -> bool:
        """
        Save efficiency data to efficiency.json file in Firebase.
        
        Args:
            filename: The processed filename (e.g., "001_2024-12-01_processed.csv")
            efficiency_score: The fuel efficiency score (0-100)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing efficiency data
            efficiency_data = self.load_efficiency_data()
            
            # Add new entry
            efficiency_data[filename] = {
                "efficiency_score": efficiency_score,
                "timestamp": datetime.now().isoformat(),
                "filename": filename
            }
            
            # Convert to JSON
            json_data = json.dumps(efficiency_data, indent=2)
            
            # Upload to Firebase
            dest_path = f"{FIXED_PREFIX}/efficiency.json"
            self.client.upload_from_bytes(
                json_data.encode("utf-8"), 
                dest_path, 
                "application/json"
            )
            
            logger.info(f"✅ Efficiency data saved for {filename}: {efficiency_score}%")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save efficiency data: {e}")
            return False
    
    def load_efficiency_data(self) -> dict:
        """
        Load efficiency data from efficiency.json file in Firebase.
        
        Returns:
            dict: Efficiency data or empty dict if file doesn't exist
        """
        try:
            dest_path = f"{FIXED_PREFIX}/efficiency.json"
            
            # Try to download the file
            blob = self.client.bucket.blob(dest_path)
            if not blob.exists():
                logger.info("📄 efficiency.json not found, returning empty data")
                return {}
            
            # Download and parse JSON
            json_data = blob.download_as_text()
            efficiency_data = json.loads(json_data)
            
            logger.info(f"✅ Loaded efficiency data: {len(efficiency_data)} entries")
            return efficiency_data
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load efficiency data: {e}")
            return {}
    
    def get_efficiency_by_filename(self, filename: str) -> Optional[dict]:
        """
        Get efficiency data for a specific filename.
        
        Args:
            filename: The processed filename
            
        Returns:
            dict: Efficiency data or None if not found
        """
        try:
            efficiency_data = self.load_efficiency_data()
            return efficiency_data.get(filename)
        except Exception as e:
            logger.error(f"❌ Failed to get efficiency data for {filename}: {e}")
            return None


# ---------- Convenience free functions ----------

def save_csv_increment(csv_path: str, date_str: Optional[str] = None) -> str:
    """
    Upload local CSV with auto-incremented name 'NNN_YYYY-MM-DD_processed.csv'.
    Returns gs:// URL or "".
    """
    saver = FirebaseSaver()
    return saver.upload_file_with_increment(csv_path, date_str=date_str)

def save_dataframe_increment(df: pd.DataFrame, date_str: Optional[str] = None) -> str:
    """
    Upload DataFrame with auto-incremented name 'NNN_YYYY-MM-DD_processed.csv'.
    Returns gs:// URL or "".
    """
    saver = FirebaseSaver()
    return saver.upload_dataframe_with_increment(df, date_str=date_str)

def save_efficiency_data(filename: str, efficiency_score: float) -> bool:
    """
    Save efficiency data to Firebase efficiency.json file.
    
    Args:
        filename: The processed filename
        efficiency_score: The fuel efficiency score (0-100)
        
    Returns:
        bool: True if successful, False otherwise
    """
    saver = FirebaseSaver()
    return saver.save_efficiency_data(filename, efficiency_score)

def get_efficiency_by_filename(filename: str) -> Optional[dict]:
    """
    Get efficiency data for a specific filename from Firebase.
    
    Args:
        filename: The processed filename
        
    Returns:
        dict: Efficiency data or None if not found
    """
    saver = FirebaseSaver()
    return saver.get_efficiency_by_filename(filename)