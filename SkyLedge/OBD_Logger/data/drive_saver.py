# Google Drive Operations for OBD Logger
# Handles authentication and file uploads to Google Drive

import os
import json
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ───────────── Logging Setup ─────────────
logger = logging.getLogger("drive-saver")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


class DriveSaver:
    """Handles Google Drive operations for saving OBD data"""
    
    def __init__(self):
        self.service = None
        self.folder_id = "1r-wefqKbK9k9BeYDW1hXRbx4B-0Fvj5P"  # Default folder ID
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize Google Drive service with credentials"""
        try:
            creds_dict = json.loads(os.getenv("GDRIVE_CREDENTIALS_JSON"))
            creds = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/drive"]
            )
            self.service = build("drive", "v3", credentials=creds)
            logger.info("✅ Google Drive service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Drive initialization failed: {e}")
            self.service = None
    
    def upload_csv_to_drive(self, file_path: str, folder_id: str = None) -> bool:
        """
        Upload a CSV file to Google Drive
        
        Args:
            file_path (str): Path to the CSV file to upload
            folder_id (str, optional): Target folder ID. Uses default if not specified.
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.service:
            logger.error("❌ Drive service not initialized")
            return False
        
        target_folder = folder_id or self.folder_id
        
        try:
            file_name = os.path.basename(file_path)
            media = MediaFileUpload(file_path, mimetype='text/csv')
            metadata = {"name": file_name, "parents": [target_folder]}
            
            result = self.service.files().create(
                body=metadata, 
                media_body=media, 
                fields="id"
            ).execute()
            
            logger.info(f"✅ File uploaded to Drive successfully: {file_name} (ID: {result.get('id')})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Drive upload failed: {e}")
            return False
    
    def is_service_available(self) -> bool:
        """Check if Drive service is available"""
        return self.service is not None
    
    def get_folder_id(self) -> str:
        """Get the default folder ID"""
        return self.folder_id
    
    def set_folder_id(self, folder_id: str):
        """Set a new default folder ID"""
        self.folder_id = folder_id
        logger.info(f"📁 Default folder ID updated to: {folder_id}")


# Convenience function for backward compatibility
def get_drive_service():
    """Legacy function - returns DriveSaver instance"""
    return DriveSaver()


def upload_to_folder(service, file_path, folder_id):
    """Legacy function - uploads file to specified folder"""
    if isinstance(service, DriveSaver):
        return service.upload_csv_to_drive(file_path, folder_id)
    else:
        # Handle legacy service object
        try:
            file_name = os.path.basename(file_path)
            media = MediaFileUpload(file_path, mimetype='text/csv')
            metadata = {"name": file_name, "parents": [folder_id]}
            return service.files().create(body=metadata, media_body=media, fields="id").execute()
        except Exception as e:
            logger.error(f"❌ Legacy upload failed: {e}")
            return None
