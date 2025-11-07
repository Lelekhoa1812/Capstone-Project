import os
import json
import gspread
import logging
from oauth2client.service_account import ServiceAccountCredentials

# Setup logging
logger = logging.getLogger("upload")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")

# Authenticate with GDrive using secret
logger.info("Authenticating to Google Drive...")
creds_json = os.getenv("GDRIVE_CREDENTIALS_JSON")
if not creds_json:
    logger.error("GDRIVE_CREDENTIALS_JSON not found!")
    exit(1)

try:
    creds_dict = json.loads(creds_json)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    logger.info("Authenticated with Google Drive")
except Exception as e:
    logger.error(f"Failed to authenticate: {e}")
    exit(1)

# Folder and files
upload_dir = "./cache/obd_data/cleaned"
if not os.path.exists(upload_dir):
    logger.warning(f"Directory {upload_dir} does not exist.")
    exit(0)

# Upload all .csv files
for file in os.listdir(upload_dir):
    if file.endswith(".csv"):
        try:
            path = os.path.join(upload_dir, file)
            logger.info(f"Uploading {file}...")
            with open(path, "rb") as f:
                client.import_csv(client.create(file).id, f.read())
            logger.info(f"Uploaded {file}")
        except Exception as e:
            logger.error(f"Failed to upload {file}: {e}")