import os
import requests
from typing import List

# Configuration
BASE_URL = "https://binkhoale1812-obd-logger.hf.space/upload-csv/"
LOG_DIRS = [f"logs/Week {i}" for i in range(11, 13)]  # Week_10 to Week_13

def find_csv_files(directories: List[str]) -> List[str]:
    """Find all CSV files in the given directories"""
    csv_files = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            continue
            
        for file in os.listdir(directory):
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(directory, file))
    return csv_files

def upload_file(file_path: str) -> None:
    """Upload a single CSV file to the endpoint"""
    file_name = os.path.basename(file_path)
    try:
        with open(file_path, 'rb') as f:
            # Note the use of 'files' parameter with the correct field name 'file'
            files = {'file': (file_name, f, 'text/csv')}
            response = requests.post(BASE_URL, files=files)
            
            if response.status_code == 200:
                print(f"Successfully uploaded {file_name}: {response.json()}")
            else:
                print(f"Failed to upload {file_name}: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error uploading {file_name}: {str(e)}")

def main():
    # Find all CSV files in the specified week directories
    csv_files = find_csv_files(LOG_DIRS)
    
    if not csv_files:
        print("No CSV files found in the specified directories.")
        return
    
    print(f"Found {len(csv_files)} CSV files to upload:")
    for file in csv_files:
        print(f" - {file}")
    
    # Upload each file
    for file_path in csv_files:
        upload_file(file_path)

if __name__ == "__main__":
    main()