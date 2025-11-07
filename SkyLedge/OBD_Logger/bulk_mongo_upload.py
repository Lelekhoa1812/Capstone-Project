#!/usr/bin/env python3
"""
Bulk MongoDB Upload Script for Fuel Efficiency Data
Processes all pending CSV files and uploads them to MongoDB when WiFi is available.
"""

import os
import sys
import glob
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Using system environment variables only.")
    print("   Install with: pip install python-dotenv")

# Add parent directory to path to import mongo_saver
current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from mongo_saver import save_csv_to_mongo

def check_mongodb_config():
    """Check if MongoDB configuration is available."""
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("Error: MONGO_URI not found in .env file")
        return False
    
    print(f"MongoDB URI configured)")
    return True

def find_pending_csv_files(logs_dir):
    """Find all OBD CSV files that haven't been uploaded yet."""
    fuel_logs_dir = os.path.join(logs_dir, "FuelLogs")
    
    if not os.path.exists(fuel_logs_dir):
        print(f"FuelLogs directory not found: {fuel_logs_dir}")
        return []
    
    # Find all CSV files matching our naming pattern
    pattern = os.path.join(fuel_logs_dir, "obd_data_log_*.csv")
    csv_files = glob.glob(pattern)
    
    # Sort by modification time (newest first)
    csv_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(csv_files)} fuel efficiency CSV files to process")
    return csv_files

def create_session_id_from_filename(csv_filepath):
    """Generate a session ID from the CSV filename."""
    filename = os.path.basename(csv_filepath)
    # Convert obd_data_log_20231201_120000.csv -> fuel_efficiency_20231201_120000
    session_id = filename.replace('obd_data_log_', 'fuel_efficiency_').replace('.csv', '')
    return session_id

def upload_csv_files_to_mongo(csv_files, max_uploads=None):
    if not csv_files:
        print("No CSV files to upload")
        return
    
    # Limit uploads if specified
    if max_uploads:
        csv_files = csv_files[:max_uploads]
        print(f"Limiting upload to {max_uploads} files for this batch")
    
    upload_stats = {
        'successful': 0,
        'failed': 0,
        'total': len(csv_files)
    }
    
    print(f"Starting bulk upload of {len(csv_files)} fuel efficiency sessions...")
    print("=" * 60)
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            # Generate session ID
            session_id = create_session_id_from_filename(csv_file)
            filename = os.path.basename(csv_file)
            
            print(f"[{i}/{len(csv_files)}] Processing: {filename}")
            print(f"   Session ID: {session_id}")
            
            success = save_csv_to_mongo(csv_file, session_id)
            
            if success:
                upload_stats['successful'] += 1
                print(f"Upload successful")
                
                move_to_processed_folder(csv_file)
                
            else:
                upload_stats['failed'] += 1
                print(f"Upload failed")
                
        except Exception as e:
            upload_stats['failed'] += 1
            print(f"Error processing {filename}: {e}")
        
        print("-" * 40)
    
    # Print summary
    print("=" * 60)
    print("BULK UPLOAD SUMMARY")
    print(f"Successful uploads: {upload_stats['successful']}")
    print(f"Failed uploads: {upload_stats['failed']}")
    print(f"Total processed: {upload_stats['total']}")
    
    success_rate = (upload_stats['successful'] / upload_stats['total']) * 100 if upload_stats['total'] > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")

def move_to_processed_folder(csv_file):
    """Move successfully uploaded CSV to a 'processed' folder."""
    try:
        # Create processed folder if it doesn't exist
        processed_dir = os.path.join(os.path.dirname(csv_file), "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Move file
        filename = os.path.basename(csv_file)
        new_path = os.path.join(processed_dir, filename)
        os.rename(csv_file, new_path)
        print(f"Moved to processed folder: {filename}")
        
    except Exception as e:
        print(f"Could not move file to processed folder: {e}")

def main():
    """Main function to run bulk upload."""
    print("Fuel Efficiency Data - Bulk MongoDB Upload")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check MongoDB configuration first
    if not check_mongodb_config():
        return
    
    # Find logs directory (relative to script location)
    logs_dir = os.path.join(current_dir, "..", "logs")
    logs_dir = os.path.abspath(logs_dir)
    
    print(f"Searching for CSV files in: {logs_dir}")
    
    # Find pending CSV files
    csv_files = find_pending_csv_files(logs_dir)
    
    if not csv_files:
        print("No pending CSV files to upload - all caught up!")
        return
    
    # Show files to be processed
    print("\nFiles to upload:")
    for i, csv_file in enumerate(csv_files[:10], 1):  # Show first 10
        filename = os.path.basename(csv_file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(csv_file))
        print(f"   {i}. {filename} (modified: {mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    if len(csv_files) > 10:
        print(f"   ... and {len(csv_files) - 10} more files")
    
    # Confirm upload
    print(f"\nUpload {len(csv_files)} fuel efficiency sessions to MongoDB? (y/n): ", end="")
    response = input().strip().lower()
    
    if response not in ['y', 'yes']:
        print("Upload cancelled by user")
        return
    
    # Perform bulk upload
    upload_csv_files_to_mongo(csv_files)
    
    print(f"\nBulk upload completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()