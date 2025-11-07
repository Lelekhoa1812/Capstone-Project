# Google Drive Integration Setup Guide

This guide explains how to set up Google Drive integration for the OBD Logger application.

## Prerequisites

1. **Google Cloud Platform Account**: You need a Google Cloud Platform account
2. **Google Drive API**: Enable the Google Drive API in your project
3. **Service Account**: Create a service account with appropriate permissions
4. **Python Dependencies**: Install the required packages

## Installation

### 1. Install Dependencies

The required packages are already included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required packages:
- `google-auth`
- `google-auth-httplib2`
- `google-auth-oauthlib`
- `google-api-python-client`

### 2. Environment Variables

Create a `.env` file in your project root with the following variables:

```bash
# Google Drive Configuration
GDRIVE_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project","private_key_id":"...","private_key":"...","client_email":"...","client_id":"...","auth_uri":"https://accounts.google.com/o/oauth2/auth","token_uri":"https://oauth2.googleapis.com/token","auth_provider_x509_cert_url":"https://www.googleapis.com/oauth2/v1/certs","client_x509_cert_url":"..."}

# Optional: Custom Google Drive Folder ID
GDRIVE_FOLDER_ID=1r-wefqKbK9k9BeYDW1hXRbx4B-0Fvj5P
```

## Google Cloud Platform Setup

### 1. Create a New Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter a project name (e.g., "OBD-Logger-Drive")
4. Click "Create"

### 2. Enable Google Drive API

1. In your project, go to "APIs & Services" → "Library"
2. Search for "Google Drive API"
3. Click on "Google Drive API"
4. Click "Enable"

### 3. Create Service Account

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "Service Account"
3. Fill in the service account details:
   - **Name**: `obd-logger-drive`
   - **Description**: `Service account for OBD Logger Google Drive operations`
4. Click "Create and Continue"
5. For roles, select "Editor" (or create a custom role with minimal permissions)
6. Click "Continue" → "Done"

### 4. Generate Service Account Key

1. In the service accounts list, click on your newly created service account
2. Go to the "Keys" tab
3. Click "Add Key" → "Create New Key"
4. Choose "JSON" format
5. Click "Create" - this will download a JSON file
6. **Important**: Keep this file secure and never commit it to version control

### 5. Share Google Drive Folder

1. Go to [Google Drive](https://drive.google.com/)
2. Create a new folder or use an existing one
3. Right-click the folder → "Share"
4. Add your service account email (found in the JSON file under `client_email`)
5. Give it "Editor" permissions
6. Copy the folder ID from the URL (the long string after `/folders/`)

## Configuration

### 1. Set Up Credentials

Copy the contents of your downloaded JSON file and set it as the `GDRIVE_CREDENTIALS_JSON` environment variable:

```bash
export GDRIVE_CREDENTIALS_JSON='{"type":"service_account","project_id":"your-project",...}'
```

Or add it to your `.env` file.

### 2. Configure Folder ID

Set the `GDRIVE_FOLDER_ID` environment variable to your target folder ID:

```bash
export GDRIVE_FOLDER_ID="your_folder_id_here"
```

## Usage

### Automatic Saving

The application automatically uploads cleaned CSV files to Google Drive after processing.

### Manual Operations

#### Initialize Drive Service

```python
from drive_saver import DriveSaver

# Create instance
drive_saver = DriveSaver()

# Check if service is available
if drive_saver.is_service_available():
    print("✅ Google Drive service ready")
else:
    print("❌ Google Drive service not available")
```

#### Upload CSV File

```python
# Upload to default folder
success = drive_saver.upload_csv_to_drive("path/to/your/file.csv")

# Upload to specific folder
success = drive_saver.upload_csv_to_drive("path/to/your/file.csv", "custom_folder_id")
```

#### Configuration Management

```python
# Get current folder ID
current_folder = drive_saver.get_folder_id()

# Set new folder ID
drive_saver.set_folder_id("new_folder_id")
```

### Legacy Functions (Backward Compatibility)

The module maintains backward compatibility with existing code:

```python
from drive_saver import get_drive_service, upload_to_folder

# Legacy usage
service = get_drive_service()
result = upload_to_folder(service, "file.csv", "folder_id")
```

## File Management

### Supported File Types

- **CSV files**: Primary format for OBD data
- **Text files**: Other data formats
- **Binary files**: Limited support

### File Naming

Files are uploaded with their original names. The system automatically:
- Preserves file extensions
- Maintains original timestamps
- Creates unique names if conflicts exist

### Storage Organization

- **Default folder**: All files go to the configured default folder
- **Custom folders**: Specify different folders for different data types
- **Session-based**: Files are organized by processing sessions

## Error Handling

### Common Issues

1. **Authentication Errors**
   - Check service account credentials
   - Verify API is enabled
   - Ensure service account has proper permissions

2. **Permission Errors**
   - Verify folder sharing settings
   - Check service account email is added to folder
   - Ensure "Editor" or higher permissions

3. **Quota Exceeded**
   - Monitor Google Drive storage usage
   - Check API quotas in Google Cloud Console
   - Consider upgrading storage plan

### Troubleshooting

#### Check Service Status

```python
from drive_saver import DriveSaver

saver = DriveSaver()
print(f"Service available: {saver.is_service_available()}")
print(f"Current folder: {saver.get_folder_id()}")
```

#### Test Connection

```python
# Try uploading a small test file
test_success = drive_saver.upload_csv_to_drive("test.csv")
if test_success:
    print("✅ Connection test successful")
else:
    print("❌ Connection test failed")
```

## Security Best Practices

### Credential Management

- **Never commit** service account JSON to version control
- **Use environment variables** for sensitive data
- **Rotate keys** regularly
- **Limit permissions** to minimum required

### Access Control

- **Restrict folder access** to necessary users only
- **Monitor access logs** in Google Drive
- **Use organization policies** for additional security
- **Consider VPC Service Controls** for production

### Network Security

- **HTTPS only** for all API communications
- **Firewall rules** to restrict access if needed
- **Audit logs** for suspicious activity

## Performance Optimization

### Upload Strategies

- **Batch uploads** for multiple files
- **Compression** for large CSV files
- **Async processing** for non-blocking operations

### Monitoring

- **Track upload success rates**
- **Monitor file sizes and upload times**
- **Set up alerts** for failures

## Integration with OBD Logger

### Automatic Uploads

The system automatically uploads files after:
1. Data processing completion
2. CSV cleaning and validation
3. Feature engineering
4. Quality checks

### File Naming Convention

Uploaded files follow the pattern:
```
cleaned_{timestamp}.csv
```

Where `{timestamp}` is the normalized timestamp from the processing session.

### Error Recovery

If uploads fail:
- Files remain in local storage
- Errors are logged for debugging
- Processing continues without interruption
- Manual retry options available

## Advanced Configuration

### Custom Scopes

Modify the authentication scopes in `drive_saver.py`:

```python
scopes = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file"  # More restrictive
]
```

### Retry Logic

The system includes automatic retry logic for:
- Network timeouts
- Rate limiting
- Temporary service unavailability

### Logging

Comprehensive logging includes:
- Upload success/failure
- File details and metadata
- Performance metrics
- Error details for debugging

## Support and Maintenance

### Regular Tasks

1. **Monitor storage usage** in Google Drive
2. **Check API quotas** in Google Cloud Console
3. **Review access logs** for security
4. **Update service account keys** as needed

### Troubleshooting Resources

- [Google Drive API Documentation](https://developers.google.com/drive/api)
- [Google Cloud Console](https://console.cloud.google.com/)
- [Google Drive Help](https://support.google.com/drive/)
- Application logs and error messages

### Getting Help

For issues with the OBD Logger integration:
1. Check application logs
2. Verify environment variables
3. Test with simple file uploads
4. Review Google Cloud Console for errors
