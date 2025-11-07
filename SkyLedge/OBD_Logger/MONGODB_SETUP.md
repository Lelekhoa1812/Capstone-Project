# MongoDB Integration Setup Guide

This guide explains how to set up MongoDB integration for the OBD Logger application.

## Prerequisites

1. **MongoDB Atlas Account**: You need a MongoDB Atlas account (free tier available)
2. **Python Dependencies**: Install the required packages

## Installation

### 1. Install Dependencies

```bash
pip install pymongo
```

Or update your requirements.txt and run:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in your project root with the following variables:

```bash
# Google Drive Configuration
GDRIVE_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project",...}

# MongoDB Atlas Connection String
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/obd_logger?retryWrites=true&w=majority

# Optional: Custom Google Drive Folder ID
GDRIVE_FOLDER_ID=1r-wefqKbK9k9BeYDW1hXRbx4B-0Fvj5P
```

## MongoDB Atlas Setup

### 1. Create Cluster
1. Go to [MongoDB Atlas](https://cloud.mongodb.com/)
2. Create a free cluster
3. Choose your preferred cloud provider and region

### 2. Database Access
1. Go to "Database Access" in the left sidebar
2. Click "Add New Database User"
3. Choose "Password" authentication
4. Set username and password (save these!)
5. Set privileges to "Read and write to any database"

### 3. Network Access
1. Go to "Network Access" in the left sidebar
2. Click "Add IP Address"
3. For development: Click "Allow Access from Anywhere" (0.0.0.0/0)
4. For production: Add your specific IP addresses

### 4. Get Connection String
1. Go to "Clusters" in the left sidebar
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Copy the connection string
5. Replace `<username>`, `<password>`, and `<dbname>` with your values

## Usage

### Automatic Saving
The application now automatically saves cleaned data to both Google Drive and MongoDB after processing.

### Manual Operations

#### Check MongoDB Status
```bash
GET /mongo/status
```

#### Get Session Summary
```bash
GET /mongo/sessions
```

#### Query Data
```bash
GET /mongo/query?session_id=session_20231201_120000&driving_style=aggressive&limit=100
```

#### Save CSV Directly to MongoDB
```bash
POST /mongo/save-csv
# Upload CSV file with optional session_id parameter
```

## Data Structure

Each document in MongoDB contains:
- All OBD sensor data from the original CSV
- `session_id`: Unique identifier for the data session
- `imported_at`: Timestamp when data was imported
- `record_index`: Original row index from CSV
- `timestamp`: OBD data timestamp (converted to datetime)
- `driving_style`: Driving style classification

## Performance Features

- **Indexes**: Automatic creation of indexes on timestamp, driving_style, and session_id
- **Connection Pooling**: Efficient connection management
- **Batch Operations**: Bulk insert for better performance
- **Error Handling**: Graceful fallback if MongoDB is unavailable

## Troubleshooting

### Connection Issues
1. Check your MongoDB URI format
2. Verify network access settings in Atlas
3. Check username/password credentials
4. Ensure cluster is running

### Data Import Issues
1. Check CSV file format
2. Verify data types in your CSV
3. Check application logs for specific error messages

### Performance Issues
1. Monitor database indexes
2. Check connection pool settings
3. Consider data partitioning for large datasets

## Security Notes

- Never commit your `.env` file to version control
- Use strong passwords for database users
- Restrict network access to necessary IP addresses only
- Consider using VPC peering for production deployments
