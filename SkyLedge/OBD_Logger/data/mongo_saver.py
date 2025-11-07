# MongoDB Operations for OBD Logger
# Handles data restructuring and saving to MongoDB Atlas

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# MongoDB dependencies
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️  PyMongo not available. Install with: pip install pymongo")

# ───────────── Logging Setup ─────────────
logger = logging.getLogger("mongo-saver")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


class MongoSaver:
    """Handles MongoDB operations for saving OBD data"""
    
    def __init__(self, mongo_uri: str = None):
        self.client = None
        self.db = None
        self.collection = None
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize MongoDB connection"""
        if not MONGODB_AVAILABLE:
            logger.error("❌ PyMongo not available. Cannot connect to MongoDB")
            return
            
        if not self.mongo_uri:
            logger.error("❌ MongoDB URI not provided. Set MONGO_URI environment variable")
            return
            
        try:
            # Connect with timeout and retry settings
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,         # 10 second connection timeout
                socketTimeoutMS=10000,          # 10 second socket timeout
                tlsAllowInvalidCertificates=True  # Fix for SSL certificate issues on macOS
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Set up database and collection
            self.db = self.client.obd_logger
            self.collection = self.db.obd_data
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info("✅ MongoDB connection established successfully")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.client = None
            self.db = None
            self.collection = None
        except Exception as e:
            logger.error(f"❌ MongoDB initialization error: {e}")
            self.client = None
            self.db = None
            self.collection = None
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            # Index on timestamp for time-based queries
            self.collection.create_index("timestamp")
            
            # Index on driving_style for filtering
            self.collection.create_index("driving_style")
            
            # Compound index for common queries
            self.collection.create_index([("timestamp", -1), ("driving_style", 1)])
            
            # Index on session_id for session-based queries
            self.collection.create_index("session_id")
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️  Index creation failed: {e}")
    
    def is_connected(self) -> bool:
        """Check if MongoDB connection is active"""
        if not self.client:
            return False
        
        try:
            # Ping the database
            self.client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def save_csv_to_mongo(self, csv_file_path: str, session_id: str = None) -> bool:
        """
        Read CSV file and save data to MongoDB
        
        Args:
            csv_file_path (str): Path to the CSV file
            session_id (str, optional): Unique identifier for this data session
            
        Returns:
            bool: True if save successful, False otherwise
        """
        if not self.is_connected():
            logger.error("❌ MongoDB not connected")
            return False
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            if df.empty:
                logger.warning("⚠️  CSV file is empty")
                return False
            
            # Generate session ID if not provided
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Convert DataFrame to MongoDB documents
            documents = self._dataframe_to_documents(df, session_id)
            
            # Insert documents into MongoDB
            if documents:
                result = self.collection.insert_many(documents)
                logger.info(f"✅ Saved {len(result.inserted_ids)} records to MongoDB (Session: {session_id})")
                return True
            else:
                logger.warning("⚠️  No valid documents to save")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to save CSV to MongoDB: {e}")
            return False
    
    def save_dataframe_to_mongo(self, df: pd.DataFrame, session_id: str = None) -> bool:
        """
        Save pandas DataFrame directly to MongoDB
        
        Args:
            df (pd.DataFrame): DataFrame to save
            session_id (str, optional): Unique identifier for this data session
            
        Returns:
            bool: True if save successful, False otherwise
        """
        if not self.is_connected():
            logger.error("❌ MongoDB not connected")
            return False
        
        try:
            if df.empty:
                logger.warning("⚠️  DataFrame is empty")
                return False
            
            # Generate session ID if not provided
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Convert DataFrame to MongoDB documents
            documents = self._dataframe_to_documents(df, session_id)
            
            # Insert documents into MongoDB
            if documents:
                result = self.collection.insert_many(documents)
                logger.info(f"✅ Saved {len(result.inserted_ids)} records to MongoDB (Session: {session_id})")
                return True
            else:
                logger.warning("⚠️  No valid documents to save")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to save DataFrame to MongoDB: {e}")
            return False
    
    def _dataframe_to_documents(self, df: pd.DataFrame, session_id: str) -> List[Dict[str, Any]]:
        """
        Convert pandas DataFrame to MongoDB documents
        
        Args:
            df (pd.DataFrame): Input DataFrame
            session_id (str): Session identifier
            
        Returns:
            List[Dict[str, Any]]: List of MongoDB documents
        """
        documents = []
        
        for index, row in df.iterrows():
            try:
                # Convert row to dictionary
                doc = row.to_dict()
                
                # Add metadata
                doc['session_id'] = session_id
                doc['imported_at'] = datetime.utcnow()
                doc['record_index'] = index
                
                # Handle timestamp conversion
                if 'timestamp' in doc and pd.notna(doc['timestamp']):
                    try:
                        # Try to parse timestamp
                        if isinstance(doc['timestamp'], str):
                            doc['timestamp'] = pd.to_datetime(doc['timestamp'])
                        # Convert to datetime object
                        doc['timestamp'] = doc['timestamp'].to_pydatetime()
                    except Exception:
                        # Keep as string if parsing fails
                        pass
                
                # Convert numeric types and handle NaN values
                for key, value in doc.items():
                    if pd.isna(value):
                        doc[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        doc[key] = value.item()
                    elif isinstance(value, np.bool_):
                        doc[key] = bool(value)
                
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to process row {index}: {e}")
                continue
        
        return documents
    
    def query_data(self, 
                   session_id: str = None, 
                   driving_style: str = None,
                   start_time: datetime = None,
                   end_time: datetime = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Query data from MongoDB
        
        Args:
            session_id (str, optional): Filter by session ID
            driving_style (str, optional): Filter by driving style
            start_time (datetime, optional): Start time filter
            end_time (datetime, optional): End time filter
            limit (int): Maximum number of records to return
            
        Returns:
            List[Dict[str, Any]]: Query results
        """
        if not self.is_connected():
            logger.error("❌ MongoDB not connected")
            return []
        
        try:
            # Build query filter
            query_filter = {}
            
            if session_id:
                query_filter['session_id'] = session_id
            
            if driving_style:
                query_filter['driving_style'] = driving_style
            
            if start_time or end_time:
                time_filter = {}
                if start_time:
                    time_filter['$gte'] = start_time
                if end_time:
                    time_filter['$lte'] = end_time
                query_filter['timestamp'] = time_filter
            
            # Execute query
            cursor = self.collection.find(query_filter).limit(limit)
            results = list(cursor)
            
            logger.info(f"✅ Query returned {len(results)} records")
            return results
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return []
    
    def get_session_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all data sessions
        
        Returns:
            List[Dict[str, Any]]: Session summaries
        """
        if not self.is_connected():
            logger.error("❌ MongoDB not connected")
            return []
        
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': '$session_id',
                        'count': {'$sum': 1},
                        'driving_styles': {'$addToSet': '$driving_style'},
                        'first_record': {'$min': '$timestamp'},
                        'last_record': {'$max': '$timestamp'},
                        'imported_at': {'$first': '$imported_at'}
                    }
                },
                {
                    '$sort': {'imported_at': -1}
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"✅ Retrieved summary for {len(results)} sessions")
            return results
            
        except Exception as e:
            logger.error(f"❌ Session summary failed: {e}")
            return []
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_connection()


# Convenience functions
def save_csv_to_mongo(csv_file_path: str, session_id: str = None) -> bool:
    """Convenience function to save CSV to MongoDB"""
    with MongoSaver() as saver:
        return saver.save_csv_to_mongo(csv_file_path, session_id)


def save_dataframe_to_mongo(df: pd.DataFrame, session_id: str = None) -> bool:
    """Convenience function to save DataFrame to MongoDB"""
    with MongoSaver() as saver:
        return saver.save_dataframe_to_mongo(df, session_id)
