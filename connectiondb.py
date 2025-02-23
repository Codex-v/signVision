import mysql.connector
from mysql.connector import Error, pooling
import time
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'visionMaster',
    'port': 3306,
    'raise_on_warnings': True,
    'connection_timeout': 30,
    'pool_size': 5,
    'pool_name': 'mypool'
}

# Connection pool
connection_pool = None

def initialize_connection_pool() -> None:
    """Initialize the database connection pool."""
    global connection_pool
    try:
        connection_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name=DB_CONFIG['pool_name'],
            pool_size=DB_CONFIG['pool_size'],
            **{k: v for k, v in DB_CONFIG.items() if k not in ['pool_name', 'pool_size']}
        )
        logger.info("Database connection pool initialized successfully")
    except Error as e:
        logger.error(f"Error initializing connection pool: {e}")
        raise

def create_db_connection(max_retries: int = 3, retry_delay: int = 2) -> Optional[mysql.connector.MySQLConnection]:
    """
    Create a database connection with retry mechanism.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
    
    Returns:
        Optional[mysql.connector.MySQLConnection]: Database connection object or None if connection fails
    """
    global connection_pool
    
    if connection_pool is None:
        try:
            initialize_connection_pool()
        except Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            return None

    for attempt in range(max_retries):
        try:
            connection = connection_pool.get_connection()
            if connection.is_connected():
                logger.info("Database connection established successfully")
                return connection
        except Error as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            continue
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            break

    logger.error(f"Failed to establish database connection after {max_retries} attempts")
    return None

def execute_query(query: str, params: tuple = None, fetch: bool = False) -> Dict[str, Any]:
    """
    Execute a database query with proper error handling.
    
    Args:
        query (str): SQL query to execute
        params (tuple): Query parameters (optional)
        fetch (bool): Whether to fetch results
    
    Returns:
        dict: Result dictionary containing status, data/error message
    """
    connection = create_db_connection()
    if not connection:
        return {
            'success': False,
            'error': 'Failed to establish database connection'
        }

    try:
        cursor = connection.cursor(dictionary=True)
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if fetch:
            result = cursor.fetchall()
            return {
                'success': True,
                'data': result
            }
        else:
            connection.commit()
            return {
                'success': True,
                'affected_rows': cursor.rowcount
            }

    except Error as e:
        logger.error(f"Database error: {e}")
        return {
            'success': False,
            'error': str(e)
        }
    finally:
        if 'cursor' in locals():
            cursor.close()
        if connection.is_connected():
            connection.close()
            logger.debug("Database connection closed")

def check_database_exists() -> bool:
    """
    Check if the database exists and create it if it doesn't.
    
    Returns:
        bool: True if database exists or was created successfully, False otherwise
    """
    temp_config = DB_CONFIG.copy()
    temp_config.pop('database', None)  # Remove database from config
    
    try:
        connection = mysql.connector.connect(**temp_config)
        cursor = connection.cursor()
        
        # Try to create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        logger.info(f"Database {DB_CONFIG['database']} checked/created successfully")
        
        return True
        
    except Error as e:
        logger.error(f"Error checking/creating database: {e}")
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def init_database() -> bool:
    """
    Initialize database and create necessary tables.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    if not check_database_exists():
        return False

    # SQL statements for creating tables
    create_tables_queries = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(256) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_training_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            sign_name VARCHAR(50) NOT NULL,
            landmarks_data JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            INDEX (user_id, sign_name)
        )
        """
    ]

    connection = create_db_connection()
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        
        for query in create_tables_queries:
            cursor.execute(query)
            
        connection.commit()
        logger.info("Database tables initialized successfully")
        return True
        
    except Error as e:
        logger.error(f"Error initializing database tables: {e}")
        return False
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Usage example
if __name__ == "__main__":
    # Initialize the database
    if init_database():
        # Example query execution
        result = execute_query(
            "SELECT * FROM users WHERE username = %s",
            params=('testuser',),
            fetch=True
        )
        
        if result['success']:
            print("Query result:", result['data'])
        else:
            print("Query error:", result['error'])