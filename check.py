import mysql.connector

# Database connection details
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'visionMaster'
}

# Create a connection
try:
    connection = mysql.connector.connect(**db_config)
    if connection.is_connected():
        print("Connection successful")
except mysql.connector.Error as err:
    print(f"Error: {err}")
