import psycopg2
import os
from dotenv import load_dotenv


load_dotenv()


conn_details = {
        'dbname' : os.getenv('MY_DB_NAME'),
        'user': os.getenv('MY_DB_USER'),
        'password': os.getenv('MY_DB_PASSWORD'),
        'host': os.getenv('MY_DB_HOST'),
        'port': os.getenv('MY_DB_PORT')
}

def test_connection():
    try:
        # Establish a connection to the database
        conn = psycopg2.connect(**conn_details)
        # Create a cursor object
        cur = conn.cursor()
        # Execute a simple query
        cur.execute("SELECT version();")
        # Fetch the result
        db_version = cur.fetchone()
        # Print the result
        print(f"Connected to the database. PostgreSQL version: {db_version[0]}")
        # Close the cursor and connection
        cur.close()
        conn.close()
    except Exception as error:
        print(f"Error connecting to the database: {error}")

if __name__ == "__main__":
    test_connection()
