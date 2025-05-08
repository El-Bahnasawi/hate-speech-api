import psycopg2
from psycopg2 import pool
import os
from dotenv import load_dotenv
import sys

load_dotenv()

try:
    print("🔌 Creating PostgreSQL connection pool...")
    db_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        dbname=os.getenv("dbname"),
        sslmode="require"
    )
    if db_pool:
        print("✅ Connection pool created successfully!")
    else:
        print("❌ Pool returned None. Exiting.")
        sys.exit(1)

except Exception as e:
    print("❌ Failed to create connection pool:", e)
    sys.exit(1)
