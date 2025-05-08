import psycopg2
from psycopg2 import pool
import os
from dotenv import load_dotenv

load_dotenv()

try:
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
except Exception as e:
    print("❌ Failed to create connection pool:", e)
    exit()
