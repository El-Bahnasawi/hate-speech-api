# Revised db_pool.py (sync using psycopg2)
import os
import psycopg2
from psycopg2 import pool

DB_URL = os.getenv("DATABASE_URL")
_pool = None

def init_db_pool_sync():
    global _pool
    _pool = psycopg2.pool.SimpleConnectionPool(
        1, 5, dsn=DB_URL
    )
    if _pool:
        print("✅ DB pool ready")

def close_db_pool_sync():
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None

def get_conn():
    if not _pool:
        raise RuntimeError("DB pool not initialized")
    return _pool.getconn()