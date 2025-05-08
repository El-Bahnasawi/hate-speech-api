import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

db_pool = None

async def init_db_pool():
    global db_pool
    print("🔌 Creating PostgreSQL async connection pool...")
    db_pool = await asyncpg.create_pool(
        user=os.getenv("user"),
        password=os.getenv("password"),
        database=os.getenv("dbname"),
        host=os.getenv("host"),
        port=os.getenv("port"),
        ssl="require"
    )
    print("✅ Async connection pool created!")

async def close_db_pool():
    global db_pool
    if db_pool:
        await db_pool.close()
        print("🧹 DB pool closed.")
