# db_pool.py
"""
Singleton asyncpg pool.
Creates exactly one pool on startup, closes it gracefully on shutdown.
"""

import os
import asyncpg

DB_URL = os.getenv("DATABASE_URL")         # must include ?sslmode=require
_pool   = None                             # type: asyncpg.Pool | None

async def init_db_pool():
    """Call once from FastAPI's startup event."""
    global _pool
    if _pool is not None:
        return

    _pool = await asyncpg.create_pool(
        dsn=DB_URL,
        min_size=1,
        max_size=5,          # free tier ≈ low concurrency
        timeout=30,          # seconds to obtain a connection
        command_timeout=60,  # per-statement timeout
    )
    print("✅ DB pool ready")

async def close_db_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None

def get_pool():          # ✅ simple function instead of @property
    """Return the global asyncpg pool (or None if not ready)."""
    return _pool
