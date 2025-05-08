# logger.py
"""
Fire-and-forget bulk insert so the API never blocks on DB hiccups.
"""

import asyncio, traceback
import db_pool

_LOG_TIMEOUT = 5   # seconds to finish the insert

async def _insert(values):
    pool = db_pool.pool
    if not pool:
        print("⚠️  DB pool not ready")
        return

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                "INSERT INTO cases(text, blur, score) VALUES($1,$2,$3)",
                values,
            )
    print(f"✅ Bulk inserted {len(values)} rows")

def log_to_db(texts, results):
    """Schedule insert but never await it."""
    values = [(t, r["blur"], r["score"]) for t, r in zip(texts, results)]
    if not values:
        return
    try:
        task = asyncio.create_task(
            asyncio.wait_for(_insert(values), _LOG_TIMEOUT)
        )
        task.add_done_callback(
            lambda t: t.exception() and print("❌ Logging failed:", t.exception())
        )
    except Exception as e:
        print("❌ Failed to schedule DB log:", e)
        traceback.print_exc()
