# logger.py
import asyncio, traceback
import db_pool

_LOG_TIMEOUT = 5   # seconds per bulk insert

async def _insert(values):
    pool = db_pool.get_pool() 
    if not pool:
        raise RuntimeError("DB pool not ready")
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(
                "INSERT INTO cases(text, blur, score) VALUES($1,$2,$3)", values
            )

async def _safe_insert(values):
    try:
        await asyncio.wait_for(_insert(values), timeout=_LOG_TIMEOUT)
        print(f"✅ Bulk inserted {len(values)} rows")
        return True
    except Exception as e:
        print("❌ Logging failed:", e)
        traceback.print_exc()
        return False

def log_to_db(texts, results) -> asyncio.Task:
    """
    Fire-and-forget insert; returns a Task the caller can await if it
    wants the success flag.
    """
    values = [(t, r["blur"], r["score"]) for t, r in zip(texts, results)]
    loop = asyncio.get_event_loop()
    return loop.create_task(_safe_insert(values))
