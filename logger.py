import db_pool
import traceback

async def log_to_db(texts, results):
    pool = db_pool.db_pool
    if not pool:
        print("❌ DB pool is not initialized!")
        return

    print("📝 Logging to DB...")
    values = [(t, r["blur"], r["score"]) for t, r in zip(texts, results)]
    if not values:
        print("⚠️ No values to insert.")
        return

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    "INSERT INTO cases (text, blur, score) VALUES ($1, $2, $3)",
                    values
                )
        print(f"✅ Bulk inserted {len(values)} rows.")
    except Exception as e:
        print("❌ Logging failed:", e)
        traceback.print_exc()
