from db_pool import db_pool
import traceback

async def log_to_db(texts, results):
    if not db_pool:
        print("❌ DB pool is not initialized!")
        return

    print("📝 Logging to DB...")
    print("Texts:", texts)
    print("Results:", results)

    values = [
        (text, result["blur"], result["score"])
        for text, result in zip(texts, results)
    ]

    if not values:
        print("⚠️ No values to insert.")
        return

    try:
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    "INSERT INTO cases (text, blur, score) VALUES ($1, $2, $3)",
                    values
                )
                print(f"✅ Bulk inserted {len(values)} rows.")
    except Exception as e:
        print("❌ Logging failed:", e)
        traceback.print_exc()
