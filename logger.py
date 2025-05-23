import traceback
import db_pool

_LOG_TIMEOUT = 5  # seconds per bulk insert

def sync_log_to_db(texts, results):
    values = [(t, r["blur"], r["score"]) for t, r in zip(texts, results)]
    try:
        conn = db_pool.get_conn()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO cases(text, blur, score) VALUES(%s, %s, %s)", values
            )
            conn.commit()
        print(f"✅ Bulk inserted {len(values)} rows")
        return True
    except Exception as e:
        print("❌ Logging failed:", e)
        traceback.print_exc()
        return False