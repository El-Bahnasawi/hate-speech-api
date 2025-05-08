from db_pool import db_pool

def log_to_db(texts, results):
    conn = None
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()

        values = [(text, result["blur"], result["score"]) for text, result in zip(texts, results)]

        cursor.executemany(
            "INSERT INTO cases (text, blur, score) VALUES (%s, %s, %s);",
            values
        )

        conn.commit()
        cursor.close()
        print(f"✅ Bulk inserted {len(values)} rows.")
    except Exception as e:
        print("❌ Logging failed:", e)
    finally:
        if conn:
            db_pool.putconn(conn)
