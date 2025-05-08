from db_pool import db_pool

def log_to_db(texts, results):
    conn = None
    try:
        conn = db_pool.getconn()
        cursor = conn.cursor()

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
            cursor.executemany(
                "INSERT INTO cases (text, blur, score) VALUES (%s, %s, %s);",
                values
            )
        except Exception as e:
            print("❌ Error while executing insert:", e)

        try:
            conn.commit()
            print(f"✅ Bulk inserted {len(values)} rows.")
        except Exception as e:
            print("❌ Commit failed:", e)

        try:
            cursor.close()
        except Exception as e:
            print("❌ Failed to close cursor:", e)

    except Exception as e:
        print("❌ Logging failed (conn/cursor):", e)

    finally:
        try:
            if conn:
                db_pool.putconn(conn)
        except Exception as e:
            print("❌ Failed to return connection to pool:", e)
