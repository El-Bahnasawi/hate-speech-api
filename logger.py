from db_pool import db_pool
import traceback  # ✅ For detailed error logs

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

        # 🧪 Show a sample of what will actually be inserted
        try:
            preview = cursor.mogrify(
                "INSERT INTO cases (text, blur, score) VALUES (%s, %s, %s);",
                values[0]
            )
            print("🧪 Sample SQL preview:", preview.decode('utf-8'))
        except Exception as e:
            print("❌ Failed to preview SQL query:", e)
            traceback.print_exc()

        try:
            cursor.executemany(
                "INSERT INTO cases (text, blur, score) VALUES (%s, %s, %s);",
                values
            )
        except Exception as e:
            print("❌ Error while executing insert:", e)
            traceback.print_exc()

        try:
            conn.commit()
            print(f"✅ Bulk inserted {len(values)} rows.")
        except Exception as e:
            print("❌ Commit failed:", e)
            traceback.print_exc()

        try:
            cursor.close()
        except Exception as e:
            print("❌ Failed to close cursor:", e)
            traceback.print_exc()

    except Exception as e:
        print("❌ Logging failed (conn/cursor):", e)
        traceback.print_exc()

    finally:
        try:
            if conn:
                db_pool.putconn(conn)
        except Exception as e:
            print("❌ Failed to return connection to pool:", e)
            traceback.print_exc()