from flask import request, jsonify
from model_loader import tokenizer, model, DEVICE
from logger import log_to_db
import torch
import psutil

def register_routes(app):

    @app.route("/check-text", methods=["POST"])
    @torch.no_grad()
    def check_text():
        try:
            texts = request.get_json().get("texts", [])
            if not texts:
                print("⚠️ No texts received.")
                return jsonify([])

            # 🧠 Log available memory
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            print(f"🧠 Available memory: {available_mb:.2f} MB")

            try:
                encodings = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=70,
                    return_tensors="pt"
                ).to(DEVICE)
            except Exception as e:
                print(f"❌ Tokenization failed: {e}")
                return jsonify({"error": "Tokenization error"}), 500

            try:
                logits = model(**encodings).logits
                probs = logits.softmax(dim=-1)[:, 1]
                scores = probs.cpu().numpy().tolist()
            except Exception as e:
                print(f"❌ Inference failed: {e}")
                return jsonify({"error": "Model inference error"}), 500

            results = [
                {"blur": s >= 0.5, "score": round(float(s), 4)}
                for s in scores
            ]

            print("🚀 Inference results:", results)

            try:
                log_to_db(texts, results)
            except Exception as e:
                print(f"❌ Logging to DB failed: {e}")

            return jsonify(results)

        except Exception as e:
            print(f"❌ /check-text route failed: {e}")
            return jsonify({"error": "Server error"}), 500

    @app.route("/test-db", methods=["GET"])
    def test_db():
        try:
            log_to_db(["Test from /test-db"], [{"blur": False, "score": 0.1111}])
            return jsonify({"status": "✅ Logged test row via psycopg2 pool + executemany."})
        except Exception as e:
            print(f"❌ /test-db logging failed: {e}")
            return jsonify({"error": str(e)}), 500
