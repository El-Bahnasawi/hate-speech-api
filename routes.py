from flask import request, jsonify
from model_loader import tokenizer, model, DEVICE
from logger import log_to_db
import torch

def register_routes(app):
    @app.route("/check-text", methods=["POST"])
    @torch.no_grad()
    def check_text():
        texts = request.get_json().get("texts", [])
        if not texts:
            return jsonify([])

        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)

        logits = model(**encodings).logits
        probs = logits.softmax(dim=-1)[:, 1]
        scores = probs.cpu().numpy().tolist()

        results = [
            {"blur": s >= 0.5, "score": round(float(s), 4)}
            for s in scores
        ]

        log_to_db(texts, results)
        return jsonify(results)

    @app.route("/test-db", methods=["GET"])
    def test_db():
        try:
            log_to_db(["Test from /test-db"], [{"blur": False, "score": 0.1111}])
            return jsonify({"status": "✅ Logged test row via psycopg2 pool + executemany."})
        except Exception as e:
            return jsonify({"error": str(e)})
