"""
AI-powered hate-speech checker for Render
----------------------------------------
Expects POST /check-text  { "texts": ["str1", "str2", ...] }

Returns [{ "blur": bool, "score": float }, ...]
"""
import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb
import psycopg2
from datetime import datetime

# Load environment variables (optional: only needed if running locally with a .env file)
from dotenv import load_dotenv
load_dotenv()

# ---------------- Environment ---------------- #
DATABASE_URL = os.getenv("DATABASE_URL")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model bootstrap (runs once) ---------------- #
run = wandb.init(project="inference", job_type="deploy", anonymous="allow")
artifact = run.use_artifact(
    "medoxz543-zewail-city-of-science-and-technology/bertweet-lora-bayes-v2/final_model:v5",
    type="model"
)
model_dir = artifact.download()

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
model.eval()

# ---------------- Flask app ---------------- #
app = Flask(__name__)
CORS(app)

# ---------------- Logging function ---------------- #
def log_to_db(texts, results):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        for text, result in zip(texts, results):
            cursor.execute(
                """
                INSERT INTO cases (text, blur, score)
                VALUES (%s, %s, %s)
                """,
                (text, result["blur"], result["score"])
            )
        conn.commit()
        cursor.close()
        conn.close()
        print("📥 Logged to Supabase DB.")
    except Exception as e:
        print("❌ Logging error:", e)

# ---------------- Inference endpoint ---------------- #
@app.route("/check-text", methods=["POST"])
@torch.no_grad()
def check_text():
    texts = request.get_json().get("texts", [])

    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    logits = model(**encodings).logits
    probs = logits.softmax(dim=-1)[:, 1]  # class 1 = hateful
    scores = probs.cpu().numpy().tolist()

    results = [
        {"blur": s >= 0.5, "score": round(float(s), 4)}
        for s in scores
    ]

    log_to_db(texts, results)
    return jsonify(results)

if __name__ == "__main__":
    # For local testing only. In Render use gunicorn.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 4000)))