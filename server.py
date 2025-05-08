import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from Weights & Biases
run = wandb.init(project="inference", job_type="deploy", anonymous="allow")
artifact = run.use_artifact(
    "medoxz543-zewail-city-of-science-and-technology/bertweet-lora-bayes-v2/final_model:v5",
    type="model"
)
model_dir = artifact.download()
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
model.eval()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Logging using psycopg2 directly
def log_to_db(texts, results):
    try:
        conn = psycopg2.connect(
            user=os.getenv("user"),
            password=os.getenv("password"),
            host=os.getenv("host"),
            port=os.getenv("port"),
            dbname=os.getenv("dbname")
        )
        cursor = conn.cursor()

        for text, result in zip(texts, results):
            cursor.execute(
                "INSERT INTO cases (text, blur, score) VALUES (%s, %s, %s);",
                (text, result["blur"], result["score"])
            )

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Logged to Supabase via psycopg2.")

    except Exception as e:
        print("❌ Logging failed:", e)

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
        return jsonify({"status": "✅ Logged test row via psycopg2."})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 4000)))