from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import wandb
import os

app = Flask(__name__)

# ✅ W&B: Download model artifact
run = wandb.init(project="bertweet-lora-bayes-v2", job_type="inference", anonymous="allow")
artifact = run.use_artifact("medoxz543-zewail-city-of-science-and-technology/bertweet-lora-bayes-v2/final_model:v5", type="model")
artifact_dir = artifact.download()

# ✅ Load tokenizer + model from the artifact
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
model.eval()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 Hate Speech Detection API (W&B) is live!"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No input text provided."}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        hate_prob = probs[0][1].item()

    return jsonify({
        "hate_prob": round(hate_prob, 4),
        "label": "Hateful" if hate_prob > 0.5 else "Not Hateful"
    })

# 👇 Required by Vercel for serverless execution
def handler(environ, start_response):
    return app(environ, start_response)