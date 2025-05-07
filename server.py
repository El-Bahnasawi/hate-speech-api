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
import wandb                # comment-out if you go the HF-only route

# ---------------- Model bootstrap (runs once) ---------------- #
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 👉 OPTION A – pull weights from W&B
run = wandb.init(project="inference", job_type="deploy", anonymous="allow")
artifact = run.use_artifact(
    "medoxz543-zewail-city-of-science-and-technology/bertweet-lora-bayes-v2/final_model:v5",
    type="model"
)
model_dir = artifact.download()

# 👉 OPTION B – if you pushed weights to HF, replace model_dir with repo id
# model_dir = "your-hf-username/bertweet-lora-bayes-v2"

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
model      = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
model.eval()

# ---------------- Flask app ---------------- #
app = Flask(__name__)
CORS(app)

@torch.no_grad()
@app.route("/check-text", methods=["POST"])
def check_text():
    texts = request.get_json().get("texts", [])
    # Tokenise as a single batch for speed
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    logits = model(**encodings).logits
    probs  = logits.softmax(dim=-1)[:, 1]          # class 1 = hateful
    scores = probs.cpu().numpy().tolist()

    results = [
        {"blur": s >= 0.5, "score": round(float(s), 4)}
        for s in scores
    ]
    return jsonify(results)

if __name__ == "__main__":
    # For local testing only.  In Render use gunicorn.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 4000)))
