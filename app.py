from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import wandb

app = Flask(__name__)

# =====================
# Load Model from W&B
# =====================
run = wandb.init(project="bertweet-lora-bayes-v2", job_type="inference", anonymous="allow")

# Download artifact from W&B
artifact = run.use_artifact("medoxz543-zewail-city-of-science-and-technology/bertweet-lora-bayes-v2/final_model:v5", type="model")
artifact_dir = artifact.download()

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
model.eval()


@app.route("/")
def home():
    return "🚀 Hate Speech Detection API is Live (via W&B Artifacts)!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No input text provided."}), 400

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        hate_prob = probs[0][1].item()

    return jsonify({
        "hate_prob": round(hate_prob, 4),
        "label": "Hateful" if hate_prob > 0.5 else "Not Hateful"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
