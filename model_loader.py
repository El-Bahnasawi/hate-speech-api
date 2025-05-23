# Revised model_loader.py (unchanged as sync)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb
import os
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model = None

def load_model_sync():
    global tokenizer, model

    try:
        print("🔄 Initializing Weights & Biases...")
        run = wandb.init(project="inference", job_type="deploy", anonymous="allow")
    except Exception as e:
        print(f"❌ wandb.init() failed: {e}")
        sys.exit(1)

    try:
        print("📦 Using model artifact from wandb...")
        artifact = run.use_artifact(
            "medoxz543-zewail-city-of-science-and-technology/bertweet-lora-bayes-v2/final_model:v5",
            type="model"
        )
        model_dir = artifact.download()
    except Exception as e:
        print(f"❌ Failed to load or download artifact: {e}")
        sys.exit(1)

    try:
        print("🔠 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        sys.exit(1)

    try:
        print("🧠 Loading model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        model.eval()
        print("✅ Model is ready!")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        sys.exit(1)
