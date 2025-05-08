import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = wandb.init(project="inference", job_type="deploy", anonymous="allow")
artifact = run.use_artifact(
    "medoxz543-zewail-city-of-science-and-technology/bertweet-lora-bayes-v2/final_model:v5",
    type="model"
)
model_dir = artifact.download()
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
model.eval()
