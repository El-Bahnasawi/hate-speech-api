# Revised routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import torch
import model_loader
from logger import sync_log_to_db

router = APIRouter()
DEVICE = model_loader.DEVICE

class TextRequest(BaseModel):
    texts: list[str]

def _predict(batch: list[str]) -> list[float]:
    enc = model_loader.tokenizer(
        batch, padding=True, truncation=True, max_length=70, return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        probs = model_loader.model(**enc).logits.softmax(-1)[:, 1]
    return probs.cpu().tolist()

@router.post("/check-text")
def check_text(payload: TextRequest):
    if not payload.texts:
        raise HTTPException(400, "No texts provided")
    scores = _predict(payload.texts)
    results = [{"blur": s >= 0.5, "score": round(s, 4)} for s in scores]
    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "results": results,
    }

@router.post("/log-results")
def log_results(payload: TextRequest):
    scores = _predict(payload.texts)
    results = [{"blur": s >= 0.5, "score": round(s, 4)} for s in scores]
    success = sync_log_to_db(payload.texts, results)
    return {"logged": success}