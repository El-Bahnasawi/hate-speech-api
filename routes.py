# routes.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import model_loader
from logger import log_to_db

router = APIRouter()
EXECUTOR = ThreadPoolExecutor(max_workers=2)   # keeps event-loop free
DEVICE   = model_loader.DEVICE

class TextRequest(BaseModel):
    texts: list[str]

def _predict(batch: list[str]) -> list[float]:
    """Runs in a worker thread."""
    enc = model_loader.tokenizer(
        batch, padding=True, truncation=True, max_length=70, return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        probs = model_loader.model(**enc).logits.softmax(-1)[:, 1]
    return probs.cpu().tolist()

@router.post("/check-text")
async def check_text(payload: TextRequest):
    if not payload.texts:
        raise HTTPException(400, "No texts provided")

    ts = datetime.utcnow().isoformat(timespec="seconds")
    scores = await asyncio.get_event_loop().run_in_executor(
        EXECUTOR, partial(_predict, payload.texts)
    )
    results = [{"blur": s >= 0.5, "score": round(s, 4)} for s in scores]

    # Fire-and-forget logging
    log_to_db(payload.texts, results)

    return {"timestamp": ts, "results": results}
