import model_loader
import torch
import psutil
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from logger import log_to_db

router = APIRouter()

class TextRequest(BaseModel):
    texts: list[str]

@router.post("/check-text")
async def check_text(payload: TextRequest):
    print(f"📅 /check-text called at {datetime.now():%Y-%m-%d %H:%M:%S}")
    texts = payload.texts
    if not texts:
        return JSONResponse(status_code=400, content={"error": "No texts provided."})

    available_mb = psutil.virtual_memory().available / 1024 / 1024
    print(f"🧠 Available memory: {available_mb:.2f} MB")

    try:
        encodings = model_loader.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=70,
            return_tensors="pt"
        ).to(model_loader.DEVICE)
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Tokenization error"})

    try:
        with torch.no_grad():
            logits = model_loader.model(**encodings).logits
            probs = logits.softmax(dim=-1)[:, 1]
            scores = probs.cpu().numpy().tolist()
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return JSONResponse(status_code=500, content={"error": "Model inference error"})

    results = [{"blur": s >= 0.5, "score": round(float(s), 4)} for s in scores]
    print("🚀 Inference results:", results)

    db_logged = False
    try:
        await log_to_db(texts, results)
        db_logged = True
    except Exception as e:
        print(f"❌ Logging to DB failed: {e}")

    return {"results": results, "db_logged": db_logged}

@router.get("/test-db")
async def test_db():
    print(f"📅 /test-db called at {datetime.now():%Y-%m-%d %H:%M:%S}")
    try:
        await log_to_db(["Test from /test-db"], [{"blur": False, "score": 0.1111}])
        return {"status": "✅ Logged test row via async pg pool."}
    except Exception as e:
        print(f"❌ /test-db logging failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
