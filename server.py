# server.py
import os, uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router
from db_pool import init_db_pool, close_db_pool
from model_loader import load_model

app = FastAPI()

# CORS first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
async def startup():
    print("🔧 Starting server…")
    await load_model()
    await init_db_pool()
    print("✅ Server ready")

@app.on_event("shutdown")
async def shutdown():
    print("🧹 Shutting down…")
    await close_db_pool()
    print("✅ Cleanup done")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "4000"))  # Render injects $PORT
    uvicorn.run("server:app", host="0.0.0.0", port=port)
