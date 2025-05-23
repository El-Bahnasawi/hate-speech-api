# Revised server.py
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from db_pool import init_db_pool_sync, close_db_pool_sync
from model_loader import load_model_sync

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
def startup():
    print("🔧 Starting server…")
    load_model_sync()
    init_db_pool_sync()
    print("✅ Server ready")

@app.on_event("shutdown")
def shutdown():
    print("🧹 Shutting down…")
    close_db_pool_sync()
    print("✅ Cleanup done")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "4000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
