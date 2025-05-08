from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from routes import router
from db_pool import init_db_pool, close_db_pool
from model_loader import load_model
from logger import log

app = FastAPI()

# Enable CORS for local testing and browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, limit this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    log("🔧 Starting server...")
    await init_db_pool()
    await load_model()
    log("✅ Server is ready!")

@app.on_event("shutdown")
async def shutdown_event():
    log("🧹 Shutting down server...")
    await close_db_pool()
    log("✅ Cleanup complete.")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=4000, reload=True)
