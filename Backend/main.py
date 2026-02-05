from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from download_weights import download_model
from core.cloudinary import init_cloudinary

from routers.complaints import router as complaints_router
from routers.yolo_live import router as yolo_live_router  # NEW YOLO Live Camera API
from routers.inspector import router as inspector_router  # NEW Inspector API
from routers.auth import router as auth_router            # NEW Auth API

from database import engine, Base
import logging

logger = logging.getLogger(__name__)

# Trigger reload again after venv install for auth setup

app = FastAPI(
    title="MDMS API",
    description="Municipal Data Management System with YOLOv5 + Live Camera Detection",
    version="2.0.0"
)

# -------------------------------------
# Startup - Create Database Tables
# -------------------------------------
@app.on_event("startup")
async def startup_event():
    try:
        # 🔥 Ensure YOLO weights are available locally
        download_model()
        init_cloudinary()

        # Create DB tables
        Base.metadata.create_all(bind=engine)
        logger.info("Startup completed successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise e

# -------------------------------------
# CORS SETTINGS
# -------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # For DEV only — tighten in PROD
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------
# ROUTERS
# -------------------------------------
app.include_router(complaints_router)
app.include_router(yolo_live_router)   # New live camera streaming
app.include_router(inspector_router)   # New inspector dashboard API
app.include_router(auth_router)        # New Authentication API
from routers.admin import router as admin_router
app.include_router(admin_router)


# -------------------------------------
# Root Endpoint
# -------------------------------------
@app.get("/")
async def root():
    return {
        "message": "MDMS API",
        "endpoints": {
            "complaints": "/api/complaints",
            "YOLO Live Stream": "/api/yolo/live",
            "Stop Live Camera": "/api/yolo/stop",
            "Inspector Dashboard": "/api/inspector",
        }
    }

#--------------Health Check Endpoint----------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}


