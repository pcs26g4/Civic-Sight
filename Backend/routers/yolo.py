from fastapi import APIRouter
# This file is deprecated. All live camera logic has moved to yolo_live.py
# to avoid route conflicts and support real-time event streaming.

router = APIRouter(prefix="/api/yolo", tags=["YOLO (Deprecated)"])