import cloudinary
import cloudinary.uploader
import os
from fastapi import HTTPException

# ✅ Configure Cloudinary ONCE (use env vars)
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# ===============================
# IMAGE UPLOAD
# ===============================
def upload_image_to_cloudinary(file_bytes: bytes, filename: str) -> str:
    """
    Upload image bytes to Cloudinary and return secure URL
    """
    try:
        result = cloudinary.uploader.upload(
            file_bytes,
            folder="mdms/images",
            public_id=filename,
            overwrite=True
        )
        return result["secure_url"]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image to Cloudinary: {str(e)}"
        )


# ===============================
# VIDEO UPLOAD
# ===============================
def upload_video_to_cloudinary(file_path: str, filename: str) -> str:
    """
    Upload video file to Cloudinary and return secure URL
    """
    try:
        result = cloudinary.uploader.upload(
            file_path,
            resource_type="video",
            folder="mdms/videos",
            public_id=filename,
            overwrite=True
        )
        return result["secure_url"]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload video to Cloudinary: {str(e)}"
        )
