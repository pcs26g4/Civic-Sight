from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from services.image_service import upload_image_to_cloudinary, upload_video_to_cloudinary
import cv2
import uuid
from database import get_db
from app_utils.exif import extract_gps_from_image_bytes
from app_utils.geo import group_by_location
from app_utils.deduplication import check_duplicate_image
from yolo_service import get_yolo_service
from app_models import Ticket, SubTicket, ComplaintImage, User
from app_utils.image_hash import calculate_image_hash

from crud import (
    get_or_create_ticket,
    get_or_create_sub_ticket,
    save_image
)

router = APIRouter(prefix="/api/complaints", tags=["Complaints"])

# ---------------- Authority Mapping ----------------
from app_utils.constants import AUTHORITY_MAP, DEFAULT_LAT, DEFAULT_LON


# ==================================================
# SINGLE IMAGE COMPLAINT UPLOAD (REFERENCE-STYLE)
# ==================================================
@router.post("/")
async def upload_complaint_image(
    issue_type: str = Form(..., description="Issue type, e.g., pathholes, garbage, street_debris"),
    latitude: Optional[float] = Form(None, description="Optional manual latitude"),
    longitude: Optional[float] = Form(None, description="Optional manual longitude"),
    user_id: Optional[int] = Form(None, description="ID of the user raising the complaint"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a single complaint image.
    - Detects GPS from EXIF if available.
    - Falls back to manual latitude/longitude if provided.
    - Creates/gets a Ticket (by location) and SubTicket (by issue type).
    - Saves the image and returns ticket + sub_ticket info.
    - Uses existing deduplication rules (same image + same location => duplicate).
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    # Normalize issue type to match AUTHORITY_MAP keys
    normalized_issue = issue_type.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    # Map common variants to our keys
    if normalized_issue in {"pothole"}:
        normalized_issue = "pathholes"
    if normalized_issue in {"streetdebris", "streetdebris"}:
        normalized_issue = "streetdebris"

    # Find matching key in AUTHORITY_MAP (since some keys have underscores)
    if normalized_issue not in AUTHORITY_MAP:
        raise HTTPException(status_code=400, detail=f"Invalid issue type: {issue_type}")

    authority = AUTHORITY_MAP[normalized_issue]

    # Read image bytes
    image_bytes = await file.read()

    # 🔍 Extract GPS from image bytes
    gps_data = extract_gps_from_image_bytes(image_bytes)

    # ✅ FINAL LOCATION LOGIC
    # ONLY use EXIF GPS data from the image itself
    # Screenshots and images without GPS metadata will have NO location
    if gps_data:
        lat = gps_data["latitude"]
        lon = gps_data["longitude"]
        gps_extracted = True
        gps_source = "exif"
    else:
        # No EXIF GPS -> No location data
        # This ensures screenshots don't get browser location
        lat = DEFAULT_LAT
        lon = DEFAULT_LON
        gps_extracted = False
        gps_source = "none"

    # ✅ CRITICAL FIX: Check duplicates BEFORE YOLO processing
    check_lat = lat if gps_extracted and lat != DEFAULT_LAT else None
    check_lon = lon if gps_extracted and lon != DEFAULT_LON else None

    is_duplicate, reason, existing_info = check_duplicate_image(
        db=db,
        image_bytes=image_bytes,  # ✅ Use ORIGINAL bytes for deduplication
        latitude=check_lat,
        longitude=check_lon,
        issue_type=normalized_issue,  # ✅ MUST pass issue_type
        distance_threshold=50,  # 50 meters for location-aware matching
    )

    if is_duplicate:
        # ✅ REJECT EARLY - No YOLO processing, no Cloudinary upload
        return {
            "status": "duplicate",
            "message": reason or "This complaint is already registered. Thanks for your concern.",
            "existing_complaint": existing_info,
            "cloudinary_saved": True,  # Indicate we saved Cloudinary costs
            "yolo_processing_saved": True,  # Indicate we saved YOLO processing
        }

    # 🔍 Run YOLO detection for results (ONLY if NOT duplicate)
    yolo_service = get_yolo_service()
    annotated_bytes = image_bytes  # Fallback
    max_confidence = None
    detections_found = False
    try:
        detections, annotated_cv2_img = yolo_service.detect_from_bytes(image_bytes)
        if annotated_cv2_img is not None:
            _, encoded_img = cv2.imencode('.jpg', annotated_cv2_img)
            annotated_bytes = encoded_img.tobytes()
            
            if detections:
                detections_found = True
                max_confidence = max(d['confidence'] for d in detections)
    except Exception as e:
        print(f"YOLO detection failed for single upload: {e}")

    if not detections_found:
        return {
            "status": "rejected",
            "message": "There is no distortion detected. Thanks for your concern.",
            "latitude": lat if gps_extracted else None,
            "longitude": lon if gps_extracted else None,
        }

    # 1️⃣ MAIN TICKET (LOCATION BASED)
    ticket = get_or_create_ticket(db, lat, lon, user_id=user_id)

    # 2️⃣ SUB TICKET (ISSUE BASED)
    sub_ticket = get_or_create_sub_ticket(
        db,
        ticket.ticket_id,
        normalized_issue,
        authority,
    )

    # ✅ Calculate hash for database storage
    image_hash = calculate_image_hash(image_bytes, use_perceptual=True)

    # ✅ UPLOAD TO CLOUDINARY AND GET URL (ONLY if not duplicate AND has detection)
    unique_id = uuid.uuid4().hex[:8]
    safe_name = f"{unique_id}_{file.filename}"
    
    try:
        image_url = upload_image_to_cloudinary(
            file_bytes=annotated_bytes,
            filename=safe_name
        )
    except Exception as e:
        print(f"Cloudinary upload failed: {e}")
        # ❌ NO LOCAL FALLBACK - Fail fast
        raise HTTPException(
            status_code=500,
            detail="Image upload failed. Please try again later."
        )

    # 3️⃣ SAVE IMAGE TO DB WITH URL (NO BYTES) AND HASH
    image = save_image(
        db=db,
        sub_id=sub_ticket.sub_id,
        image_url=image_url,  # ✅ Store URL instead of bytes
        content_type=file.content_type,
        gps_extracted=gps_extracted,
        media_type="image",
        file_name=safe_name,
        latitude=lat if gps_extracted else None,
        longitude=lon if gps_extracted else None,
        confidence=max_confidence,
        image_hash=image_hash  # ✅ Store hash for future deduplication
    )

    return {
        "status": "success",
        "ticket_id": ticket.ticket_id,
        "sub_id": sub_ticket.sub_id,
        "issue_type": normalized_issue,
        "authority": authority,
        "area": ticket.area,
        "district": ticket.district,
        "gps": {
            "latitude": lat if gps_extracted else None,
            "longitude": lon if gps_extracted else None,
            "source": gps_source,
        },
        "image_url": image.image_url,  # ✅ Return URL to frontend
        "confidence": image.confidence,
        "user_id": user_id,
        "deduplication_performed": True  # ✅ Confirmation deduplication was done
    }


# ==================================================
# BATCH COMPLAINT UPLOAD (IMAGES + VIDEOS) - FIXED
# ==================================================
@router.post("/batch")
async def upload_batch_complaints(
    files: List[UploadFile] = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    user_id: Optional[int] = Form(None),
    db: Session = Depends(get_db)
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    yolo_service = get_yolo_service()
    processed_items = []
    duplicate_count = 0

    # ---------------- PROCESS EACH FILE ----------------
    for file in files:
        content_type = file.content_type
        file_bytes = await file.read()

        lat, lon = None, None
        gps_extracted = False
        gps_source = None

        # ---------- IMAGE GPS (EXIF ONLY) ----------
        if content_type and content_type.startswith("image/"):
            gps_data = extract_gps_from_image_bytes(file_bytes)
            if gps_data and gps_data.get("latitude") and gps_data.get("longitude"):
                lat = gps_data["latitude"]
                lon = gps_data["longitude"]
                gps_extracted = True
                gps_source = "exif"

        # If no EXIF GPS found, use defaults (no location)
        if not gps_extracted:
            lat = DEFAULT_LAT
            lon = DEFAULT_LON
            gps_source = "none"

        # ✅ CRITICAL FIX: Quick YOLO scan JUST for issue detection (not full processing)
        potential_issues = []
        if content_type.startswith("image/"):
            try:
                # Quick scan to detect potential issues
                quick_detections, _ = yolo_service.quick_detect_from_bytes(file_bytes)
                if not quick_detections:
                    # Try full detection as fallback
                    quick_detections, _ = yolo_service.detect_from_bytes(file_bytes)
                
                # Map detected classes to issue types
                detected_issues_map = {}
                for det in quick_detections:
                    class_name = det["class_name"].lower().replace("_", "").replace(" ", "")
                    confidence = det["confidence"]

                    if class_name in AUTHORITY_MAP:
                        if (class_name not in detected_issues_map or 
                            confidence > detected_issues_map[class_name]):
                            detected_issues_map[class_name] = confidence
                
                potential_issues = [
                    {"issue_type": issue, "confidence": conf}
                    for issue, conf in detected_issues_map.items()
                ]
                
            except Exception as e:
                print(f"Quick issue detection failed for {file.filename}: {e}")
                potential_issues = []
        
        # For videos or if no issues detected
        if not potential_issues:
            # Still need to check if it might be a video
            if content_type.startswith("video/"):
                # For videos, we'll process them fully and check duplicates later
                potential_issues = [{"issue_type": "unknown", "confidence": 0.5}]
            else:
                processed_items.append({
                    "file_bytes": file_bytes,
                    "content_type": content_type,
                    "file_name": file.filename,
                    "media_type": "video" if content_type.startswith("video/") else "image",
                    "latitude": lat,
                    "longitude": lon,
                    "issue_type": None,
                    "gps_extracted": gps_extracted,
                    "detection_confidence": None,
                    "no_detection": True,
                    "skip_processing": True,  # Skip full processing
                })
                continue

        # ✅ Check duplicates for EACH potential issue BEFORE full processing
        valid_issues = []
        duplicate_issues = []
        
        for issue in potential_issues:
            if issue["issue_type"] == "unknown":
                # For videos or unknown issues, skip duplicate check
                valid_issues.append(issue)
                continue
                
            is_duplicate, reason, existing_info = check_duplicate_image(
                db=db,
                image_bytes=file_bytes,  # Original bytes
                latitude=lat if gps_extracted and lat != DEFAULT_LAT else None,
                longitude=lon if gps_extracted and lon != DEFAULT_LON else None,
                issue_type=issue["issue_type"],
                distance_threshold=50
            )
            
            if not is_duplicate:
                valid_issues.append(issue)
            else:
                duplicate_count += 1
                duplicate_issues.append({
                    "issue": issue["issue_type"],
                    "reason": reason,
                    "existing_info": existing_info
                })
        
        # If all issues are duplicates, skip this file entirely
        if not valid_issues and duplicate_issues:
            # Mark as duplicate for the first detected issue
            processed_items.append({
                "file_bytes": file_bytes,
                "content_type": content_type,
                "file_name": file.filename,
                "media_type": "video" if content_type.startswith("video/") else "image",
                "latitude": lat,
                "longitude": lon,
                "issue_type": duplicate_issues[0]["issue"],
                "gps_extracted": gps_extracted,
                "detection_confidence": None,
                "no_detection": False,
                "is_duplicate": True,
                "duplicate_reason": duplicate_issues[0]["reason"],
                "existing_complaint": duplicate_issues[0]["existing_info"],
                "skip_processing": True,  # Skip YOLO and Cloudinary
            })
            continue

        # ✅ ONLY NOW: Run FULL YOLO processing for non-duplicate files
        detections = []
        annotated_bytes = file_bytes
        output_vid_path = None  # For videos
        
        if content_type.startswith("image/"):
            try:
                detections, annotated_cv2_img = yolo_service.detect_from_bytes(file_bytes)
                if annotated_cv2_img is not None:
                    _, encoded_img = cv2.imencode('.jpg', annotated_cv2_img)
                    annotated_bytes = encoded_img.tobytes()
            except Exception as e:
                print(f"YOLO detection failed for image {file.filename}: {e}")
                detections = []
        
        elif content_type.startswith("video/"):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                
                output_vid_path, detections, _ = yolo_service.detect_video(tmp_path)
                
                if output_vid_path and os.path.exists(output_vid_path):
                    # Read the processed video
                    with open(output_vid_path, "rb") as f:
                        annotated_bytes = f.read()
                    
                    # Cleanup temp file
                    try: os.remove(output_vid_path)
                    except: pass
                
                # Cleanup input temp file
                try: os.remove(tmp_path)
                except: pass
                    
            except Exception as e:
                print(f"YOLO detection failed for video {file.filename}: {e}")
                detections = []

        # Find the actual detected issues from full processing
        detected_issues_map = {}
        for det in detections:
            class_name = det["class_name"].lower().replace("_", "").replace(" ", "")
            confidence = det["confidence"]

            if class_name in AUTHORITY_MAP:
                if (class_name not in detected_issues_map or 
                    confidence > detected_issues_map[class_name]):
                    detected_issues_map[class_name] = confidence

        detected_issues = [
            {"issue_type": issue, "confidence": conf}
            for issue, conf in detected_issues_map.items()
        ]

        # If no detection found after full processing
        if not detected_issues:
            processed_items.append({
                "file_bytes": file_bytes,
                "annotated_bytes": annotated_bytes,
                "content_type": content_type,
                "file_name": file.filename,
                "media_type": "video" if content_type.startswith("video/") else "image",
                "latitude": lat,
                "longitude": lon,
                "issue_type": None,
                "gps_extracted": gps_extracted,
                "detection_confidence": None,
                "no_detection": True,
                "skip_processing": False,
            })
            continue

        # ---------- STORE TEMP RECORDS (MULTI ISSUE) ----------
        for issue in detected_issues:
            # Skip if this issue was already marked as duplicate
            if any(dup["issue"] == issue["issue_type"] for dup in duplicate_issues):
                continue
                
            processed_items.append({
                "file_bytes": file_bytes,
                "annotated_bytes": annotated_bytes,
                "content_type": content_type,
                "file_name": file.filename,
                "media_type": "video" if content_type.startswith("video/") else "image",
                "latitude": lat,
                "longitude": lon,
                "issue_type": issue["issue_type"],
                "gps_extracted": gps_extracted,
                "detection_confidence": issue["confidence"],
                "no_detection": False,
                "is_duplicate": False,
                "skip_processing": False,
                "output_vid_path": output_vid_path if content_type.startswith("video/") else None,
            })

    if not processed_items:
        raise HTTPException(400, "No valid complaints detected")

    # ---------------- GROUP BY LOCATION ----------------
    location_groups = group_by_location(
        processed_items,
        distance_threshold=20  # meters
    )

    results = []
    total_rejected = 0

    # ---------------- PROCESS EACH LOCATION GROUP ----------------
    for group in location_groups:
        rep = group[0]
        
        # Lazy ticket creation
        ticket_obj = None
        ticket_result = {
            "ticket_id": None,
            "latitude": rep["latitude"],
            "longitude": rep["longitude"],
            "area": None,
            "district": None,
            "sub_tickets": [],
            "rejected_items": []
        }

        # ---------------- GROUP BY ISSUE TYPE ----------------
        issue_groups = {}
        for item in group:
            # Skip items marked for skipping
            if item.get("skip_processing", False):
                continue
            issue_groups.setdefault(item["issue_type"], []).append(item)

        for issue_type, items in issue_groups.items():
            # Handle non-detections explicitly
            if issue_type is None or issue_type not in AUTHORITY_MAP:
                for item in items:
                    ticket_result["rejected_items"].append({
                        "file_name": item["file_name"],
                        "media_type": item["media_type"],
                        "message": "There is no distortion. Thanks for your concern.",
                        "latitude": item.get("latitude"),
                        "longitude": item.get("longitude"),
                    })
                    total_rejected += 1
                continue

            authority = AUTHORITY_MAP[issue_type]
            sub_ticket_obj = None
            saved_images = []
            rejected_items_for_issue = []
            saved_count = 0
            rejected_count = 0

            for item in items:
                # Skip items that shouldn't be processed
                if item.get("skip_processing", False) or item.get("is_duplicate", False):
                    continue
                    
                # 1. Check if it's a non-detection
                if item.get("no_detection"):
                    rejected_count += 1
                    rejected_items_for_issue.append({
                        "file_name": item["file_name"],
                        "media_type": item["media_type"],
                        "message": "There is no distortion. Thanks for your concern.",
                    })
                    continue

                # 2. Final duplicate check (just in case)
                is_image = item["media_type"] == "image" and item["content_type"].startswith("image/")
                has_gps = item["gps_extracted"] and item["latitude"] is not None and item["longitude"] is not None
                
                if is_image:
                    check_lat = item["latitude"] if has_gps else None
                    check_lon = item["longitude"] if has_gps else None

                    is_duplicate, reason, existing_info = check_duplicate_image(
                        db=db,
                        image_bytes=item["file_bytes"],
                        latitude=check_lat,
                        longitude=check_lon,
                        issue_type=issue_type, 
                        distance_threshold=50
                    )
                    
                    if is_duplicate:
                        rejected_count += 1
                        rejected_items_for_issue.append({
                            "file_name": item["file_name"],
                            "media_type": item["media_type"],
                            "message": reason or "This complaint is already registered. Thanks for your concern.",
                            "existing_complaint": existing_info.get("ticket_info") if existing_info else None,
                        })
                        continue

                # 3. If NOT a duplicate, ensure TICKET and SUB-TICKET exist
                if ticket_obj is None:
                    ticket_obj = get_or_create_ticket(db, rep["latitude"], rep["longitude"], user_id=user_id)
                    ticket_result["ticket_id"] = ticket_obj.ticket_id
                    ticket_result["area"] = ticket_obj.area
                    ticket_result["district"] = ticket_obj.district

                if sub_ticket_obj is None:
                    sub_ticket_obj = get_or_create_sub_ticket(db, ticket_obj.ticket_id, issue_type, authority)

                # 4. Calculate hash for database storage
                image_hash = calculate_image_hash(item["file_bytes"], use_perceptual=True) if is_image else None

                # 5. Upload to Cloudinary
                unique_id = uuid.uuid4().hex[:8]
                safe_name = f"{unique_id}_{item['file_name']}"
                
                try:
                    if item["media_type"] == "image":
                        image_url = upload_image_to_cloudinary(
                            file_bytes=item["annotated_bytes"],
                            filename=safe_name
                        )
                    else:
                        # For videos, we need to save processed video first
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                            tmp.write(item["annotated_bytes"])
                            tmp_video_path = tmp.name
                        
                        try:
                            image_url = upload_video_to_cloudinary(
                                file_path=tmp_video_path,
                                filename=safe_name
                            )
                        finally:
                            # Cleanup temp file
                            try: os.remove(tmp_video_path)
                            except: pass

                except Exception as e:
                    print(f"Cloudinary upload failed for {safe_name}: {e}")
                    # ❌ NO LOCAL FALLBACK - Fail for this item only
                    rejected_count += 1
                    rejected_items_for_issue.append({
                        "file_name": item["file_name"],
                        "media_type": item["media_type"],
                        "message": "Failed to upload media. Please try again.",
                    })
                    continue

                # 6. Save to DB
                image_obj = save_image(
                    db=db,
                    sub_id=sub_ticket_obj.sub_id,
                    image_url=image_url,  # ✅ Store URL, not bytes
                    content_type=item["content_type"],
                    gps_extracted=item["gps_extracted"],
                    media_type=item["media_type"],
                    file_name=safe_name,
                    latitude=item["latitude"] if has_gps else None,
                    longitude=item["longitude"] if has_gps else None,
                    confidence=item.get("detection_confidence"),
                    image_hash=image_hash
                )

                saved_count += 1
                saved_images.append({
                    "id": image_obj.id,
                    "file_name": safe_name,
                    "media_type": item["media_type"],
                    "confidence": image_obj.confidence,
                    "image_url": image_obj.image_url
                })

            # Add issue-specific results
            if sub_ticket_obj or rejected_items_for_issue:
                ticket_result["sub_tickets"].append({
                    "sub_id": sub_ticket_obj.sub_id if sub_ticket_obj else None,
                    "issue_type": issue_type,
                    "authority": authority,
                    "media_count": saved_count,
                    "images": saved_images,
                    "rejected_count": rejected_count,
                    "rejected_items": rejected_items_for_issue if rejected_items_for_issue else None
                })
                total_rejected += rejected_count

        results.append(ticket_result)

    response = {
        "status": "success",
        "tickets_created": results,
        "total_duplicates_skipped": duplicate_count
    }
    
    if total_rejected > 0 or duplicate_count > 0:
        response["message"] = f"Processed {len(processed_items)} items. {duplicate_count} duplicates skipped, {total_rejected} rejected."
        response["duplicates_found"] = duplicate_count
        response["non_detections"] = total_rejected
    
    return response


# ==================================================
# GET ALL TICKETS
# ==================================================
@router.get("/tickets")
async def get_tickets(
    status: Optional[str] = Query(None, description="Filter by status"),
    issue_type: Optional[str] = Query(None, description="Filter by issue type"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    db: Session = Depends(get_db)
):
    """
    Get all tickets with optional filtering
    """
    query = db.query(Ticket)
    
    if status:
        query = query.filter(Ticket.status == status)
    
    if user_id:
        query = query.filter(Ticket.user_id == user_id)
    
    tickets = query.all()
    
    results = []
    for ticket in tickets:
        # Get sub-tickets for this ticket
        sub_tickets = db.query(SubTicket).filter(
            SubTicket.ticket_id == ticket.ticket_id
        ).all()
        
        # Filter by issue_type if provided
        if issue_type:
            sub_tickets = [st for st in sub_tickets if st.issue_type == issue_type]
        
        if issue_type and not sub_tickets:
            continue  # Skip ticket if no matching sub-tickets
        
        # Get images for each sub-ticket
        ticket_data = {
            "ticket_id": ticket.ticket_id,
            "latitude": ticket.latitude,
            "longitude": ticket.longitude,
            "area": ticket.area,
            "district": ticket.district,
            "status": ticket.status,
            "address": ticket.address,
            "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
            "updated_at": ticket.updated_at.isoformat() if ticket.updated_at else None,
            "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
            "user_id": ticket.user_id,
            "user_name": db.query(User.name).filter(User.id == ticket.user_id).scalar() or "Anonymous" if ticket.user_id else "Anonymous",
            "sub_tickets": []
        }
        
        for sub_ticket in sub_tickets:
            # Get all media for preview (image or video)
            images = db.query(ComplaintImage).filter(
                ComplaintImage.sub_id == sub_ticket.sub_id
            ).all()

            # Get earliest image timestamp for this sub_ticket
            earliest_image = db.query(ComplaintImage).filter(
                ComplaintImage.sub_id == sub_ticket.sub_id
            ).order_by(ComplaintImage.created_at.asc()).first()
            
            # Get GPS coordinates from first image with GPS in this sub_ticket
            gps_image = db.query(ComplaintImage).filter(
                ComplaintImage.sub_id == sub_ticket.sub_id,
                ComplaintImage.latitude.isnot(None),
                ComplaintImage.longitude.isnot(None)
            ).first()
            
            # Get the media with highest confidence for preview
            first_media = max(images, key=lambda x: x.confidence or 0) if images else None

            sub_ticket_data = {
                "id": sub_ticket.id,
                "sub_id": sub_ticket.sub_id,
                "issue_type": sub_ticket.issue_type,
                "authority": sub_ticket.authority,
                "status": sub_ticket.status,
                "assigned_to": sub_ticket.assigned_to,

                # location
                "latitude": gps_image.latitude if gps_image else None,
                "longitude": gps_image.longitude if gps_image else None,

                # 🔑 REQUIRED FOR PREVIEW
                "has_image": first_media is not None,
                "image_url": first_media.image_url if first_media else None,  # ✅ Use URL instead of ID
                "media_type": first_media.media_type if first_media else None,

                # confidence (first image is fine)
                "confidence": first_media.confidence if first_media else None,

                # counts
                "image_count": len(images),

                # optional full list with URLs
                "images": [
                    {
                        "id": img.id,
                        "file_name": img.file_name,
                        "media_type": img.media_type,
                        "confidence": img.confidence,
                        "image_url": img.image_url  # ✅ Include URL
                    }
                    for img in images
                ],

                "created_at": sub_ticket.created_at.isoformat()
                    if sub_ticket.created_at else None,
            }

            ticket_data["sub_tickets"].append(sub_ticket_data)
        
        if ticket_data["sub_tickets"]:  # Only add if has sub_tickets
            results.append(ticket_data)
    
    return {
        "status": "success",
        "count": len(results),
        "tickets": results
    }


@router.get("/geocode")
async def geocode_location(
    lat: float = Query(...),
    lon: float = Query(...)
):
    """
    Get area and district for given coordinates
    """
    from app_utils.geo import get_address_details
    details = get_address_details(lat, lon)
    return {
        "status": "success",
        "area": details.get("area", "-"),
        "district": details.get("district", "-"),
        "address": details.get("full_address", "")
    }


# ==================================================
# GET TICKET BY ID
# ==================================================
@router.get("/tickets/{ticket_id}")
async def get_ticket_by_id(
    ticket_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a specific ticket by ID
    """
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    sub_tickets = db.query(SubTicket).filter(
        SubTicket.ticket_id == ticket_id
    ).all()
    
    sub_tickets_data = []
    for sub_ticket in sub_tickets:
        images = db.query(ComplaintImage).filter(
            ComplaintImage.sub_id == sub_ticket.sub_id
        ).all()
        
        # Get GPS coordinates from first image with GPS in this sub_ticket
        gps_image = db.query(ComplaintImage).filter(
            ComplaintImage.sub_id == sub_ticket.sub_id,
            ComplaintImage.latitude.isnot(None),
            ComplaintImage.longitude.isnot(None)
        ).first()
        
        # Get earliest image timestamp for this sub_ticket
        earliest_image = db.query(ComplaintImage).filter(
            ComplaintImage.sub_id == sub_ticket.sub_id
        ).order_by(ComplaintImage.created_at.asc()).first()
        
        sub_tickets_data.append({
            "sub_id": sub_ticket.sub_id,
            "issue_type": sub_ticket.issue_type,
            "authority": sub_ticket.authority,
            "status": sub_ticket.status,
            "latitude": gps_image.latitude if gps_image else None,
            "longitude": gps_image.longitude if gps_image else None,
            "created_at": earliest_image.created_at.isoformat() if earliest_image and earliest_image.created_at else None,
            "images": [
                {
                    "id": img.id,
                    "file_name": img.file_name,
                    "content_type": img.content_type,
                    "media_type": img.media_type,
                    "gps_extracted": img.gps_extracted,
                    "latitude": img.latitude,
                    "longitude": img.longitude,
                    "confidence": img.confidence,
                    "image_url": img.image_url  # ✅ Include URL
                }
                for img in images
            ]
        })
    
    # Set ticket-level created_at based on the earliest sub-ticket
    timestamps = [st["created_at"] for st in sub_tickets_data if st["created_at"]]
    ticket_created_at = min(timestamps) if timestamps else None

    return {
        "status": "success",
        "ticket": {
            "ticket_id": ticket.ticket_id,
            "latitude": ticket.latitude,
            "longitude": ticket.longitude,
            "area": ticket.area,
            "district": ticket.district,
            "status": ticket.status,
            "address": ticket.address,
            "created_at": ticket_created_at,
            "sub_tickets": sub_tickets_data
        }
    }


# ==================================================
# UPDATE TICKET LOCATION
# ==================================================
@router.patch("/tickets/{ticket_id}/location")
async def update_ticket_location(
    ticket_id: str,
    latitude: float = Form(...),
    longitude: float = Form(...),
    db: Session = Depends(get_db)
):
    """
    Update the location for a specific ticket and all its associated images.
    """
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    from app_utils.geo import get_address_details
    address_info = get_address_details(latitude, longitude)
    
    ticket.latitude = latitude
    ticket.longitude = longitude
    ticket.area = address_info.get("area")
    ticket.district = address_info.get("district")
    ticket.address = address_info.get("full_address")
    
    # Also update images associated with this ticket's subtickets
    from sqlalchemy import update
    sub_tickets = db.query(SubTicket).filter(SubTicket.ticket_id == ticket_id).all()
    sub_ids = [st.sub_id for st in sub_tickets]
    
    if sub_ids:
        db.execute(
            update(ComplaintImage)
            .where(ComplaintImage.sub_id.in_(sub_ids))
            .values(latitude=latitude, longitude=longitude)
        )
    
    db.commit()
    return {"status": "success", "message": "Location updated successfully"}


# ==================================================
# DELETE TICKET
# ==================================================
@router.delete("/tickets/{ticket_id}")
async def delete_ticket(
    ticket_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete a specific ticket and all its associated sub-tickets and images.
    """
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Get all sub-tickets
    sub_tickets = db.query(SubTicket).filter(SubTicket.ticket_id == ticket_id).all()
    sub_ids = [st.sub_id for st in sub_tickets]
    
    # Delete images
    if sub_ids:
        # Get image file paths before deleting from DB
        images = db.query(ComplaintImage).filter(ComplaintImage.sub_id.in_(sub_ids)).all()
        
        # Optionally delete files from Cloudinary here
        # You might want to add a service function for this
        
        db.query(ComplaintImage).filter(ComplaintImage.sub_id.in_(sub_ids)).delete(synchronize_session=False)
        db.query(SubTicket).filter(SubTicket.ticket_id == ticket_id).delete(synchronize_session=False)
    
    db.delete(ticket)
    db.commit()
    
    return {"status": "success", "message": f"Ticket {ticket_id} and all related data deleted successfully"}


# ==================================================
# UPDATE TICKET STATUS
# ==================================================
@router.patch("/tickets/{ticket_id}/status")
async def update_ticket_status(
    ticket_id: str,
    status: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Update status for a ticket and all its sub-tickets.
    If status is 'resolved' or 'closed', records the resolved_at timestamp.
    """
    from datetime import datetime
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    current_time = datetime.now()
    ticket.status = status
    
    if status.lower() in ["resolved", "closed"]:
        ticket.resolved_at = current_time
    else:
        ticket.resolved_at = None
        
    # Also update all sub-tickets
    sub_tickets = db.query(SubTicket).filter(SubTicket.ticket_id == ticket_id).all()
    for st in sub_tickets:
        st.status = status
        if status.lower() in ["resolved", "closed"]:
            st.resolved_at = current_time
        else:
            st.resolved_at = None
            
    db.commit()
    return {
        "status": "success", 
        "message": f"Status updated to {status}",
        "resolved_at": current_time.isoformat() if status.lower() in ["resolved", "closed"] else None
    }