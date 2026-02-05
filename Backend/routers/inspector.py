from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import os
import uuid
from pathlib import Path
from database import get_db
from app_models import Ticket, SubTicket, ComplaintImage
from crud import save_image

router = APIRouter(prefix="/api/inspector", tags=["Inspector"])

# Directories for storing media
UPLOAD_DIR = Path("uploads")
RESOLUTION_IMG_DIR = UPLOAD_DIR / "resolution" / "images"
RESOLUTION_IMG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# GET ASSIGNED TICKETS (Inspector View)
# --------------------------------------------------
@router.get("/tickets")
async def get_inspector_tickets(
    inspector_id: Optional[int] = Query(None, description="ID of the inspector requesting their tickets"),
    status: Optional[str] = Query(None, description="Filter by status (open, resolved, etc.)"),
    db: Session = Depends(get_db)
):
    """
    Get tickets relevant to an inspector.
    - Can filter by 'authority' (e.g., only show Garbage issues).
    - Can filter by 'status'.
    - Returns a flattened view of SubTickets since inspectors work on specific issues.
    """
    query = db.query(SubTicket)

    if inspector_id:
        query = query.filter(SubTicket.assigned_to == inspector_id)
        
    if authority:
        query = query.filter(SubTicket.authority == authority)
    
    if status:
        query = query.filter(SubTicket.status == status)

    # Order by newest first
    sub_tickets = query.order_by(SubTicket.created_at.desc()).all()

    results = []
    for sub in sub_tickets:
        # Get parent ticket for location info
        parent_ticket = db.query(Ticket).filter(Ticket.ticket_id == sub.ticket_id).first()
        
        # Get first image (complaint proof)
        complaint_image = db.query(ComplaintImage).filter(
            ComplaintImage.sub_id == sub.sub_id
        ).order_by(ComplaintImage.created_at.asc()).first()

        results.append({
            "sub_id": sub.sub_id,
            "ticket_id": sub.ticket_id,
            "issue_type": sub.issue_type,
            "authority": sub.authority,
            "status": sub.status,
            "created_at": sub.created_at,
            "resolved_at": sub.resolved_at,
            "location": {
                "latitude": parent_ticket.latitude if parent_ticket else None,
                "longitude": parent_ticket.longitude if parent_ticket else None,
                "area": parent_ticket.area if parent_ticket else None,
                "district": parent_ticket.district if parent_ticket else None,
                "address": parent_ticket.address if parent_ticket else None,
            },
            "complaint_image": {
                "url": f"/api/complaints/images/{complaint_image.id}" if complaint_image else None,
                "id": complaint_image.id if complaint_image else None
            }
        })

    return {
        "status": "success",
        "count": len(results),
        "tickets": results
    }

# --------------------------------------------------
# UPDATE SUB-TICKET STATUS & UPLOAD PROOF
# --------------------------------------------------
@router.post("/sub-tickets/{sub_id}/resolve")
async def resolve_sub_ticket(
    sub_id: str,
    status: str = Form(..., description="New status (e.g., resolved)"),
    comment: Optional[str] = Form(None, description="Inspector resolution comment"),
    resolved_by: Optional[str] = Form(None, description="Name of inspector resolving the ticket"),
    file: Optional[UploadFile] = File(None, description="Proof of resolution image"),
    db: Session = Depends(get_db)
):
    """
    Resolve a specific issue (SubTicket).
    - Updates status.
    - Optionally uploads a 'Resolution Proof' image.
    - Optionally saves a resolution comment.
    """
    sub_ticket = db.query(SubTicket).filter(SubTicket.sub_id == sub_id).first()
    if not sub_ticket:
        raise HTTPException(status_code=404, detail="SubTicket not found")

    # Update Status
    sub_ticket.status = status
    if comment:
        sub_ticket.resolution_comment = comment
    if resolved_by:
        sub_ticket.resolved_by = resolved_by

    if status.lower() in ["resolved", "closed"]:
        sub_ticket.resolved_at = datetime.now()
    else:
        sub_ticket.resolved_at = None

    # Handle Proof Image Upload
    image_info = None
    if file:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files allowed for proof")
        
        file_bytes = await file.read()
        unique_id = uuid.uuid4().hex[:8]
        safe_name = f"resolution_{unique_id}_{file.filename}"
        
        # Save to resolution folder
        save_path = RESOLUTION_IMG_DIR / safe_name
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        
        # Save to DB (using existing save_image crud, but we might want to mark it?)
        # For now, we save it as a regular image linked to the sub-ticket.
        # Ideally, we should add a 'type' field to ComplaintImage, but for now we'll rely on the folder or just date.
        
        image = save_image(
            db=db,
            sub_id=sub_id,
            image_bytes=file_bytes, # Logic inside save_image handles binary, but we pass bytes here
            content_type=file.content_type,
            gps_extracted=False, # Usually resolution pics might need GPS, but keeping it simple
            media_type="image",
            file_name=safe_name,
            latitude=None,
            longitude=None,
            confidence=None
        )
        image_info = {"id": image.id, "url": f"/api/complaints/images/{image.id}"}

    # Check if ALL sub-tickets for this parent ticket are resolved
    # If so, resolve the parent ticket too
    parent_ticket = db.query(Ticket).filter(Ticket.ticket_id == sub_ticket.ticket_id).first()
    if parent_ticket:
        all_subs = db.query(SubTicket).filter(SubTicket.ticket_id == parent_ticket.ticket_id).all()
        if all(s.status == 'resolved' for s in all_subs):
            parent_ticket.status = 'resolved'
            parent_ticket.resolved_at = datetime.now()
        elif any(s.status == 'open' for s in all_subs):
             # If at least one is open, parent is open (or partial)
             parent_ticket.status = 'open'
             parent_ticket.resolved_at = None

    db.commit()

    return {
        "status": "success",
        "message": f"Issue marked as {status}",
        "sub_ticket": {
             "sub_id": sub_ticket.sub_id,
             "status": sub_ticket.status,
             "resolved_at": sub_ticket.resolved_at
        },
        "proof_image": image_info
    }