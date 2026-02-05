from app_models import Ticket, SubTicket, ComplaintImage, User, PendingInspector, ApprovedInspector
import uuid
from typing import Optional


# ---------- Ticket ----------
def get_or_create_ticket(db, lat, lon, user_id=None):
    """
    ALWAYS create a new ticket.
    Previously this reused an open ticket at the same lat/lon, which caused
    new complaints to show the same ticket_id. The product requirement is to
    generate a fresh ticket ID for each submission.
    """
    from app_utils.geo import get_address_details
    address_info = get_address_details(lat, lon)
    
    ticket = Ticket(
        ticket_id=f"MDMS-{uuid.uuid4().hex[:8].upper()}",
        user_id=user_id,
        latitude=lat,
        longitude=lon,
        area=address_info.get("area"),
        district=address_info.get("district"),
        address=address_info.get("full_address")
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


# ---------- Sub Ticket ----------
def get_next_inspector(db, authority):
    """
    Round Robin Logic (Department-based):
    1. Get ALL approved inspectors for the specific department.
    2. Find the last assigned sub-ticket for this department.
    3. Pick the next inspector in the department list.
    """
    from app_models import ApprovedInspector
    
    # Get sorted list of approved inspectors for this department
    inspectors = db.query(ApprovedInspector).filter(
        ApprovedInspector.department == authority
    ).order_by(ApprovedInspector.id).all()
    
    if not inspectors:
        # Fallback: Broaden search if no specific department matches found
        # (Though usually authority maps strictly to department)
        return None
        
    inspector_ids = [i.user_id for i in inspectors]
    
    # Find last assigned sub-ticket for this department
    last_sub = db.query(SubTicket).filter(
        SubTicket.authority == authority,
        SubTicket.assigned_to != None
    ).order_by(SubTicket.id.desc()).first()
    
    if last_sub and last_sub.assigned_to in inspector_ids:
        try:
            current_index = inspector_ids.index(last_sub.assigned_to)
            next_index = (current_index + 1) % len(inspector_ids)
            return inspector_ids[next_index]
        except ValueError:
            # Last assigned inspector removed or dept changed, restart
            return inspector_ids[0]
            
    # Default: pick first available in dept
    return inspector_ids[0]


def get_or_create_sub_ticket(
    db,
    ticket_id,
    issue_type,
    authority
):
    """
    Reuse sub-ticket when:
    - same ticket_id
    - same issue_type
    """

    # ✅ ALWAYS reuse if exists
    existing = (
        db.query(SubTicket)
        .filter(
            SubTicket.ticket_id == ticket_id,
            SubTicket.issue_type == issue_type
        )
        .first()
    )

    if existing:
        return existing

    # Create only if not exists
    # 🔍 REUSE ASSIGNMENT: If this ticket already has a sub-ticket assigned 
    # to an inspector in the SAME department, reuse that inspector.
    existing_assignment = (
        db.query(SubTicket)
        .filter(
            SubTicket.ticket_id == ticket_id,
            SubTicket.authority == authority,
            SubTicket.assigned_to != None
        )
        .first()
    )

    if existing_assignment:
        assigned_user_id = existing_assignment.assigned_to
    else:
        assigned_user_id = get_next_inspector(db, authority)

    sub_ticket = SubTicket(
        sub_id=f"SUB-{uuid.uuid4().hex[:6].upper()}",
        ticket_id=ticket_id,
        issue_type=issue_type,
        authority=authority,
        assigned_to=assigned_user_id
    )

    db.add(sub_ticket)
    db.commit()
    db.refresh(sub_ticket)
    return sub_ticket


# ---------- Image ----------
def save_image(
    db, 
    sub_id, 
    image_url: str,
    content_type: str, 
    gps_extracted: bool, 
    media_type: str = "image",
    file_name: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    confidence: Optional[float] = None,
    image_hash: Optional[str] = None   # ✅ ACCEPT HASH FROM ROUTER
):

    """

    Save image metadata with Cloudinary URL and precomputed hash.
    Hash is calculated upstream (router) during deduplication.
    

    Args:
        image_bytes: Optional. If provided, used for hash calculation.
                     If not provided, hash will be calculated from URL (slower).
    """
    # from app_utils.image_hash import get_image_hash_for_deduplication
    
    # # Calculate image hash for deduplication
    # image_hash = get_image_hash_for_deduplication(
    #     image_bytes=image_bytes,  # Use bytes if available (faster)
    #     image_url=image_url       # Fallback to URL if bytes not available
    # )
    
    image = ComplaintImage(
        sub_id=sub_id,
        image_url=image_url,  # ✅ Store URL instead of bytes
        content_type=content_type,
        media_type=media_type,
        file_name=file_name,
        gps_extracted=gps_extracted,
        image_hash=image_hash,
        latitude=latitude,
        longitude=longitude,
        confidence=confidence
    )
    db.add(image)
    db.commit()
    db.refresh(image)
    return image

# ---------- Helper for backward compatibility ----------
def save_image_with_url(
    db,
    sub_id: str,
    image_url: str,
    content_type: str,
    gps_extracted: bool,
    media_type: str = "image",
    file_name: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    confidence: Optional[float] = None
):
    """
    Alias for save_image with clearer naming.
    """
    return save_image(
        db=db,
        sub_id=sub_id,
        image_url=image_url,
        content_type=content_type,
        gps_extracted=gps_extracted,
        media_type=media_type,
        file_name=file_name,
        latitude=latitude,
        longitude=longitude,
        confidence=confidence
    )


def delete_user(db, user_id: int):
    """
    Delete a user by ID.
    - Releases any assigned tickets (sets assigned_to = NULL).
    - Deletes from auxiliary tables (PendingInspector, ApprovedInspector).
    - Finally deletes from User table.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None
    
    # Clean up assignments
    db.query(SubTicket).filter(SubTicket.assigned_to == user_id).update(
        {SubTicket.assigned_to: None}, synchronize_session=False
    )
    
    # Clean up auxiliary tables
    db.query(PendingInspector).filter(PendingInspector.user_id == user_id).delete(synchronize_session=False)
    db.query(ApprovedInspector).filter(ApprovedInspector.user_id == user_id).delete(synchronize_session=False)
    
    db.delete(user)
    db.commit()
    return user


# ---------- Get image by ID (for backward compatibility) ----------
def get_image_by_id(db, image_id: int):
    """
    Get image metadata by ID.
    Returns URL, not bytes.
    """
    image = db.query(ComplaintImage).filter(ComplaintImage.id == image_id).first()
    return image