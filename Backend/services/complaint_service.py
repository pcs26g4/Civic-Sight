
# Common cleanup
from app_models import Ticket, SubTicket, ComplaintImage
from crud import get_or_create_ticket, get_or_create_sub_ticket, save_image
import uuid
from app_utils.deduplication import check_duplicate_image

def create_complaint_workflow(
    db, 
    lat, 
    lon, 
    issue_type, 
    image_bytes, 
    content_type, 
    media_type="image",
    file_name=None, 
    check_duplicates=True,
    authority_map=None
):
    """
    Unified workflow to create a Ticket -> SubTicket -> Image.
    Handles defaults and deduplication.
    """
    # Default authority map if not provided
    if authority_map is None:
        # Reduced map for internal usage if needed, but optimally should be passed
        authority_map = {
            "garbage": "Sanitation Department",
            "street_debris": "Municipal Corporation",
            "pothole": "Roads Department",
             # add others as fallback or assume caller passes logic
        }

    # 1. Deduplication (Optional)
    if check_duplicates and media_type == "image":
        is_duplicate, reason, existing = check_duplicate_image(
            db=db,
            image_bytes=image_bytes,
            latitude=lat,
            longitude=lon,
            issue_type=issue_type,
            distance_threshold=50
        )
        if is_duplicate:
            return {
                "status": "duplicate",
                "message": reason,
                "existing": existing
            }

    # 2. Create Ticket (Location)
    ticket = get_or_create_ticket(db, lat, lon)

    # 3. Create SubTicket (Issue)
    # Determine authority
    authority = "Municipal Corporation" # Default
    if authority_map and issue_type in authority_map:
        authority = authority_map[issue_type]
    elif authority_map and "street_debris" in authority_map: # generic fallback
        authority = authority_map["street_debris"]

    sub_ticket = get_or_create_sub_ticket(db, ticket.ticket_id, issue_type, authority)

    # 4. Save Image
    if file_name is None:
        file_name = f"{uuid.uuid4().hex[:8]}.jpg"

    image = save_image(
        db=db,
        sub_id=sub_ticket.sub_id,
        image_bytes=image_bytes,
        content_type=content_type,
        gps_extracted=(lat is not None and lon is not None),
        media_type=media_type,
        file_name=file_name,
        latitude=lat,
        longitude=lon,
        confidence=0.99 # Default or passed? passing logic might be better
    )

    return {
        "status": "success",
        "ticket": ticket,
        "sub_ticket": sub_ticket,
        "image": image
    }
