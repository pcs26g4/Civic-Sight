"""
Image Deduplication Service - MANDATORY for Project

FINAL RULES (MUST BE ENFORCED):
1. Same issue + same location (<=50m) → REJECT (Already registered)
2. Same issue + different location + similar image → REJECT (Duplicate image detected)
3. Same location + different issue → ALLOW
4. Different location + different image → ALLOW

ARCHITECTURE:
- Calculate hash FROM BYTES during upload (fast, no egress)
- Store hash in database for future comparisons
- Use Cloudinary URLs for storage, not database bytes
"""

from sqlalchemy.orm import Session
from typing import Optional, Tuple, Dict, List
from app_models import ComplaintImage, SubTicket, Ticket
from app_utils.geo import calculate_distance
from app_utils.image_hash import calculate_image_hash, compare_image_hashes
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
DEFAULT_DISTANCE_THRESHOLD = 50  # meters - SAME LOCATION
DEFAULT_HASH_THRESHOLD = 5       # perceptual hash distance - SIMILAR IMAGE
MAX_QUERY_LIMIT = 1000           # Performance limit for duplicate checks


class DeduplicationEngine:
    """
    Main deduplication engine that enforces all mandatory rules.
    """
    
    def __init__(
        self,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        hash_threshold: int = DEFAULT_HASH_THRESHOLD
    ):
        self.distance_threshold = distance_threshold
        self.hash_threshold = hash_threshold
    
    def check_duplicate(
        self,
        db: Session,
        image_bytes: bytes,  # MANDATORY: Must have bytes for hash calculation
        latitude: Optional[float],
        longitude: Optional[float],
        issue_type: str,  # MANDATORY: Issue type must be provided
        skip_image_check: bool = False
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        MANDATORY duplicate check with all rules enforced.
        
        Args:
            image_bytes: Raw image bytes (REQUIRED for hash calculation)
            latitude: GPS latitude (optional)
            longitude: GPS longitude (optional)
            issue_type: Issue type (REQUIRED)
            skip_image_check: Skip image similarity check (only location)
            
        Returns:
            (is_duplicate, reason_message, existing_complaint_info)
        """
        logger.info(f"Checking duplicate for issue_type: {issue_type}")
        
        # 1. VALIDATE INPUTS
        if not image_bytes:
            raise ValueError("image_bytes is REQUIRED for deduplication")
        
        if not issue_type:
            raise ValueError("issue_type is REQUIRED for deduplication")
        
        # 2. CALCULATE IMAGE HASH (MANDATORY)
        try:
            image_hash = calculate_image_hash(image_bytes, use_perceptual=True)
            logger.debug(f"Calculated image hash: {image_hash[:16]}...")
        except Exception as e:
            logger.error(f"Failed to calculate image hash: {e}")
            # Without hash, we cannot do image similarity checks
            image_hash = None
        
        # 3. CHECK FOR DUPLICATES WITH SAME ISSUE TYPE
        duplicates_found = []
        
        # Query existing complaints with SAME ISSUE TYPE
        existing_complaints = self._get_existing_complaints_by_issue(
            db, issue_type, MAX_QUERY_LIMIT
        )
        
        logger.info(f"Checking against {len(existing_complaints)} existing complaints")
        
        for existing in existing_complaints:
            duplicate_type, details = self._check_complaint_duplicate(
                existing=existing,
                new_hash=image_hash,
                new_lat=latitude,
                new_lon=longitude,
                skip_image_check=skip_image_check
            )
            
            if duplicate_type:
                duplicates_found.append((duplicate_type, existing, details))
                break  # Stop at first duplicate
        
        # 4. RETURN RESULTS
        if duplicates_found:
            duplicate_type, existing, details = duplicates_found[0]
            ticket_info = self._build_ticket_info(db, existing)
            
            if duplicate_type == "location":
                return (
                    True,
                    "🚫 This complaint is already registered at this location. Thanks for your concern!",
                    {
                        "duplicate_type": "location",
                        "distance_meters": details["distance"],
                        "existing_complaint": ticket_info
                    }
                )
            elif duplicate_type == "image":
                return (
                    True,
                    "🚫 Duplicate image detected. This issue has already been reported elsewhere.",
                    {
                        "duplicate_type": "image",
                        "hash_similarity": details["hash_distance"],
                        "existing_complaint": ticket_info
                    }
                )
        
        # 5. NO DUPLICATES FOUND
        return False, None, None
    
    def _get_existing_complaints_by_issue(
        self, 
        db: Session, 
        issue_type: str, 
        limit: int
    ) -> List[ComplaintImage]:
        """
        Get existing complaints with same issue type.
        Optimized query with proper joins.
        """
        return (
            db.query(ComplaintImage)
            .join(SubTicket, SubTicket.sub_id == ComplaintImage.sub_id)
            .filter(SubTicket.issue_type == issue_type)
            .filter(ComplaintImage.image_hash.isnot(None))
            .order_by(ComplaintImage.created_at.desc())  # Check recent first
            .limit(limit)
            .all()
        )
    
    def _check_complaint_duplicate(
        self,
        existing: ComplaintImage,
        new_hash: Optional[str],
        new_lat: Optional[float],
        new_lon: Optional[float],
        skip_image_check: bool
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Check if a specific existing complaint is a duplicate.
        Returns (duplicate_type, details) or (None, None)
        """
        details = {}
        
        # RULE 1: Check SAME LOCATION
        if (new_lat and new_lon and 
            existing.latitude and existing.longitude):
            
            distance = calculate_distance(
                new_lat, new_lon,
                existing.latitude, existing.longitude
            )
            
            if distance <= self.distance_threshold:
                details["distance"] = round(distance, 2)
                return "location", details
        
        # RULE 2: Check SIMILAR IMAGE (if not skipped)
        if not skip_image_check and new_hash and existing.image_hash:
            if compare_image_hashes(new_hash, existing.image_hash, self.hash_threshold):
                # Optional: Calculate hash distance for details
                details["hash_distance"] = self._calculate_hash_distance(new_hash, existing.image_hash)
                return "image", details
        
        # No duplicate found
        return None, None
    
    def _calculate_hash_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes."""
        try:
            # Assuming perceptual hash format
            if len(hash1) == len(hash2):
                return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        except:
            pass
        return -1
    
    def _build_ticket_info(self, db: Session, image: ComplaintImage) -> Dict:
        """Build complete ticket information for duplicate response."""
        try:
            sub_ticket = db.query(SubTicket).filter(
                SubTicket.sub_id == image.sub_id
            ).first()
            
            if not sub_ticket:
                return None
            
            ticket = db.query(Ticket).filter(
                Ticket.ticket_id == sub_ticket.ticket_id
            ).first()
            
            return {
                "ticket_id": ticket.ticket_id if ticket else None,
                "sub_id": sub_ticket.sub_id,
                "issue_type": sub_ticket.issue_type,
                "authority": sub_ticket.authority,
                "status": sub_ticket.status,
                "area": ticket.area if ticket else None,
                "district": ticket.district if ticket else None,
                "created_at": image.created_at.isoformat() if image.created_at else None,
                "image_url": image.image_url,  # Cloudinary URL
                "confidence": image.confidence,
                "location": {
                    "latitude": image.latitude,
                    "longitude": image.longitude
                } if image.latitude and image.longitude else None
            }
        except Exception as e:
            logger.error(f"Error building ticket info: {e}")
            return None


# Global deduplication engine instance
deduplication_engine = DeduplicationEngine()


# --------------------------------------------------
# Public API functions (backward compatible)
# --------------------------------------------------
def check_duplicate_image(
    db: Session,
    image_bytes: bytes,  # REQUIRED
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    issue_type: Optional[str] = None,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    hash_threshold: int = DEFAULT_HASH_THRESHOLD
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """
    Main deduplication function (public API).
    MANDATORY: Requires image_bytes for hash calculation.
    """
    if not image_bytes:
        raise ValueError("image_bytes is REQUIRED for deduplication")
    
    if not issue_type:
        raise ValueError("issue_type is REQUIRED for deduplication")
    
    # Create engine with custom thresholds
    engine = DeduplicationEngine(
        distance_threshold=distance_threshold,
        hash_threshold=hash_threshold
    )
    
    return engine.check_duplicate(
        db=db,
        image_bytes=image_bytes,
        latitude=latitude,
        longitude=longitude,
        issue_type=issue_type
    )


def validate_upload_for_duplicates(
    db: Session,
    image_bytes: bytes,
    latitude: Optional[float],
    longitude: Optional[float],
    issue_type: str
) -> Dict:
    """
    Comprehensive validation for uploads.
    Returns complete validation result.
    """
    try:
        is_duplicate, message, existing_info = check_duplicate_image(
            db=db,
            image_bytes=image_bytes,
            latitude=latitude,
            longitude=longitude,
            issue_type=issue_type
        )
        
        return {
            "valid": not is_duplicate,
            "is_duplicate": is_duplicate,
            "message": message,
            "existing_complaint": existing_info,
            "hash": calculate_image_hash(image_bytes, use_perceptual=True) if not is_duplicate else None
        }
        
    except Exception as e:
        logger.error(f"Deduplication validation failed: {e}")
        return {
            "valid": False,
            "is_duplicate": False,
            "message": f"Deduplication check failed: {str(e)}",
            "error": str(e)
        }


def batch_check_duplicates(
    db: Session,
    uploads: List[Dict]
) -> List[Dict]:
    """
    Check duplicates for multiple uploads in batch.
    
    Each upload dict should contain:
    - image_bytes: bytes
    - latitude: Optional[float]
    - longitude: Optional[float]
    - issue_type: str
    - file_name: str (for reference)
    """
    results = []
    
    for upload in uploads:
        try:
            is_duplicate, message, existing_info = check_duplicate_image(
                db=db,
                image_bytes=upload["image_bytes"],
                latitude=upload.get("latitude"),
                longitude=upload.get("longitude"),
                issue_type=upload["issue_type"]
            )
            
            results.append({
                "file_name": upload.get("file_name"),
                "is_duplicate": is_duplicate,
                "message": message,
                "existing_complaint": existing_info,
                "valid": not is_duplicate
            })
            
        except Exception as e:
            logger.error(f"Batch deduplication failed for {upload.get('file_name')}: {e}")
            results.append({
                "file_name": upload.get("file_name"),
                "is_duplicate": False,
                "message": f"Check failed: {str(e)}",
                "valid": False,
                "error": str(e)
            })
    
    return results


# --------------------------------------------------
# Migration and maintenance functions
# --------------------------------------------------
def calculate_and_store_hash(
    db: Session,
    image_id: int,
    image_url: str
) -> bool:
    """
    Calculate hash from Cloudinary URL and store it.
    For migrating existing images.
    """
    try:
        from app_utils.image_hash import calculate_image_hash
        
        image = db.query(ComplaintImage).filter(ComplaintImage.id == image_id).first()
        if not image:
            return False
        
        # Calculate hash from URL
        image_hash = calculate_image_hash(image_url=image_url, use_perceptual=True)
        
        if image_hash:
            image.image_hash = image_hash
            db.commit()
            logger.info(f"Updated hash for image {image_id}")
            return True
        else:
            logger.warning(f"Could not calculate hash for image {image_id}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to update hash for image {image_id}: {e}")
        db.rollback()
        return False


def get_duplicate_statistics(db: Session) -> Dict:
    """
    Get statistics about duplicates in the system.
    """
    total_images = db.query(ComplaintImage).count()
    images_with_hash = db.query(ComplaintImage).filter(
        ComplaintImage.image_hash.isnot(None)
    ).count()
    
    # Count potential duplicates by issue type
    from sqlalchemy import func
    
    issue_stats = (
        db.query(
            SubTicket.issue_type,
            func.count(ComplaintImage.id).label('total')
        )
        .join(ComplaintImage, ComplaintImage.sub_id == SubTicket.sub_id)
        .group_by(SubTicket.issue_type)
        .all()
    )
    
    return {
        "total_images": total_images,
        "images_with_hash": images_with_hash,
        "hash_coverage_percentage": round((images_with_hash / total_images * 100) if total_images > 0 else 0, 2),
        "images_by_issue": {issue: count for issue, count in issue_stats}
    }