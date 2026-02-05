from sqlalchemy import Column, Integer, String, Float, Boolean, LargeBinary, ForeignKey, DateTime
from sqlalchemy.sql import func
from database import Base


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True)
    ticket_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # ID of user who raised the complaint

    latitude = Column(Float, index=True)
    longitude = Column(Float, index=True)
    address = Column(String)
    area = Column(String)
    district = Column(String)

    status = Column(String, default="open")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True))


class SubTicket(Base):
    __tablename__ = "sub_tickets"

    id = Column(Integer, primary_key=True)
    sub_id = Column(String, unique=True, index=True, nullable=False)

    ticket_id = Column(String, ForeignKey("tickets.ticket_id"), nullable=False)
    issue_type = Column(String, nullable=False)
    authority = Column(String, nullable=False)

    status = Column(String, default="open")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True))
    resolution_comment = Column(String, nullable=True)
    resolved_by = Column(String, nullable=True) # Store inspector name
    assigned_to = Column(Integer, ForeignKey("users.id"), nullable=True) # ID of assigned inspector


class ComplaintImage(Base):
    __tablename__ = "complaint_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sub_id = Column(String, ForeignKey("sub_tickets.sub_id"), nullable=False)

    # ✅ CHANGED: Store URL instead of image bytes to fix egress issue
    image_url = Column(String, nullable=False)  # Cloudinary/Storage URL
    # ❌ REMOVED: image_data = Column(LargeBinary, nullable=False) 
    
    content_type = Column(String, nullable=False)
    media_type = Column(String, nullable=False, default="image")
    file_name = Column(String, nullable=True)
    gps_extracted = Column(Boolean, default=False)
    
    # Image deduplication fields
    image_hash = Column(String, index=True, nullable=True)  # Perceptual hash for similarity detection
    latitude = Column(Float, index=True, nullable=True)  # GPS latitude for geospatial queries
    longitude = Column(Float, index=True, nullable=True)  # GPS longitude for geospatial queries
    confidence = Column(Float, nullable=True)  # Detection confidence score
    
    # Timestamp - when image was uploaded
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="USER")  # USER, INSPECTOR, ADMIN
    department = Column(String, nullable=True) # Department for inspectors
    is_approved = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PendingInspector(Base):
    __tablename__ = "pending_inspectors"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")) # Link to User table
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    department = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ApprovedInspector(Base):
    __tablename__ = "approved_inspectors"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")) # Link to User table
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    department = Column(String, nullable=True)
    approved_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)