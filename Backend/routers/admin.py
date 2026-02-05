from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from app_models import SubTicket, User, ApprovedInspector
from schemas import UserCreate, UserResponse
from routers.auth import get_password_hash
import datetime

router = APIRouter(prefix="/api/admin", tags=["Admin"])

@router.get("/inspector-actions")
def get_inspector_actions(db: Session = Depends(get_db)):
    """
    Get all resolved tickets with inspector info
    """
    # Fetch all sub-tickets that are resolved or closed
    # Join with User table if we had a direct user_id link, but here we saved 'resolved_by' string
    
    actions = db.query(SubTicket).filter(
        SubTicket.status.in_(["resolved", "closed"]),
        SubTicket.resolved_by.isnot(None)
    ).order_by(SubTicket.resolved_at.desc()).all()
    
    results = []
    for action in actions:
        results.append({
            "id": action.id,
            "sub_id": action.sub_id,
            "ticketId": action.ticket_id,
            "inspectorName": action.resolved_by,
            "action": f"Marked as {action.status}",
            "time": action.resolved_at.strftime("%Y-%m-%d %H:%M:%S") if action.resolved_at else None,
            "issue_type": action.issue_type,
            "department": action.authority
        })
        
    return {
        "status": "success",
        "actions": results
    }

@router.post("/create-inspector", response_model=UserResponse)
def create_inspector(user: UserCreate, db: Session = Depends(get_db)):
    # Check existing
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        name=user.name,
        email=user.email,
        hashed_password=hashed_password,
        role="INSPECTOR",
        is_approved=True, # Auto approve since Admin created it
        department=user.department
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Add to approved_inspectors
    approved = ApprovedInspector(
        user_id=new_user.id,
        name=new_user.name,
        email=new_user.email,
        department=user.department
    )
    db.add(approved)
    db.commit()
    
    return new_user
