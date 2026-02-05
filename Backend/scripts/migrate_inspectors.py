from database import engine, Base
from app_models import User, PendingInspector, ApprovedInspector
from sqlalchemy import text

def migrate():
    print("Migrating inspector tables...")
    
    # Create new tables
    Base.metadata.create_all(bind=engine)
    print("Created PendingInspector and ApprovedInspector tables if they didn't exist.")

    # Alter 'users' table to add 'department' if it doesn't exist
    try:
        with engine.connect() as conn:
             conn.execute(text("SELECT department FROM users LIMIT 1"))
             print("'department' column already exists in 'users' table.")
    except Exception:
        print("Adding 'department' column to 'users' table...")
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN department VARCHAR"))
        print("Added 'department' column.")

if __name__ == "__main__":
    migrate()
