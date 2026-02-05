from database import engine, Base
from sqlalchemy import text

def add_resolved_by_column():
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE sub_tickets ADD COLUMN resolved_by VARCHAR"))
            print("Successfully added 'resolved_by' column to sub_tickets table.")
        except Exception as e:
            print(f"Error (might already exist): {e}")

if __name__ == "__main__":
    add_resolved_by_column()
