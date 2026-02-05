from sqlalchemy import text
from database import engine

def migrate():
    with engine.connect() as conn:
        print("Starting manual migration...")
        
        # Add columns to 'tickets' table if they don't exist
        for col in ["created_at", "updated_at", "resolved_at"]:
            try:
                if col == "created_at":
                    conn.execute(text(f"ALTER TABLE tickets ADD COLUMN {col} TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"))
                else:
                    conn.execute(text(f"ALTER TABLE tickets ADD COLUMN {col} TIMESTAMP WITH TIME ZONE"))
                print(f"Added {col} to 'tickets'")
            except Exception as e:
                print(f"Could not add {col} to 'tickets': {e}")
        
        # Add columns to 'sub_tickets' table if they don't exist
        for col in ["created_at", "updated_at", "resolved_at"]:
            try:
                if col == "created_at":
                    conn.execute(text(f"ALTER TABLE sub_tickets ADD COLUMN {col} TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP"))
                else:
                    conn.execute(text(f"ALTER TABLE sub_tickets ADD COLUMN {col} TIMESTAMP WITH TIME ZONE"))
                print(f"Added {col} to 'sub_tickets'")
            except Exception as e:
                print(f"Could not add {col} to 'sub_tickets': {e}")
        
        conn.commit()
        print("Migration complete.")

if __name__ == "__main__":
    migrate()
