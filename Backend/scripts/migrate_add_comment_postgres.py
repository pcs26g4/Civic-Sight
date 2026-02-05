from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("DATABASE_URL not found!")
    exit(1)

def migrate():
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()
    
    try:
        print(f"Connecting to {DATABASE_URL}...")
        # Check if column exists
        # In postgres, we can check information_schema, but simpler is to just try add and ignore if exists?
        # Or better separate check.
        
        check_query = text("SELECT column_name FROM information_schema.columns WHERE table_name='sub_tickets' AND column_name='resolution_comment';")
        result = conn.execute(check_query)
        if result.fetchone():
            print("Column 'resolution_comment' already exists.")
        else:
            print("Adding 'resolution_comment' column...")
            alter_query = text("ALTER TABLE sub_tickets ADD COLUMN resolution_comment VARCHAR;")
            conn.execute(alter_query)
            conn.commit()
            print("Successfully added 'resolution_comment' column.")
            
    except Exception as e:
        print(f"Error migrating database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
