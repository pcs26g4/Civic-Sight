from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("DATABASE_URL not found!")
    exit(1)

def migrate_users():
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()
    
    try:
        print(f"Connecting to {DATABASE_URL}...")
        
        # Check if users table exists
        check_query = text("SELECT to_regclass('public.users');")
        result = conn.execute(check_query).fetchone()
        
        if result[0]:
            print("Table 'users' already exists.")
        else:
            print("Table 'users' does not exist. It will be created by SQLAlchemy on startup if main.py is reloaded.")
            # Alternatively, we can force create here if we import models, 
            # but usually restarting the backend app is safer to let SQLAlchemy handle it.
            
            # For immediate fix without restart if possible:
            create_query = text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR NOT NULL,
                email VARCHAR UNIQUE NOT NULL,
                hashed_password VARCHAR NOT NULL,
                role VARCHAR DEFAULT 'USER',
                is_approved BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """)
            conn.execute(create_query)
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_id ON users (id);"))
            conn.commit()
            print("Successfully created 'users' table.")
            
    except Exception as e:
        print(f"Error migrating database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_users()
