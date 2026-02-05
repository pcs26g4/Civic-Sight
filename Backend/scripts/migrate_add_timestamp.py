"""
Database Migration Script
Adds created_at timestamp field to complaint_images table
"""
from sqlalchemy import text
from database import engine
import sys


def migrate():
    """Run migration to add created_at timestamp field"""
    print("Starting migration: Adding created_at timestamp to complaint_images...")
    
    try:
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # Check if column already exists
                if engine.url.drivername == 'sqlite':
                    # SQLite
                    result = conn.execute(text("""
                        SELECT COUNT(*) FROM pragma_table_info('complaint_images') 
                        WHERE name = 'created_at'
                    """))
                    existing = result.scalar() > 0
                    
                    if not existing:
                        print("Adding created_at column...")
                        conn.execute(text("""
                            ALTER TABLE complaint_images 
                            ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        """))
                        print("[OK] Column added")
                    else:
                        print("[OK] Column already exists")
                        
                elif engine.url.drivername.startswith('postgresql'):
                    # PostgreSQL
                    conn.execute(text("""
                        DO $$ 
                        BEGIN
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                         WHERE table_name='complaint_images' AND column_name='created_at') THEN
                                ALTER TABLE complaint_images 
                                ADD COLUMN created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
                            END IF;
                        END $$;
                    """))
                    print("[OK] Column checked/added")
                    
                    # Create index
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_complaint_images_created_at 
                        ON complaint_images(created_at);
                    """))
                    print("[OK] Index created")
                else:
                    # MySQL or other databases
                    try:
                        conn.execute(text("""
                            ALTER TABLE complaint_images 
                            ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        """))
                        print("[OK] Column added")
                    except Exception as e:
                        if "Duplicate column" in str(e) or "already exists" in str(e).lower():
                            print("[OK] Column already exists")
                        else:
                            raise
                    
                    # Create index
                    try:
                        conn.execute(text("""
                            CREATE INDEX idx_complaint_images_created_at 
                            ON complaint_images(created_at);
                        """))
                        print("[OK] Index created")
                    except Exception:
                        pass  # Index may already exist
                
                # Commit transaction
                trans.commit()
                print("\n[SUCCESS] Migration completed successfully!")
                
            except Exception as e:
                trans.rollback()
                raise e
                
    except Exception as e:
        print(f"\n[ERROR] Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    migrate()

