from sqlalchemy import text
from database import engine
import sys

def migrate():
    """Run migration to add confidence field to complaint_images table"""
    print("Starting migration: Adding confidence to complaint_images...")
    
    try:
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # Check if column already exists
                if engine.url.drivername == 'sqlite':
                    result = conn.execute(text("""
                        SELECT COUNT(*) FROM pragma_table_info('complaint_images') 
                        WHERE name = 'confidence'
                    """))
                    existing = result.scalar() > 0
                    
                    if not existing:
                        print("Adding confidence column (FLOAT)...")
                        conn.execute(text("ALTER TABLE complaint_images ADD COLUMN confidence FLOAT"))
                        print("[OK] Column added")
                    else:
                        print("[OK] Column already exists")
                        
                else:
                    # PostgreSQL or others
                    try:
                        conn.execute(text("ALTER TABLE complaint_images ADD COLUMN confidence FLOAT"))
                        print("[OK] Column added")
                    except Exception as e:
                        if "Duplicate column" in str(e) or "already exists" in str(e).lower():
                            print("[OK] Column already exists")
                        else:
                            raise
                
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
