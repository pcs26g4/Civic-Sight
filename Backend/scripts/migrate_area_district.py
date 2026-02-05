from sqlalchemy import text
from database import engine
import sys

def migrate():
    """Run migration to add area and district fields to tickets table"""
    print("Starting migration: Adding area and district to tickets...")
    
    try:
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # Check for area
                if engine.url.drivername == 'sqlite':
                    result = conn.execute(text("SELECT COUNT(*) FROM pragma_table_info('tickets') WHERE name = 'area'"))
                    if result.scalar() == 0:
                        print("Adding area column...")
                        conn.execute(text("ALTER TABLE tickets ADD COLUMN area VARCHAR"))
                    
                    result = conn.execute(text("SELECT COUNT(*) FROM pragma_table_info('tickets') WHERE name = 'district'"))
                    if result.scalar() == 0:
                        print("Adding district column...")
                        conn.execute(text("ALTER TABLE tickets ADD COLUMN district VARCHAR"))
                        
                else:
                    # Generic SQL
                    for col in ["area", "district"]:
                        try:
                            conn.execute(text(f"ALTER TABLE tickets ADD COLUMN {col} VARCHAR"))
                            print(f"[OK] {col} column added")
                        except Exception as e:
                            if "already exists" in str(e).lower():
                                print(f"[OK] {col} column already exists")
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
