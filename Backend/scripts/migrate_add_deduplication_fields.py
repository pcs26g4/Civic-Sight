"""
Database Migration Script
Adds deduplication fields to complaint_images table:
- image_hash: For storing perceptual hash
- latitude: GPS latitude
- longitude: GPS longitude
- Indexes for performance
"""
from sqlalchemy import text
from database import engine
import sys


def migrate():
    """Run migration to add deduplication fields"""
    print("Starting migration: Adding deduplication fields to complaint_images...")
    
    try:
        with engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            
            try:
                # Check if columns already exist
                if engine.url.drivername == 'sqlite':
                    # SQLite
                    result = conn.execute(text("""
                        SELECT COUNT(*) FROM pragma_table_info('complaint_images') 
                        WHERE name IN ('image_hash', 'latitude', 'longitude')
                    """))
                    existing_count = result.scalar()
                    
                    if existing_count < 3:
                        print("Adding columns...")
                        conn.execute(text("ALTER TABLE complaint_images ADD COLUMN image_hash VARCHAR(255)"))
                        conn.execute(text("ALTER TABLE complaint_images ADD COLUMN latitude FLOAT"))
                        conn.execute(text("ALTER TABLE complaint_images ADD COLUMN longitude FLOAT"))
                        print("[OK] Columns added")
                    else:
                        print("[OK] Columns already exist")
                    
                    # SQLite doesn't support CREATE INDEX IF NOT EXISTS easily
                    # Just try to create indexes
                    try:
                        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_complaint_images_hash ON complaint_images(image_hash)"))
                        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_complaint_images_lat ON complaint_images(latitude)"))
                        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_complaint_images_lon ON complaint_images(longitude)"))
                        print("[OK] Indexes created")
                    except Exception as e:
                        print(f"Note: Index creation skipped (may already exist): {e}")
                        
                elif engine.url.drivername.startswith('postgresql'):
                    # PostgreSQL
                    # Check and add columns using DO block
                    conn.execute(text("""
                        DO $$ 
                        BEGIN
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                         WHERE table_name='complaint_images' AND column_name='image_hash') THEN
                                ALTER TABLE complaint_images ADD COLUMN image_hash VARCHAR(255);
                            END IF;
                            
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                         WHERE table_name='complaint_images' AND column_name='latitude') THEN
                                ALTER TABLE complaint_images ADD COLUMN latitude FLOAT;
                            END IF;
                            
                            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                         WHERE table_name='complaint_images' AND column_name='longitude') THEN
                                ALTER TABLE complaint_images ADD COLUMN longitude FLOAT;
                            END IF;
                        END $$;
                    """))
                    print("[OK] Columns checked/added")
                    
                    # Create indexes for PostgreSQL
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_complaint_images_hash 
                        ON complaint_images(image_hash);
                    """))
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_complaint_images_lat 
                        ON complaint_images(latitude);
                    """))
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_complaint_images_lon 
                        ON complaint_images(longitude);
                    """))
                    print("[OK] Indexes created")
                else:
                    # MySQL or other databases
                    # Try to add columns (will fail if they exist, but that's OK)
                    try:
                        conn.execute(text("ALTER TABLE complaint_images ADD COLUMN image_hash VARCHAR(255)"))
                    except Exception:
                        pass  # Column may already exist
                    try:
                        conn.execute(text("ALTER TABLE complaint_images ADD COLUMN latitude FLOAT"))
                    except Exception:
                        pass
                    try:
                        conn.execute(text("ALTER TABLE complaint_images ADD COLUMN longitude FLOAT"))
                    except Exception:
                        pass
                    print("[OK] Columns checked/added")
                    
                    # Create indexes
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_complaint_images_hash 
                        ON complaint_images(image_hash);
                    """))
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_complaint_images_lat 
                        ON complaint_images(latitude);
                    """))
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_complaint_images_lon 
                        ON complaint_images(longitude);
                    """))
                    print("[OK] Indexes created")
                
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

