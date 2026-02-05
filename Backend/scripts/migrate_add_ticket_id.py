"""
Migration: ensure tickets.id primary key exists.
Fixes error: psycopg2.errors.UndefinedColumn: column tickets.id does not exist
Works for PostgreSQL and SQLite (dev).
"""
from sqlalchemy import text
from database import engine


def migrate():
    """
    Add id serial/identity primary key to tickets if missing.
    """
    print("Starting migration: ensure tickets.id exists...")

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            driver = engine.url.drivername

            if driver.startswith("postgresql"):
                # Check if column exists
                exists = conn.execute(
                    text("""
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name='tickets' AND column_name='id'
                    """)
                ).scalar()

                if exists:
                    print("[OK] tickets.id already exists")
                else:
                    print("[ADD] Adding tickets.id serial primary key...")
                    conn.execute(text("ALTER TABLE tickets ADD COLUMN id SERIAL PRIMARY KEY"))
                    print("[OK] tickets.id added")

            elif driver.startswith("sqlite"):
                # SQLite needs table rebuild; simpler approach: check then skip with notice
                exists = conn.execute(
                    text("""
                        PRAGMA table_info('tickets');
                    """)
                ).fetchall()
                has_id = any(row[1] == "id" for row in exists)
                if has_id:
                    print("[OK] tickets.id already exists (sqlite)")
                else:
                    print("[WARN] tickets.id missing on SQLite. "
                          "SQLite ALTER TABLE add primary key is non-trivial. "
                          "Consider recreating the table or migrating data manually.")
            else:
                print(f"[WARN] Unsupported driver {driver}; no changes applied.")

            trans.commit()
        except Exception as e:
            trans.rollback()
            print(f"[ERROR] Migration failed: {e}")
            raise


if __name__ == "__main__":
    migrate()
