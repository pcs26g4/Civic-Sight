import sqlite3

# Define the database file path
DB_FILE = "mdms.db"

def add_column():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        print("Checking if column 'resolution_comment' exists in 'sub_tickets'...")
        # Check if column exists to avoid errors
        cursor.execute("PRAGMA table_info(sub_tickets)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if "resolution_comment" not in columns:
            print("Adding 'resolution_comment' column...")
            cursor.execute("ALTER TABLE sub_tickets ADD COLUMN resolution_comment TEXT")
            conn.commit()
            print("Successfully added 'resolution_comment' column.")
        else:
            print("Column 'resolution_comment' already exists.")
            
    except Exception as e:
        print(f"Error migrating database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    add_column()
