import sqlite3

# Create database file
conn = sqlite3.connect("events.db")

# Create cursor
cursor = conn.cursor()

# Create events table
cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time TEXT,
    object_name TEXT,
    confidence REAL
)
""")

# Save changes
conn.commit()

# Close database
conn.close()

print("Database created successfully!")