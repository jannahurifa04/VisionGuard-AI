import sqlite3

conn = sqlite3.connect("events.db")
cursor = conn.cursor()

print("\n====================")
print("LAST 10 EVENTS")
print("====================")

cursor.execute("""
SELECT id, event_time, object_name, confidence
FROM events
ORDER BY id DESC
LIMIT 10
""")

rows = cursor.fetchall()

if not rows:
    print("No events found.")
else:
    for row in rows:
        print(f"ID: {row[0]} | Time: {row[1]} | Event: {row[2]} | Confidence: {row[3]}")

conn.close()