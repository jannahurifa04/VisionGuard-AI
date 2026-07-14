import sqlite3

conn = sqlite3.connect("events.db")
cursor = conn.cursor()

event_type = input("Enter event type: ")

cursor.execute("""
SELECT *
FROM events
WHERE object_name = ?
ORDER BY id DESC
""", (event_type,))

rows = cursor.fetchall()

print("\nRESULTS")
print("========")

if not rows:
    print("No matching events found.")
else:
    for row in rows:
        print(row)

conn.close()