import sqlite3
from incident_memory import add_memory

conn = sqlite3.connect("events.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT event_time, object_name, person_id, shirt_color, snapshot, video_clip
    FROM events
    ORDER BY id ASC
""")

rows = cursor.fetchall()
conn.close()

for event_time, event_name, person_id, shirt_color, snapshot, video_clip in rows:
    text = (
        f"CCTV Event: {event_name}. "
        f"Time: {event_time}. "
        f"Person ID: {person_id}. "
        f"Shirt Color: {shirt_color}. "
        f"Snapshot: {snapshot}. "
        f"Video: {video_clip}."
    )

    add_memory(text)

print(f"Synced {len(rows)} CCTV events to RAG memory.")