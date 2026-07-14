import sqlite3
from memory import save_incident_memory

conn = sqlite3.connect("events.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT id, event_time, object_name, snapshot, video_clip, person_id, shirt_color
    FROM events
    ORDER BY id ASC
""")

rows = cursor.fetchall()
conn.close()

for event_id, event_time, event_name, snapshot, video_clip, person_id, shirt_color in rows:
    text = f"""
Time: {event_time}
Event: {event_name}
Person ID: {person_id}
Shirt Color: {shirt_color}
Snapshot: {snapshot}
Video Clip: {video_clip}
"""

    metadata = {
        "event_id": str(event_id),
        "event": str(event_name),
        "person_id": str(person_id),
        "shirt_color": str(shirt_color),
        "time": str(event_time),
    }

    try:
        save_incident_memory(event_id, text, metadata)
    except Exception as e:
        print(f"[SKIPPED] Event ID {event_id}: {e}")

print("Memory sync complete.")