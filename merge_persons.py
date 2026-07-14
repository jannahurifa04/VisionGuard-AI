import sqlite3
import shutil
from datetime import datetime

DATABASE_PATH = "events.db"

KEEP_PERSON_ID = 1
REMOVE_PERSON_ID = 2


def safe_min_date(date1, date2):
    values = [value for value in [date1, date2] if value]
    return min(values) if values else None


def safe_max_date(date1, date2):
    values = [value for value in [date1, date2] if value]
    return max(values) if values else None


# Create a backup before changing anything
backup_name = f"events_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
shutil.copy2(DATABASE_PATH, backup_name)

print(f"Backup created: {backup_name}")

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

cursor.execute(
    """
    SELECT
        person_id,
        display_name,
        status,
        first_seen,
        last_seen,
        visit_count,
        last_shirt_color,
        last_camera_name,
        risk_level,
        avg_visit_seconds,
        longest_visit_seconds,
        loiter_count,
        last_loiter
    FROM persons
    WHERE person_id = ?
    """,
    (KEEP_PERSON_ID,),
)

keep_person = cursor.fetchone()

cursor.execute(
    """
    SELECT
        person_id,
        display_name,
        status,
        first_seen,
        last_seen,
        visit_count,
        last_shirt_color,
        last_camera_name,
        risk_level,
        avg_visit_seconds,
        longest_visit_seconds,
        loiter_count,
        last_loiter
    FROM persons
    WHERE person_id = ?
    """,
    (REMOVE_PERSON_ID,),
)

remove_person = cursor.fetchone()

if not keep_person:
    print(f"Person ID {KEEP_PERSON_ID} was not found.")
    conn.close()
    raise SystemExit

if not remove_person:
    print(f"Person ID {REMOVE_PERSON_ID} was not found.")
    conn.close()
    raise SystemExit

keep_visits = keep_person[5] or 0
remove_visits = remove_person[5] or 0

keep_avg = keep_person[9] or 0
remove_avg = remove_person[9] or 0

total_visits = keep_visits + remove_visits

if total_visits > 0:
    combined_avg = round(
        ((keep_avg * keep_visits) + (remove_avg * remove_visits))
        / total_visits
    )
else:
    combined_avg = 0

first_seen = safe_min_date(keep_person[3], remove_person[3])
last_seen = safe_max_date(keep_person[4], remove_person[4])
last_loiter = safe_max_date(keep_person[12], remove_person[12])

longest_visit = max(
    keep_person[10] or 0,
    remove_person[10] or 0,
)

combined_loiter_count = (
    (keep_person[11] or 0)
    + (remove_person[11] or 0)
)

# Prefer the most recently detected appearance information
if remove_person[4] and (
    not keep_person[4] or remove_person[4] > keep_person[4]
):
    latest_shirt = remove_person[6]
    latest_camera = remove_person[7]
else:
    latest_shirt = keep_person[6]
    latest_camera = keep_person[7]

# Keep the highest risk level
risk_order = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
}

keep_risk = keep_person[8] or "Low"
remove_risk = remove_person[8] or "Low"

combined_risk = max(
    [keep_risk, remove_risk],
    key=lambda risk: risk_order.get(risk, 0),
)

# Move all old events from Person 2 to Person 1
cursor.execute(
    """
    UPDATE events
    SET person_id = ?
    WHERE person_id = ?
    """,
    (KEEP_PERSON_ID, REMOVE_PERSON_ID),
)

moved_events = cursor.rowcount

# Update Person 1 with the combined information
cursor.execute(
    """
    UPDATE persons
    SET
        first_seen = ?,
        last_seen = ?,
        visit_count = ?,
        last_shirt_color = ?,
        last_camera_name = ?,
        risk_level = ?,
        avg_visit_seconds = ?,
        longest_visit_seconds = ?,
        loiter_count = ?,
        last_loiter = ?
    WHERE person_id = ?
    """,
    (
        first_seen,
        last_seen,
        total_visits,
        latest_shirt,
        latest_camera,
        combined_risk,
        combined_avg,
        longest_visit,
        combined_loiter_count,
        last_loiter,
        KEEP_PERSON_ID,
    ),
)

# Delete the duplicate person
cursor.execute(
    """
    DELETE FROM persons
    WHERE person_id = ?
    """,
    (REMOVE_PERSON_ID,),
)

conn.commit()
conn.close()

print(f"Moved {moved_events} events to Person ID {KEEP_PERSON_ID}.")
print(f"Deleted duplicate Person ID {REMOVE_PERSON_ID}.")
print("Merge completed successfully.")