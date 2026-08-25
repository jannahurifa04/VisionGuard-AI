"""
VisionGuard privacy and data retention module.
"""

import os
import sqlite3
import pickle
from datetime import datetime, timedelta

# =========================================================
# RETENTION POLICY - adjust these to your actual requirements
# =========================================================

SNAPSHOT_RETENTION_DAYS = 30
CLIP_RETENTION_DAYS = 30
EVENT_LOG_RETENTION_DAYS = 90
FACE_PROFILE_INACTIVITY_LIMIT_DAYS = 180  # delete if unseen this long

DATABASE_NAME = "events.db"
SNAPSHOT_FOLDER = "snapshots"
CLIP_FOLDER = "clips"

DELETION_LOG_FILE = "deletion_log.txt"


def _log_deletion(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DELETION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[PRIVACY CLEANUP] {message}")


def cleanup_expired_snapshots_and_clips():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    snapshot_cutoff = datetime.now() - timedelta(days=SNAPSHOT_RETENTION_DAYS)
    clip_cutoff = datetime.now() - timedelta(days=CLIP_RETENTION_DAYS)

    cursor.execute("""
        SELECT id, event_time, snapshot, video_clip
        FROM events
        WHERE snapshot IS NOT NULL OR video_clip IS NOT NULL
    """)

    rows = cursor.fetchall()
    deleted_count = 0

    for event_id, event_time_str, snapshot, video_clip in rows:
        try:
            event_time = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError):
            continue

        updated = False

        if snapshot and event_time < snapshot_cutoff:
            snapshot_path = os.path.join(SNAPSHOT_FOLDER, snapshot)
            if os.path.exists(snapshot_path):
                os.remove(snapshot_path)
                deleted_count += 1
            cursor.execute("UPDATE events SET snapshot = NULL WHERE id = ?", (event_id,))
            updated = True

        if video_clip and event_time < clip_cutoff:
            clip_path = os.path.join(CLIP_FOLDER, video_clip)
            if os.path.exists(clip_path):
                os.remove(clip_path)
                deleted_count += 1
            cursor.execute("UPDATE events SET video_clip = NULL WHERE id = ?", (event_id,))
            updated = True

        if updated:
            conn.commit()

    conn.close()

    if deleted_count > 0:
        _log_deletion(
            f"Removed {deleted_count} expired snapshot/clip file(s) "
            f"older than {SNAPSHOT_RETENTION_DAYS}/{CLIP_RETENTION_DAYS} days."
        )


def cleanup_expired_event_logs():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cutoff = datetime.now() - timedelta(days=EVENT_LOG_RETENTION_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("SELECT COUNT(*) FROM events WHERE event_time < ?", (cutoff_str,))
    count = cursor.fetchone()[0]

    if count > 0:
        cursor.execute("DELETE FROM events WHERE event_time < ?", (cutoff_str,))
        conn.commit()
        _log_deletion(f"Removed {count} event log row(s) older than {EVENT_LOG_RETENTION_DAYS} days.")

    conn.close()


def delete_person_completely(person_id):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT snapshot, video_clip FROM events WHERE person_id = ?", (person_id,))
    media_files = cursor.fetchall()

    removed_files = 0
    for snapshot, video_clip in media_files:
        if snapshot:
            path = os.path.join(SNAPSHOT_FOLDER, snapshot)
            if os.path.exists(path):
                os.remove(path)
                removed_files += 1
        if video_clip:
            path = os.path.join(CLIP_FOLDER, video_clip)
            if os.path.exists(path):
                os.remove(path)
                removed_files += 1

    cursor.execute("DELETE FROM events WHERE person_id = ?", (person_id,))
    events_deleted = cursor.rowcount

    cursor.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))

    try:
        cursor.execute("DELETE FROM watchlist WHERE person_id = ?", (person_id,))
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

    face_removed = False
    if os.path.exists("face_memory.pkl"):
        with open("face_memory.pkl", "rb") as f:
            data = pickle.load(f)
        known_faces = data.get("known_faces", {})
        if person_id in known_faces:
            del known_faces[person_id]
            with open("face_memory.pkl", "wb") as f:
                pickle.dump(data, f)
            face_removed = True

    reid_removed = False
    if os.path.exists("reid_memory.pkl"):
        with open("reid_memory.pkl", "rb") as f:
            data = pickle.load(f)
        known_persons = data.get("known_persons", {})
        if person_id in known_persons:
            del known_persons[person_id]
            with open("reid_memory.pkl", "wb") as f:
                pickle.dump(data, f)
            reid_removed = True

    _log_deletion(
        f"FULL DELETION - Person ID {person_id}: "
        f"{events_deleted} event(s), {removed_files} media file(s), "
        f"face_memory={face_removed}, reid_memory={reid_removed}"
    )

    return True


def find_inactive_persons():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    cutoff = datetime.now() - timedelta(days=FACE_PROFILE_INACTIVITY_LIMIT_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("SELECT person_id, last_seen FROM persons WHERE last_seen < ?", (cutoff_str,))
    inactive = cursor.fetchall()
    conn.close()

    return inactive


def run_daily_cleanup():
    print("[PRIVACY CLEANUP] Starting daily retention cleanup...")
    cleanup_expired_snapshots_and_clips()
    cleanup_expired_event_logs()

    inactive = find_inactive_persons()
    if inactive:
        print(
            f"[PRIVACY CLEANUP] {len(inactive)} person(s) inactive for "
            f"{FACE_PROFILE_INACTIVITY_LIMIT_DAYS}+ days - review with "
            f"find_inactive_persons() and call delete_person_completely() manually."
        )
    print("[PRIVACY CLEANUP] Daily cleanup complete.")


if __name__ == "__main__":
    run_daily_cleanup()