import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).with_name("events.db")


def search_events(
    person_id=None,
    camera_id=None,
    event_type=None,
    limit=20,
):
    """
    Search VisionGuard security events.

    Optional filters:
    - person_id: permanent person ID, for example 11
    - camera_id: camera code, for example CAM01
    - event_type: PERSON_ENTERED, PERSON_PRESENT, PERSON_LEFT, LOITERING
    - limit: maximum number of results
    """

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        query = """
            SELECT
                id,
                event_time,
                object_name,
                confidence,
                snapshot,
                video_clip,
                person_id,
                saved_evidence,
                shirt_color,
                camera_id,
                camera_name
            FROM events
            WHERE 1 = 1
        """

        parameters = []

        if person_id is not None:
            query += " AND person_id = ?"
            parameters.append(person_id)

        if camera_id is not None:
            query += " AND camera_id = ?"
            parameters.append(camera_id)

        if event_type is not None:
            query += " AND object_name = ?"
            parameters.append(event_type)

        query += " ORDER BY event_time DESC LIMIT ?"
        parameters.append(limit)

        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

        return [dict(row) for row in rows]

    finally:
        connection.close()

def get_person_journey(person_id, date=None, limit=100):
    """
    Return a person's VisionGuard events in chronological order
    across all cameras.

    date example:
    "2026-10-08"
    """

    events = search_events(
        person_id=person_id,
        limit=1000 if date else limit,
    )

    if date is not None:
        events = [
            event
            for event in events
            if str(event["event_time"]).startswith(date)
        ]

    events = sorted(
        events,
        key=lambda event: event["event_time"],
    )

    return events[-limit:]

def get_latest_evidence(
    person_id=None,
    camera_id=None,
    event_type=None,
):
    """
    Return the newest VisionGuard event that has
    a snapshot or video clip.
    """

    events = search_events(
        person_id=person_id,
        camera_id=camera_id,
        event_type=event_type,
        limit=200,
    )

    for event in events:
        if event["snapshot"] or event["video_clip"]:
            return event

    return None

def get_people_present(camera_id=None, max_age_minutes=5):
    """
    Return people who appear to be currently present.

    A person is considered present when:
    - their latest event is not PERSON_LEFT
    - their latest event happened recently
    """

    from datetime import datetime, timedelta

    events = search_events(
        camera_id=camera_id,
        limit=1000,
    )

    latest_events = {}

    for event in events:
        person_id = event["person_id"]
        event_camera_id = event["camera_id"]

        if person_id is None or event_camera_id is None:
            continue

        key = (
            event_camera_id,
            person_id,
        )

        if key not in latest_events:
            latest_events[key] = event

    cutoff_time = datetime.now() - timedelta(
        minutes=max_age_minutes
    )

    people_present = []

    for event in latest_events.values():
        try:
            event_time = datetime.strptime(
                event["event_time"],
                "%Y-%m-%d %H:%M:%S",
            )
        except (TypeError, ValueError):
            continue

        if event_time < cutoff_time:
            continue

        if event["object_name"] == "PERSON_LEFT":
            continue

        people_present.append(event)

    return sorted(
        people_present,
        key=lambda event: event["event_time"],
        reverse=True,
    )