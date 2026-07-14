import sqlite3
from datetime import datetime, timedelta
from intent_parser import parse_intent


def get_date_range(intent):
    today = datetime.now().date()
    mode = intent.get("date_mode", "today")

    if mode == "none":
        return None, None, "none"

    if mode == "latest":
        return None, None, "latest"

    if mode == "all":
        return None, None, "all records"

    if mode == "today":
        return today, today, "today"

    if mode == "yesterday":
        d = today - timedelta(days=1)
        return d, d, "yesterday"

    if mode == "days_ago":
        days = int(intent.get("days_ago") or 0)
        d = today - timedelta(days=days)
        return d, d, f"{days} days ago"

    if mode == "months_ago":
        months = int(intent.get("months_ago") or 0)
        d = today - timedelta(days=30 * months)
        return d, d, f"{months} month(s) ago"

    if mode == "last_week":
        return today - timedelta(days=7), today, "last 7 days"

    if mode == "last_month":
        return today - timedelta(days=30), today, "last 30 days"

    if mode == "last_year":
        return today - timedelta(days=365), today, "last year"

    return today, today, "today"


def smart_search_events(question, limit=50):
    intent = parse_intent(question)

    # Chat messages should not search the database
    if intent.get("is_chat"):
        return {
            "intent": intent,
            "date_label": "chat",
            "event_type": None,
            "person_id": None,
            "rows": []
        }

    # Latest questions should only return 1 record
    if intent.get("date_mode") == "latest":
        limit = 1

    start_date, end_date, date_label = get_date_range(intent)

    event_type = intent.get("event_type")
    if event_type == "ANY":
        event_type = None

    person_id = intent.get("person_id")

    where_parts = []
    params = []

    if event_type:
        where_parts.append("object_name = ?")
        params.append(event_type)

    if person_id is not None:
        where_parts.append("person_id = ?")
        params.append(person_id)

    if start_date and end_date:
        where_parts.append("date(event_time) >= date(?)")
        params.append(start_date.strftime("%Y-%m-%d"))

        where_parts.append("date(event_time) <= date(?)")
        params.append(end_date.strftime("%Y-%m-%d"))

    where_sql = ""
    if where_parts:
        where_sql = "WHERE " + " AND ".join(where_parts)

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT event_time, object_name, person_id, shirt_color, snapshot, video_clip
        FROM events
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
    """, params + [limit])

    rows = cursor.fetchall()
    conn.close()

    return {
        "intent": intent,
        "date_label": date_label,
        "event_type": event_type,
        "person_id": person_id,
        "rows": rows
    }


def format_events_for_llm(search_result):
    intent = search_result.get("intent", {})

    if intent.get("is_chat"):
        return "Chat message. No CCTV database search needed."

    rows = search_result["rows"]

    if not rows:
        return "No matching CCTV records found."

    text = ""

    for event_time, event_name, person_id, shirt_color, snapshot, video_clip in rows:
        text += (
            f"Time: {event_time}. "
            f"Event: {event_name}. "
            f"Person ID: {person_id}. "
            f"Shirt Color: {shirt_color}. "
            f"Snapshot: {snapshot}. "
            f"Video: {video_clip}.\n"
        )

    return text