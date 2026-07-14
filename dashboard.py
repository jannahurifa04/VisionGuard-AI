from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import sqlite3
from datetime import datetime
import cv2
import requests
import numpy as np
import time
import os
import re
import psutil
import shutil
from rag_engine import build_rag_context
from watchlist import add_watchlist, get_watchlist, is_watchlisted
from memory import save_incident_memory, search_memory, get_person_profile, get_person_timeline

app = FastAPI()

CAMERA_URL = "http://192.168.18.152:8080/shot.jpg"
ROTATE_FRAME = True
SNAPSHOT_FOLDER = "snapshots"
CLIP_FOLDER = "clips"

def check_database_status():
    try:
        conn = sqlite3.connect("events.db")
        conn.execute("SELECT 1")
        conn.close()
        return "Online"
    except Exception:
        return "Offline"

def check_camera_status():
    try:
        response = requests.get(CAMERA_URL, timeout=2)
        if response.status_code == 200 and len(response.content) > 1000:
            return "Online"
        return "Offline"
    except Exception:
        return "Offline"

def check_storage_status():
    total, used, free = shutil.disk_usage("/")
    percent = int((used / total) * 100)
    return percent

def check_ai_engine():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return "Online"
        return "Offline"
    except Exception:
        return "Offline"

def ensure_chat_tables():
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        role TEXT,
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def create_chat_session(title="New Chat"):
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO chat_sessions (title) VALUES (?)",
        (title,)
    )

    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id


def save_chat_message(session_id, role, message):
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_messages (session_id, role, message)
        VALUES (?, ?, ?)
    """, (session_id, role, message))

    conn.commit()
    conn.close()


def get_chat_sessions():
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title
        FROM chat_sessions
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows

def check_yolo_status():
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = " ".join(proc.info['cmdline'] or [])
            if "main_reid.py" in cmdline:
                return "Online"
        return "Offline"
    except Exception:
        return "Offline"

def check_memory_status():
    try:
        search_memory("health check", limit=1)
        return "Online"
    except Exception:
        return "Offline"

def safe_filename(filename: str):
    filename = os.path.basename(filename)
    if ".." in filename or "/" in filename or "\\" in filename:
        return None
    return filename


def action_result_page(title: str, message: str):
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #111827;
                color: white;
                margin: 0;
                padding: 40px;
                text-align: center;
            }}
            .box {{
                background: #1f2937;
                padding: 30px;
                border-radius: 14px;
                display: inline-block;
                max-width: 600px;
            }}
            .back-button {{
                background: #2563eb;
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                display: inline-block;
                margin-top: 20px;
            }}

        </style>
    </head>
    <body>
        <div class="box">
            <h1>{title}</h1>
            <p>{message}</p>
            <a class="back-button" href="/">← Back to Dashboard</a>
        </div>
    </body>
    </html>
    """)


@app.get("/snapshot_file/{filename}")
def get_snapshot_file(filename: str):
    filepath = os.path.join(SNAPSHOT_FOLDER, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return HTMLResponse("Snapshot file not found", status_code=404)


@app.get("/clip_file/{filename}")
def get_clip_file(filename: str):
    filepath = os.path.join(CLIP_FOLDER, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4")
    return HTMLResponse("Video clip not found", status_code=404)



@app.get("/save_snapshot/{filename}", response_class=HTMLResponse)
def save_snapshot_evidence(filename: str):
    filename = safe_filename(filename)
    if not filename:
        return HTMLResponse("Invalid filename", status_code=400)

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE events ADD COLUMN saved_evidence INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    cursor.execute(
        "UPDATE events SET saved_evidence = 1 WHERE snapshot = ?",
        (filename,)
    )
    conn.commit()
    conn.close()

    return action_result_page("💾 Evidence Saved", f"Snapshot evidence saved: {filename}")


@app.get("/save_clip/{filename}", response_class=HTMLResponse)
def save_clip_evidence(filename: str):
    filename = safe_filename(filename)
    if not filename:
        return HTMLResponse("Invalid filename", status_code=400)

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE events ADD COLUMN saved_evidence INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    cursor.execute(
        "UPDATE events SET saved_evidence = 1 WHERE video_clip = ?",
        (filename,)
    )
    conn.commit()
    conn.close()

    return action_result_page("💾 Evidence Saved", f"Video evidence saved: {filename}")


@app.get("/delete_snapshot/{filename}", response_class=HTMLResponse)
def delete_snapshot_evidence(filename: str):
    filename = safe_filename(filename)
    if not filename:
        return HTMLResponse("Invalid filename", status_code=400)

    filepath = os.path.join(SNAPSHOT_FOLDER, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE events SET snapshot = NULL WHERE snapshot = ?",
        (filename,)
    )
    conn.commit()
    conn.close()

    return action_result_page("🗑 Evidence Deleted", f"Snapshot evidence deleted: {filename}")


@app.get("/delete_clip/{filename}", response_class=HTMLResponse)
def delete_clip_evidence(filename: str):
    filename = safe_filename(filename)
    if not filename:
        return HTMLResponse("Invalid filename", status_code=400)

    filepath = os.path.join(CLIP_FOLDER, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE events SET video_clip = NULL WHERE video_clip = ?",
        (filename,)
    )
    conn.commit()
    conn.close()

    return action_result_page("🗑 Evidence Deleted", f"Video evidence deleted: {filename}")


@app.get("/clip/{filename}", response_class=HTMLResponse)
def view_clip(filename: str):
    filepath = os.path.join(CLIP_FOLDER, filename)
    if not os.path.exists(filepath):
        return HTMLResponse("Video clip not found", status_code=404)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Evidence</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #111827;
                margin: 0;
                padding: 30px;
                color: white;
                text-align: center;
            }}

            .top-bar {{
                text-align: left;
                margin-bottom: 30px;
            }}

            .back-button {{
                background: #2563eb;
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                display: inline-block;
                margin-right: 10px;
            }}

            .save-button {{
                background: #16a34a;
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                display: inline-block;
                margin-right: 10px;
            }}

            .delete-button {{
                background: #dc2626;
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                display: inline-block;
            }}

            .video-box {{
                background: #1f2937;
                padding: 20px;
                border-radius: 12px;
                display: inline-block;
            }}

            video {{
                max-width: 90vw;
                max-height: 75vh;
                border-radius: 10px;
                border: 4px solid #374151;
                background: black;
            }}

            .filename {{
                margin-top: 15px;
                color: #d1d5db;
                font-size: 14px;
                word-break: break-all;
            }}
        </style>
    </head>

    <body>
        <div class="top-bar">
            <a class="back-button" href="/">← Back to Dashboard</a>
            <a class="save-button" href="/save_clip/{filename}">💾 Save Evidence</a>
            <a class="delete-button" href="/delete_clip/{filename}" onclick="return confirm('Delete this video evidence?');">🗑 Delete Evidence</a>
        </div>

        <h1>🎞️ Video Evidence</h1>

        <div class="video-box">
            <video controls autoplay muted>
                <source src="/clip_file/{filename}" type="video/mp4">
                Your browser does not support video playback.
            </video>
            <div class="filename">{filename}</div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


@app.get("/snapshot/{filename}", response_class=HTMLResponse)
def view_snapshot(filename: str):
    filepath = os.path.join(SNAPSHOT_FOLDER, filename)
    if not os.path.exists(filepath):
        return HTMLResponse("Snapshot not found", status_code=404)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Snapshot Evidence</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #111827;
                margin: 0;
                padding: 30px;
                color: white;
                text-align: center;
            }}

            .top-bar {{
                text-align: left;
                margin-bottom: 30px;
            }}

            .back-button {{
                background: #2563eb;
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                display: inline-block;
                margin-right: 10px;
            }}

            .save-button {{
                background: #16a34a;
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                display: inline-block;
                margin-right: 10px;
            }}

            .delete-button {{
                background: #dc2626;
                color: white;
                padding: 12px 18px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
                display: inline-block;
            }}

            .snapshot-box {{
                background: #1f2937;
                padding: 20px;
                border-radius: 12px;
                display: inline-block;
            }}

            img {{
                max-width: 90vw;
                max-height: 75vh;
                border-radius: 10px;
                border: 4px solid #374151;
            }}

            .filename {{
                margin-top: 15px;
                color: #d1d5db;
                font-size: 14px;
                word-break: break-all;
            }}
        </style>
    </head>

    <body>
        <div class="top-bar">
            <a class="back-button" href="/">← Back to Dashboard</a>
            <a class="save-button" href="/save_snapshot/{filename}">💾 Save Evidence</a>
            <a class="delete-button" href="/delete_snapshot/{filename}" onclick="return confirm('Delete this snapshot evidence?');">🗑 Delete Evidence</a>
        </div>

        <h1>📸 Snapshot Evidence</h1>

        <div class="snapshot-box">
            <img src="/snapshot_file/{filename}">
            <div class="filename">{filename}</div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)


def event_badge(event_name):
    if event_name == "PERSON_ENTERED":
        return "<span class='badge green'>🟢 PERSON_ENTERED</span>"
    elif event_name == "PERSON_PRESENT":
        return "<span class='badge blue'>🔵 PERSON_PRESENT</span>"
    elif event_name == "LOITERING":
        return "<span class='badge orange'>🟠 LOITERING</span>"
    elif event_name == "PERSON_LEFT":
        return "<span class='badge red'>🔴 PERSON_LEFT</span>"
    else:
        return f"<span class='badge gray'>{event_name}</span>"


def evidence_link(snapshot, video_clip=None):
    links = []

    if snapshot:
        links.append(f"<a class='view-link' href='/snapshot/{snapshot}'>View Snapshot</a>")

    if video_clip:
        links.append(f"<a class='video-link' href='/clip/{video_clip}'>View Video</a>")

    if links:
        return " ".join(links)

    return "-"


def person_label(person_id):
    if person_id is None:
        return "-"
    return f"Person ID {person_id}"


def shirt_label(shirt_color):
    if not shirt_color:
        return "-"
    return f"{shirt_color} Shirt"


def event_text_with_person(event_name, person_id):
    if person_id is None:
        return event_name
    return f"{event_name} - Person ID {person_id}"


def event_text_with_person_and_shirt(event_name, person_id, shirt_color):
    if person_id is None:
        text = event_name
    else:
        text = f"{event_name} - Person ID {person_id}"

    if shirt_color:
        text += f" - {shirt_color} Shirt"

    return text


def build_fast_person_profile(person_id):
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        WHERE person_id = ?
        ORDER BY event_time ASC
    """, (person_id,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return f"No records found for Person ID {person_id}."

    total_events = len(rows)
    entered_count = sum(1 for r in rows if r[1] == "PERSON_ENTERED")
    loitering_count = sum(1 for r in rows if r[1] == "LOITERING")
    present_count = sum(1 for r in rows if r[1] == "PERSON_PRESENT")
    left_count = sum(1 for r in rows if r[1] == "PERSON_LEFT")
    evidence_count = sum(1 for r in rows if r[2] or r[3])

    shirts = [r[5] for r in rows if r[5]]
    if shirts:
        most_common_shirt = max(set(shirts), key=shirts.count) + " Shirt"
    else:
        most_common_shirt = "Unknown"

    first_seen = rows[0][0]
    last_seen = rows[-1][0]

    risk_note = "Low"
    if loitering_count >= 3:
        risk_note = "Medium"
    if loitering_count >= 5:
        risk_note = "High"

    html = f"""
<b>Person ID {person_id} Profile</b><br><br>
<b>Total Recorded Events:</b> {total_events}<br>
<b>Entry Events:</b> {entered_count}<br>
<b>Loitering Events:</b> {loitering_count}<br>
<b>Present Events:</b> {present_count}<br>
<b>Left Events:</b> {left_count}<br>
<b>Most Common Shirt:</b> {most_common_shirt}<br>
<b>First Seen:</b> {first_seen}<br>
<b>Last Seen:</b> {last_seen}<br>
<b>Evidence Records:</b> {evidence_count}<br>
<b>Risk Level:</b> {risk_note}<br><br>
"""

    if loitering_count > 0:
        html += "<b>Note:</b> This person has loitering history. Review available snapshot and video evidence.<br>"

    return html

def person_loitering_count(person_id):

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*)
        FROM events
        WHERE person_id = ?
        AND object_name = 'LOITERING'
    """, (person_id,))

    count = cursor.fetchone()[0]

    conn.close()

    return count
def get_highest_risk_person():

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT person_id,
               COUNT(*) as total_events,
               SUM(CASE WHEN object_name = 'LOITERING' THEN 1 ELSE 0 END) as loitering_count
        FROM events
        WHERE person_id IS NOT NULL
        GROUP BY person_id
        ORDER BY loitering_count DESC, total_events DESC
        LIMIT 1
    """)

    result = cursor.fetchone()
    conn.close()

    if not result:
        return "No person records found."

    person_id, total_events, loitering_count = result

    risk_score = (loitering_count * 10) + total_events

    risk_level = "Low"
    if risk_score >= 20:
        risk_level = "Medium"
    if risk_score >= 40:
        risk_level = "High"

    return f"""
<b>Highest Risk Person</b><br><br>
<b>Person ID:</b> {person_id}<br>
<b>Total Events:</b> {total_events}<br>
<b>Loitering Events:</b> {loitering_count}<br>
<b>Risk Score:</b> {risk_score}/100<br>
<b>Risk Level:</b> {risk_level}<br><br>
<b>Reason:</b> This person has the highest loitering activity in the CCTV database.
"""

def get_top_suspicious_persons():

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT person_id,
               COUNT(*) as total_events,
               SUM(CASE WHEN object_name='LOITERING' THEN 1 ELSE 0 END) as loitering_count
        FROM events
        WHERE person_id IS NOT NULL
        GROUP BY person_id
        ORDER BY loitering_count DESC, total_events DESC
        LIMIT 5
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No suspicious persons found."

    html = "<b>Top 5 Suspicious Persons</b><br><br>"

    rank = 1

    for person_id, total_events, loitering_count in rows:

        risk_score = (loitering_count * 10) + total_events

        html += f"""
        <b>#{rank} Person ID {person_id}</b><br>
        Total Events: {total_events}<br>
        Loitering Events: {loitering_count}<br>
        Risk Score: {risk_score}/100<br><br>
        """

        rank += 1

    return html

def build_person_timeline(person_id, only_today=False):
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    params = [person_id]
    date_filter = ""

    if only_today:
        today = datetime.now().strftime("%Y-%m-%d")
        date_filter = "AND event_time LIKE ?"
        params.append(today + "%")

    cursor.execute(f"""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        WHERE person_id = ?
        {date_filter}
        ORDER BY event_time ASC
        LIMIT 100
    """, params)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        if only_today:
            return f"No events found today for Person ID {person_id}."
        return f"No events found for Person ID {person_id}."

    entered_count = sum(1 for r in rows if r[1] == "PERSON_ENTERED")
    loitering_count = sum(1 for r in rows if r[1] == "LOITERING")
    present_count = sum(1 for r in rows if r[1] == "PERSON_PRESENT")
    left_count = sum(1 for r in rows if r[1] == "PERSON_LEFT")
    evidence_count = sum(1 for r in rows if r[2] or r[3])

    shirt_colors = []
    for row in rows:
        if row[5] and row[5] not in shirt_colors:
            shirt_colors.append(row[5])

    shirt_summary = ", ".join([f"{color} Shirt" for color in shirt_colors]) if shirt_colors else "Unknown"

    timeline_html = f"""
<b>Person ID {person_id} Timeline Report</b><br><br>
<b>Total Events:</b> {len(rows)}<br>
<b>Entry Events:</b> {entered_count}<br>
<b>Loitering Events:</b> {loitering_count}<br>
<b>Present Events:</b> {present_count}<br>
<b>Left Events:</b> {left_count}<br>
<b>Shirt Color Observed:</b> {shirt_summary}<br>
<b>Evidence Records:</b> {evidence_count}<br><br>
<b>Timeline:</b><br>
"""

    for event_time, event_name, snapshot, video_clip, row_person_id, shirt_color, camera_name in rows:
        timeline_html += f"""
<div style="margin-bottom:10px;padding:10px;border-left:4px solid #2563eb;background:#f9fafb;border-radius:6px;">
<b>{event_time}</b><br>
{event_text_with_person_and_shirt(event_name, row_person_id, shirt_color)}<br>
{evidence_link(snapshot, video_clip)}
</div>
"""

    if loitering_count > 0:
        timeline_html += """
<br><b>Risk Note:</b><br>
This person has one or more loitering events. Review available snapshot and video evidence.<br>
"""

    return timeline_html


def ensure_video_clip_column():
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE events ADD COLUMN video_clip TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE events ADD COLUMN person_id INTEGER")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE events ADD COLUMN saved_evidence INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE events ADD COLUMN shirt_color TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    conn.close()

def build_evidence_report(event_time, event_name, snapshot, video_clip=None, person_id=None, shirt_color=None):
    risk_level = "Low"
    evidence_text = f"The VisionGuard system recorded a {event_name} event."
    recommendation = "Review the event in the dashboard if further verification is needed."

    if event_name == "LOITERING":
        risk_level = "Medium"
        evidence_text = "The system recorded a LOITERING event where a person remained in the monitored area beyond the configured threshold."
        recommendation = "Review the snapshot evidence and verify whether the activity was authorized."
    elif event_name == "PERSON_ENTERED":
        evidence_text = "The system recorded that a person entered the monitored area."
    elif event_name == "PERSON_PRESENT":
        evidence_text = "The system recorded that a person was present in the monitored area."
    elif event_name == "PERSON_LEFT":
        evidence_text = "The system recorded that the person left the monitored area."

    report = f"""
<b>Evidence Report</b><br><br>
<b>Event Type:</b> {event_name}<br>
<b>Person:</b> {person_label(person_id)}<br>
<b>Shirt Color:</b> {shirt_label(shirt_color)}<br>
<b>Time:</b> {event_time}<br><br>
<b>Evidence:</b><br>
{evidence_text}<br><br>
<b>Risk Level:</b> {risk_level}<br><br>
<b>Recommendation:</b><br>
{recommendation}<br>
"""

    if snapshot:
        report += (
            f"<br><b>Snapshot Evidence:</b><br>"
            f"<img src='/snapshot_file/{snapshot}' "
            f"style='max-width:350px;border-radius:10px;margin-top:10px;'><br><br>"
            f"<a class='view-link' href='/snapshot/{snapshot}'>View Snapshot</a><br>"
        )

    if video_clip:
        report += (
            f"<br><b>Video Evidence:</b><br>"
            f"<a class='video-link' href='/clip/{video_clip}'>View Video</a><br>"
        )

    return report

def build_incident_timeline():
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        ORDER BY id DESC
        LIMIT 10
    """)

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No CCTV events recorded yet."

    # Oldest first for natural timeline reading
    events = list(reversed(rows))

    timeline_html = "<b>Incident Timeline</b><br><br>"

    has_entered = False
    has_loitering = False
    has_present = False
    has_left = False

    for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in events:
        timeline_html += f"{event_time} - {event_text_with_person_and_shirt(event_name, person_id, shirt_color)}"

        evidence = evidence_link(snapshot, video_clip)
        if evidence != "-":
            timeline_html += f" - {evidence}"

        timeline_html += "<br>"

        if event_name == "PERSON_ENTERED":
            has_entered = True
        elif event_name == "LOITERING":
            has_loitering = True
        elif event_name == "PERSON_PRESENT":
            has_present = True
        elif event_name == "PERSON_LEFT":
            has_left = True

    summary_parts = []

    if has_entered:
        summary_parts.append("a person entered the monitored area")
    if has_present:
        summary_parts.append("the system confirmed the person was present")
    if has_loitering:
        summary_parts.append("a loitering event was triggered")
    if has_left:
        summary_parts.append("the person later left the monitored area")

    if summary_parts:
        summary = "The timeline shows that " + ", then ".join(summary_parts) + "."
    else:
        summary = "The timeline shows recent CCTV activity, but there is not enough information to reconstruct a full incident."

    timeline_html += f"<br><b>AI Summary:</b><br>{summary}<br>"

    return timeline_html


def build_watchlist_html():
    rows = get_watchlist()

    if not rows:
        return "Watchlist is empty."

    html = "<b>VisionGuard Watchlist</b><br><br>"

    for person_id, reason, risk_level, created_at in rows:
        html += f"""
<div style="margin-bottom:10px;padding:10px;border-left:4px solid #ef4444;background:#1f2937;color:white;border-radius:6px;">
<b>Person ID {person_id}</b><br>
Reason: {reason}<br>
Risk Level: {risk_level}<br>
Added: {created_at}
</div>
"""

    return html


def build_watchlist_alert_html():
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT person_id, event_time, object_name
        FROM events
        WHERE person_id IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
    """)

    latest = cursor.fetchone()
    conn.close()

    if not latest:
        return ""

    person_id, event_time, event_name = latest
    watch = is_watchlisted(person_id)

    if not watch:
        return ""

    reason, risk_level = watch

    return f"""
<div class="watchlist-alert">
    🚨 <b>WATCHLIST ALERT</b><br>
    Person ID {person_id} detected<br>
    Event: {event_name}<br>
    Time: {event_time}<br>
    Reason: {reason}<br>
    Risk Level: {risk_level}
</div>
"""

def build_cctv_context(limit=50):
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No CCTV events recorded yet."

    context = ""

    for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in rows:
        context += (
            f"Time: {event_time}, "
            f"Event: {event_name}, "
            f"Person ID: {person_id}, "
            f"Shirt: {shirt_color}, "
            f"Snapshot: {snapshot}, "
            f"Video: {video_clip}\n"
        )

    return context

from visionguard_llm import ask_rag_llm

def ask_visionguard(question):
    ensure_video_clip_column()

    today = datetime.now().strftime("%Y-%m-%d")
    question = question.strip()
    question_lower = question.lower()

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM events WHERE event_time LIKE ?", (today + "%",))
    total_today = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM events
        WHERE object_name = 'PERSON_ENTERED'
        AND event_time LIKE ?
    """, (today + "%",))
    visitor_entries_today = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM events
        WHERE object_name = 'LOITERING'
        AND event_time LIKE ?
    """, (today + "%",))
    loitering_today = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM events
        WHERE object_name = 'PERSON_PRESENT'
        AND event_time LIKE ?
    """, (today + "%",))
    present_today = cursor.fetchone()[0]

    cursor.execute("""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        WHERE event_time LIKE ?
        ORDER BY id DESC
        LIMIT 10
    """, (today + "%",))
    recent_events = cursor.fetchall()

    cursor.execute("""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        ORDER BY id DESC
        LIMIT 1
    """)
    last_event = cursor.fetchone()

    cursor.execute("""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        WHERE object_name = 'LOITERING'
        ORDER BY id DESC
        LIMIT 1
    """)
    latest_loitering = cursor.fetchone()

    conn.close()

    recent_event_text = ""
    for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in recent_events:
        evidence = ""
        if snapshot:
            evidence += f" Snapshot: {snapshot}"
        if video_clip:
            evidence += f" Video: {video_clip}"
        recent_event_text += f"- {event_time}: {event_text_with_person_and_shirt(event_name, person_id, shirt_color)}.{evidence}\n"

    if not recent_event_text:
        recent_event_text = "No CCTV events recorded today."

    # Fast local profile and timeline queries
    person_match = re.search(r"person\s*id\s*(\d+)", question_lower)

    if "search memory" in question_lower:
        results = search_memory(question, limit=5)

        docs = results.get("documents", [[]])[0]

        if not docs:
            return "No memory found."

        answer = "<b>Memory Search Results</b><br><br>"

        for doc in docs:
            answer += f"- {doc.replace(chr(10), '<br>')}<br><br>"

        return answer
    
    if person_match and "memory timeline" in question_lower:
        requested_person_id = int(person_match.group(1))
        return get_person_timeline(requested_person_id).replace("\n", "<br>")

    if person_match and "loitered before" in question_lower:

        requested_person_id = int(person_match.group(1))

        count = person_loitering_count(requested_person_id)

        if count == 0:
            return f"Person ID {requested_person_id} has never been recorded loitering."

        return (
            f"Yes. Person ID {requested_person_id} has been recorded "
            f"loitering {count} time(s)."
       )

    if person_match and (
        "memory profile" in question_lower
        or "person profile" in question_lower
        or "profile memory" in question_lower
    ):
        requested_person_id = int(person_match.group(1))
        return get_person_profile(requested_person_id).replace("\n", "<br>")
    
    if person_match and "memory" in question_lower:
        requested_person_id = int(person_match.group(1))
        return get_person_profile(requested_person_id).replace("\n", "<br>")
        # Watchlist commands
    if "show watchlist" in question_lower or "view watchlist" in question_lower:
        return build_watchlist_html()

    if "highest risk person" in question_lower:
        return get_highest_risk_person()

    if "top suspicious persons" in question_lower:
        return get_top_suspicious_persons()

    if "top 5 suspicious persons" in question_lower:
        return get_top_suspicious_persons()

    if "show suspicious ranking" in question_lower:
        return get_top_suspicious_persons()

    if "most suspicious person" in question_lower:
        return get_highest_risk_person()

    if "who should be investigated first" in question_lower:
        return get_highest_risk_person()

    if person_match and "add" in question_lower and "watchlist" in question_lower:
        requested_person_id = int(person_match.group(1))
        add_watchlist(requested_person_id, "Repeated Loitering", "Medium")
        return f"Person ID {requested_person_id} has been added to the watchlist.<br>Reason: Repeated Loitering<br>Risk Level: Medium"

    if person_match and "watchlist" in question_lower and (
        "is" in question_lower or "check" in question_lower
    ):
        requested_person_id = int(person_match.group(1))
        watch = is_watchlisted(requested_person_id)

        if watch:
            reason, risk_level = watch
            return f"Yes. Person ID {requested_person_id} is on the watchlist.<br>Reason: {reason}<br>Risk Level: {risk_level}"

        return f"No. Person ID {requested_person_id} is not on the watchlist."

    if person_match and (
        "tell me about" in question_lower
        or "profile" in question_lower
        or "who is" in question_lower
    ):
        requested_person_id = int(person_match.group(1))
        return build_fast_person_profile(requested_person_id)

    if person_match and (
        "full history" in question_lower
        or "history" in question_lower
        or "timeline" in question_lower
        or "what did" in question_lower
        or "activity" in question_lower
        or "do today" in question_lower
        or "did today" in question_lower
    ):
        requested_person_id = int(person_match.group(1))
        only_today = "today" in question_lower
        return build_person_timeline(requested_person_id, only_today=only_today)

    if person_match and (
        "how many times" in question_lower
        or "appear this week" in question_lower
        or "appeared this week" in question_lower
    ):
        requested_person_id = int(person_match.group(1))
        conn = sqlite3.connect("events.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE person_id = ?
            AND event_time >= date('now', '-7 days')
        """, (requested_person_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return f"Person ID {requested_person_id} appeared in {count} recorded event(s) during the last 7 days."

    last_event_text = "No events recorded yet."
    if last_event:
        last_time, last_name, last_snapshot, last_video, last_person_id, last_shirt_color, camera_name = last_event
        last_event_text = f"{event_text_with_person_and_shirt(last_name, last_person_id, last_shirt_color)} at {last_time}"
        if last_snapshot:
            last_event_text += f" with evidence snapshot {last_snapshot}"
        if last_video:
            last_event_text += f" and video clip {last_video}"

    # Timeline AI v1: reconstruct latest incident using latest 10 records
    if (
        "incident timeline" in question_lower
        or "explain latest timeline" in question_lower
        or "reconstruct latest incident" in question_lower
        or "reconstruct incident" in question_lower
    ):
        return build_incident_timeline()

    # Evidence AI v1: latest incident/evidence, ignores today's date
    if (
        "latest evidence" in question_lower
        or "latest incident" in question_lower
        or "summarize latest incident" in question_lower
        or "explain latest incident" in question_lower
    ):
        if not last_event:
            return "No CCTV incidents have been recorded yet."

        last_time, last_name, last_snapshot, last_video, last_person_id, last_shirt_color, camera_name = last_event
        return build_evidence_report(last_time, last_name, last_snapshot, last_video, last_person_id, last_shirt_color)

    # Evidence AI v1: latest loitering, ignores today's date
    if (
        "explain latest loitering" in question_lower
        or "latest loitering evidence" in question_lower
        or "show latest loitering evidence" in question_lower
    ):
        if not latest_loitering:
            return "No loitering incidents have been recorded yet."

        loiter_time, loiter_name, loiter_snapshot, loiter_video, loiter_person_id, loiter_shirt_color, camera_name = latest_loitering
        return build_evidence_report(loiter_time, loiter_name, loiter_snapshot, loiter_video, loiter_person_id, loiter_shirt_color)

        if (
            "daily security report" in question_lower
            or "today security report" in question_lower
            or "generate today security report" in question_lower
            or "security report today" in question_lower
        ):
            return f"""
<b>📋 VisionGuard Daily Security Report</b><br><br>

<b>Date:</b> {today}<br><br>

<b>Summary:</b><br>
- Total CCTV Events Today: {total_today}<br>
- Visitor Entry Events: {visitor_entries_today}<br>
- Person Present Events: {present_today}<br>
- Loitering Incidents: {loitering_today}<br><br>

<b>Latest Event:</b><br>
{last_event_text}<br><br>

<b>Risk Assessment:</b><br>
{"Medium risk. Loitering activity was detected and should be reviewed." if loitering_today > 0 else "Low risk. No loitering activity was detected today."}<br><br>

<b>Recommendation:</b><br>
{"Review latest evidence and check saved snapshots/videos." if loitering_today > 0 else "Continue monitoring. No urgent action needed."}
"""
    
    # Who loitered most
    if (
        "who loitered most" in question_lower
        or "highest loitering" in question_lower
        or "most loitering" in question_lower
    ):

        conn = sqlite3.connect("events.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT person_id, COUNT(*) as total
            FROM events
            WHERE object_name = 'LOITERING'
            AND person_id IS NOT NULL
            GROUP BY person_id
            ORDER BY total DESC
            LIMIT 1
        """)

        result = cursor.fetchone()
        conn.close()

        if not result:
            return "No loitering incidents have been recorded yet."

        person_id, total = result

        risk = "Low"
        if total >= 3:
            risk = "Medium"
        if total >= 5:
            risk = "High"

        return f"""
    <b>Highest Loitering Person</b><br><br>
    <b>Person ID:</b> {person_id}<br>
    <b>Loitering Incidents:</b> {total}<br>
    <b>Risk Level:</b> {risk}
    """
  
    

    # Exact visitor answer
    if "visitor" in question_lower and (
        "how many" in question_lower
        or "berapa" in question_lower
        or "jumlah" in question_lower
    ):
        return f"There have been {visitor_entries_today} visitor entry events today."

    # Yesterday loitering
    if "loitering" in question_lower and "yesterday" in question_lower:

        conn = sqlite3.connect("events.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
            FROM events
            WHERE object_name = 'LOITERING'
            AND date(event_time) = date('now', '-1 day')
            ORDER BY id DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No loitering incidents were recorded yesterday."

        html = f"Yes. There were {len(rows)} loitering incident(s) yesterday.<br><br>"

        for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in rows:
            html += f"{event_time} - {event_text_with_person_and_shirt(event_name, person_id, shirt_color)}<br>"
            html += evidence_link(snapshot, video_clip)
            html += "<br><br>"
 
        return html

    # Today's loitering summary
    if "loitering" in question_lower:
        if loitering_today == 0:
            return "No loitering incidents have been recorded today."

        evidence_html = ""

        for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in recent_events:
            if event_name == "LOITERING":
                evidence_html += f"{event_time} - {event_text_with_person_and_shirt(event_name, person_id, shirt_color)}<br>"
                evidence_html += evidence_link(snapshot, video_clip)
                evidence_html += "<br><br>"

        if evidence_html:
            return (
                f"Yes. There have been {loitering_today} loitering incident(s) today.<br><br>"
                f"Evidence:<br>{evidence_html}"
            )

        return f"Yes. There have been {loitering_today} loitering incident(s) today, but no snapshot evidence was found."

    # Latest event
    if "latest event" in question_lower or "last event" in question_lower:
        return f"The latest CCTV event is: {last_event_text}"

    # Latest timeline / recent events: ignores today's date and shows latest 10 records
    if (
        "latest timeline" in question_lower
        or "recent events" in question_lower
        or "last 10 events" in question_lower
        or "latest events" in question_lower
    ):
        conn = sqlite3.connect("events.db")
        cursor = conn.cursor()

        cursor.execute("""
            SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
            FROM events
            ORDER BY id DESC
            LIMIT 10
        """)

        timeline_events = cursor.fetchall()
        conn.close()

        if not timeline_events:
            return "No CCTV events recorded yet."

        timeline_html = "<b>Latest CCTV Timeline</b><br><br>"

        for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in timeline_events:
            timeline_html += f"{event_time} - {event_text_with_person_and_shirt(event_name, person_id, shirt_color)}"

            evidence = evidence_link(snapshot, video_clip)
            if evidence != "-":
                timeline_html += f" - {evidence}"

            timeline_html += "<br>"

        return timeline_html

    # Today's timeline only
    if "timeline" in question_lower or "activity list" in question_lower or "event list" in question_lower:
        if not recent_events:
            return "No CCTV events recorded today."

        timeline_html = "<b>Today CCTV Timeline</b><br><br>"

        for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in reversed(recent_events):
            timeline_html += f"{event_time} - {event_text_with_person_and_shirt(event_name, person_id, shirt_color)}"

            evidence = evidence_link(snapshot, video_clip)
            if evidence != "-":
                timeline_html += f" - {evidence}"

            timeline_html += "<br>"

        return timeline_html

    if "what happened today" in question_lower:
        return (
            f"Today's CCTV activity:<br>"
            f"- Visitor entry events: {visitor_entries_today}<br>"
            f"- Loitering incidents: {loitering_today}<br>"
            f"- Person present events: {present_today}<br>"
            f"- Total events: {total_today}<br>"
            f"- Latest event: {last_event_text}"
        )

    cctv_context = build_cctv_context(limit=50)

    memory_results = search_memory(question, limit=5)
    memory_docs = memory_results.get("documents", [[]])[0]

    memory_context = ""

    for doc in memory_docs:
        memory_context += doc + "\n"

    prompt = f"""
You are VisionGuard AI, an intelligent CCTV security assistant.

Rules:
- Use ONLY the CCTV data provided below.
- Do NOT guess or infer beyond the event records.
- Do NOT invent people, names, identities, genders, locations, or events.
- If Person ID is None, say "unknown person".
- Do not say "different locations" unless location data is provided.
- Use exact event time, event type, Person ID, shirt color, snapshot, and video fields.
- Prefer factual bullet points.
- If asked "what happened", summarize the latest CCTV events in chronological order.
- If the data is insufficient, say: "The CCTV records do not contain enough detail to answer that."
- Answer in the same language as the user's question.

Today's Date:
{today}

CCTV Database Context:
{cctv_context}

Relevant Memory:
{memory_context}

Recent CCTV Events:
{recent_event_text}

User Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        data = response.json()
        return data.get("response", "No response from Ollama.")

    except Exception as e:
        return f"Ollama error: {e}"




def generate_camera_frames():
    while True:
        try:
            response = requests.get(CAMERA_URL, timeout=2)
            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            if ROTATE_FRAME:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            ret, buffer = cv2.imencode(".jpg", frame)

            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            time.sleep(0.1)

        except Exception as e:
            print("Camera stream error:", e)
            time.sleep(1)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/live-cameras", response_class=HTMLResponse)
def live_cameras():
    ensure_video_clip_column()

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        ORDER BY id DESC
        LIMIT 10
    """)
    recent_events = cursor.fetchall()

    cursor.execute("""
        SELECT event_time, object_name, person_id, shirt_color
        FROM events
        WHERE person_id IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
    """)
    latest_person = cursor.fetchone()

    conn.close()

    if latest_person:
        last_seen, last_event, person_id, shirt_color = latest_person
        person_html = f"""
        <h3>Person ID {person_id}</h3>
        <p>Shirt: {shirt_color or "Unknown"}</p>
        <p>Last Event: {last_event}</p>
        <p>Last Seen: {last_seen}</p>
        """
    else:
        person_html = "<p>No person detected yet.</p>"

    event_rows = ""
    for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in recent_events:
        event_rows += f"""
        <tr>
            <td>{event_time}</td>
            <td>{event_badge(event_name)}</td>
            <td>{person_label(person_id)}</td>
            <td>{shirt_label(shirt_color)}</td>
            <td>{evidence_link(snapshot, video_clip)}</td>
        </tr>
        """

    if not event_rows:
        event_rows = """
        <tr>
            <td colspan="4">No recent events found.</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VisionGuard AI - Live Cameras</title>
        <style>
            body {{
                margin: 0;
                background: #070b18;
                color: white;
                font-family: Arial, sans-serif;
            }}

            .page {{
                padding: 24px;
            }}

            .top {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }}

            .back {{
                background: #6d4cff;
                color: white;
                padding: 10px 16px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: bold;
            }}

            .grid {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 18px;
            }}

            .card {{
                background: #111827;
                border: 1px solid #253047;
                border-radius: 14px;
                padding: 16px;
            }}

            .camera {{
                width: 100%;
                max-height: 650px;
                object-fit: contain;
                background: black;
                border-radius: 12px;
                border: 2px solid #6d4cff;
            }}

            .status {{
                color: #22c55e;
                font-weight: bold;
            }}

            .small-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 18px;
                margin-top: 18px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}

            th {{
                background: #4f46e5;
                padding: 10px;
                text-align: left;
            }}

            td {{
                padding: 10px;
                border-bottom: 1px solid #253047;
            }}

            .muted {{
                color: #9ca3af;
            }}
        </style>
    </head>

    <body>
        <div class="page">
            <div class="top">
                <div>
                    <h1>📹 Live Cameras</h1>
                    <p class="muted">Real-time CCTV monitoring screen</p>
                </div>
                <a class="back" href="/">← Back to Dashboard</a>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>CAM 01 - Main Entrance</h2>
                    <img class="camera" src="/video_feed">
                </div>

                <div class="card">
                    <h2>Camera Status</h2>
                    <p>Status: <span class="status">Online</span></p>
                    <p>Camera: CAM 01</p>
                    <p>Location: Main Entrance</p>
                    <p>Source: IP Webcam</p>
                    <p>Stream: /video_feed</p>
                </div>
            </div>

            <div class="small-grid">
                <div class="card">
                    <h2>Current Person Tracking</h2>
                    {person_html}
                </div>

                <div class="card">
                    <h2>Watchlist / Risk</h2>
                    <p class="muted">Watchlist alerts will appear here.</p>
                </div>
            </div>

            <div class="card" style="margin-top:18px;">
                <h2>Recent Camera Events</h2>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Event</th>
                        <th>Person ID</th>
                        <th>Shirt Color</th>
                        <th>Camera</th>

                        <th>Evidence</th>
                    </tr>
                    {event_rows}
                </table>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(html)

@app.get("/persons", response_class=HTMLResponse)
def persons_page():
    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT person_id, display_name, status, first_seen, last_seen,
                visit_count, last_shirt_color, last_camera_name, risk_level,
                avg_visit_seconds, longest_visit_seconds, loiter_count, last_loiter
        FROM persons
        ORDER BY last_seen DESC
    """)
    persons = cursor.fetchall()
    
    print(persons)

    if persons:
        print(len(persons[0]))
        print(persons[0])

    cards_html = ""

    for (
    person_id,
    display_name,
    status,
    first_seen,
    last_seen,
    visit_count,
    shirt,
    camera,
    risk,
    avg_visit_seconds,
    longest_visit_seconds,
    loiter_count,
    last_loiter
) in persons:
        name = display_name or "Unknown Person"

        cursor.execute("""
            SELECT snapshot
            FROM events
            WHERE person_id = ?
            AND snapshot IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
        """, (person_id,))
        snap = cursor.fetchone()

        image_src = f"/snapshot_file/{snap[0]}" if snap else "/video_feed"

        risk_class = "risk-low"
        if risk == "Medium":
            risk_class = "risk-medium"
        elif risk == "High":
            risk_class = "risk-high"

        cards_html += f"""
        <div class="person-card">
            <img src="{image_src}" class="person-img">
            <h2>Person ID {person_id}</h2>
            <p class="name">{name}</p>

            <div class="info"><span>Status</span><b>{status or "Unknown"}</b></div>
            <div class="info"><span>Visits</span><b>{visit_count}</b></div>
            <div class="info"><span>Shirt</span><b>{shirt or "Unknown"}</b></div>
            <div class="info"><span>Camera</span><b>{camera or "-"}</b></div>
            <div class="info"><span>First Seen</span><b>{first_seen or "-"}</b></div>
            <div class="info"><span>Last Seen</span><b>{last_seen or "-"}</b></div>
            <div class="info"><span>Avg Visit</span><b>{int(avg_visit_seconds)} sec</b></div>
            <div class="info"><span>Longest Visit</span><b>{int(longest_visit_seconds)} sec</b></div>
            <div class="info"><span>Loiter Count</span><b>{loiter_count}</b></div>
<div class="info"><span>Last Loiter</span><b>{last_loiter or "-"}</b></div>

            <div class="risk {risk_class}">{risk or "Low"} Risk</div>
            <a class="profile-btn" href="/?ask=tell me about person id {person_id}">Ask AI About This Person</a>
        </div>
        """

    conn.close()

    if not cards_html:
        cards_html = "<p>No person profiles found yet.</p>"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Person Intelligence</title>
        <style>
            body {{
                margin: 0;
                background: radial-gradient(circle at top left, #1b1240 0, #080d18 36%, #05070d 100%);
                color: white;
                font-family: Arial, sans-serif;
            }}

            .page {{
                padding: 30px;
            }}

            .top {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
            }}

            .back {{
                background: #6d4cff;
                color: white;
                padding: 11px 16px;
                border-radius: 10px;
                text-decoration: none;
                font-weight: bold;
            }}

            .subtitle {{
                color: #94a3b8;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                gap: 18px;
            }}

            .person-card {{
                background: rgba(15, 23, 42, .86);
                border: 1px solid rgba(148, 163, 184, .18);
                border-radius: 18px;
                padding: 16px;
                box-shadow: 0 14px 40px rgba(0,0,0,.35);
            }}

            .person-img {{
                width: 100%;
                height: 190px;
                object-fit: cover;
                border-radius: 14px;
                background: #020617;
                border: 1px solid rgba(148,163,184,.18);
            }}

            h1 {{
                margin: 0;
                font-size: 34px;
            }}

            h2 {{
                margin: 14px 0 4px;
            }}

            .name {{
                color: #a78bfa;
                margin-top: 0;
            }}

            .info {{
                display: flex;
                justify-content: space-between;
                padding: 9px 0;
                border-bottom: 1px solid rgba(148,163,184,.12);
                color: #cbd5e1;
                font-size: 14px;
            }}

            .info b {{
                color: white;
                text-align: right;
            }}

            .risk {{
                margin-top: 14px;
                display: inline-block;
                padding: 8px 12px;
                border-radius: 999px;
                font-weight: bold;
                font-size: 13px;
            }}

            .risk-low {{
                background: rgba(22,163,74,.18);
                color: #86efac;
            }}

            .risk-medium {{
                background: rgba(234,88,12,.18);
                color: #fdba74;
            }}

            .risk-high {{
                background: rgba(220,38,38,.2);
                color: #fca5a5;
            }}

            .profile-btn {{
                display: block;
                margin-top: 14px;
                text-align: center;
                background: linear-gradient(135deg,#7c3aed,#4f46e5);
                color: white;
                padding: 11px;
                border-radius: 10px;
                text-decoration: none;
                font-weight: bold;
            }}
        </style>
    </head>

    <body>
        <div class="page">
            <div class="top">
                <div>
                    <h1>👥 Person Intelligence</h1>
                    <p class="subtitle">VisionGuard AI remembers detected people and builds profiles over time.</p>
                </div>
                <a class="back" href="/">← Back to Dashboard</a>
            </div>

            <div class="grid">
                {cards_html}
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(html)

@app.get("/system-health", response_class=HTMLResponse)
def system_health():
    memory_status = check_memory_status()
    html = f"""
    <html>
    <head>
        <title>System Health</title>
        <style>
            body {{
                background:#111827;
                color:white;
                font-family:Arial;
                padding:30px;
            }}
            .card {{
                background:#1f2937;
                padding:25px;
                border-radius:16px;
                max-width:700px;
                margin:auto;
            }}
            .row {{
                display:flex;
                justify-content:space-between;
                padding:12px 0;
                border-bottom:1px solid #333;
            }}
            a {{
                color:#60a5fa;
                text-decoration:none;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>System Health</h1>
            <div class="row"><span>AI Engine</span><span>{check_ai_engine()}</span></div>
            <div class="row"><span>YOLO Engine</span><span>{check_yolo_status()}</span></div>
            <div class="row"><span>Memory Engine</span><span>{memory_status}</span></div>
            <div class="row"><span>Database</span><span>{check_database_status()}</span></div>
            <div class="row"><span>Cameras</span><span>{check_camera_status()}</span></div>
            <div class="row"><span>Storage</span><span>{check_storage_status()}%</span></div>
            <br>
            <a href="/">← Back to Dashboard</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/settings", response_class=HTMLResponse)
def settings():
    html = f"""
    <html>
    <head>
        <title>Settings</title>
        <style>
            body {{
                background:#111827;
                color:white;
                font-family:Arial;
                padding:30px;
            }}
            .card {{
                background:#1f2937;
                padding:25px;
                border-radius:16px;
                max-width:800px;
                margin:auto;
            }}
            .item {{
                display:block;
                padding:16px;
                margin:12px 0;
                background:#111827;
                border-radius:12px;
                color:white;
                text-decoration:none;
            }}
            a {{
                color:#60a5fa;
                text-decoration:none;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Settings</h1>

            <a class="item" href="/system-health">🩺 System Health</a>
            <div class="item">🤖 AI Engine Config (Coming Soon)</div>
            <div class="item">📹 Camera Settings (Coming Soon)</div>
            <div class="item">💾 Storage Management (Coming Soon)</div>

            <br>
            <a href="/">← Back to Dashboard</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.get("/", response_class=HTMLResponse)
def home(
    q: str = Query(default=""),
    start_date: str = Query(default=""),
    end_date: str = Query(default=""),
    ask: str = Query(default="")
):
    ensure_video_clip_column()
    ensure_chat_tables()

    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%I:%M:%S %p")
    now_date = datetime.now().strftime("%A, %B %d, %Y")

    search_text = q.strip()
    ai_question = ask.strip()
    session_id = create_chat_session(ai_question[:40] if ai_question else "New Chat")
    ai_answer = ""
    
    if ai_question:
        save_chat_message(session_id, "user", ai_question)
        ai_answer = ask_visionguard(ai_question)
        save_chat_message(session_id, "assistant", ai_answer)

    chat_sessions = get_chat_sessions()

    conn = sqlite3.connect("events.db")
    cursor = conn.cursor()

    where_parts = []
    params = []

    if search_text:
        where_parts.append("object_name LIKE ?")
        params.append(f"%{search_text}%")

    if start_date:
        where_parts.append("event_time >= ?")
        params.append(start_date + " 00:00:00")

    if end_date:
        where_parts.append("event_time <= ?")
        params.append(end_date + " 23:59:59")

    where_sql = ""
    if where_parts:
        where_sql = "WHERE " + " AND ".join(where_parts)

    limit_value = 50 if where_parts else 10

    cursor.execute(f"""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        {where_sql}
        ORDER BY id DESC
        LIMIT {limit_value}
    """, params)
    events = cursor.fetchall()

    cursor.execute("SELECT COUNT(*) FROM events")
    total_events = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM events
        WHERE object_name = 'PERSON_ENTERED'
        AND event_time LIKE ?
    """, (today + "%",))
    visitors_today = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM events
        WHERE object_name = 'LOITERING'
        AND event_time LIKE ?
    """, (today + "%",))
    loitering_today = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM events
        WHERE object_name = 'PERSON_PRESENT'
        AND event_time LIKE ?
    """, (today + "%",))
    present_today = cursor.fetchone()[0]

    cursor.execute("""
        SELECT object_name
        FROM events
        ORDER BY id DESC
        LIMIT 1
    """)
    last_event = cursor.fetchone()

    cursor.execute("""
        SELECT object_name, COUNT(*)
        FROM events
        GROUP BY object_name
        ORDER BY COUNT(*) DESC
    """)
    chart_data = cursor.fetchall()

    cursor.execute("""
        SELECT event_time, object_name, snapshot, video_clip, person_id, shirt_color, camera_name
        FROM events
        ORDER BY id DESC
        LIMIT 5
    """)
    recent_for_cards = cursor.fetchall()

    conn.close()

    active_alerts = loitering_today
    risk_level = "Medium" if loitering_today >= 1 else "Low"

    def time_only(value):
        try:
            return str(value).split(" ")[-1]
        except Exception:
            return str(value)

    def event_icon(name):
        icons = {
            "PERSON_ENTERED": "👤",
            "PERSON_PRESENT": "✅",
            "LOITERING": "🚶",
            "PERSON_LEFT": "↩️",
            "VEHICLE_DETECTED": "🚘",
            "CROWD_GATHERED": "👥",
        }
        return icons.get(name, "📌")

    def event_color(name):
        if name == "LOITERING":
            return "orange"
        if name == "PERSON_PRESENT":
            return "green"
        if name == "PERSON_ENTERED":
            return "blue"
        if name == "PERSON_LEFT":
            return "red"
        return "purple"

    def img_for_snapshot(snapshot):
        if snapshot:
            return f"/snapshot_file/{snapshot}"
        return "/video_feed"

    evidence_cards = ""
    for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in recent_for_cards[:4]:
        image_src = img_for_snapshot(snapshot)
        ev_time = time_only(event_time)
        pid = person_label(person_id)
        evidence_buttons = evidence_link(snapshot, video_clip)
        evidence_cards += f"""
        <div class="evidence-card">
            <div class="evidence-img-wrap">
                <img src="{image_src}" class="evidence-img">
                <span class="time-chip">{ev_time}</span>
            </div>
            <div class="evidence-title">{event_name}</div>
            <div class="muted-small">{pid}</div>
            <div class="mini-actions">{evidence_buttons}</div>
        </div>
        """

    if not evidence_cards:
        evidence_cards = "<div class='empty'>No evidence yet.</div>"

    alert_cards = ""
    for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in recent_for_cards[:3]:
        if event_name in ["LOITERING", "PERSON_PRESENT", "PERSON_ENTERED", "CROWD_GATHERED"]:
            alert_cards += f"""
            <div class="alert-row">
                <div>
                    <b>{event_name.replace('_', ' ').title()}</b><br>
                    <span>Main Entrance</span>
                </div>
                <div class="alert-time">{time_only(event_time)}</div>
                <img src="{img_for_snapshot(snapshot)}" class="alert-thumb">
            </div>
            """
    if not alert_cards:
        alert_cards = "<div class='empty'>No active alerts.</div>"

    timeline_html = ""
    for event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name in reversed(recent_for_cards):
        timeline_html += f"""
        <div class="timeline-item {event_color(event_name)}">
            <div class="timeline-dot">{event_icon(event_name)}</div>
            <div class="timeline-time">{time_only(event_time)}</div>
            <div class="timeline-name">{event_name}</div>
            <div class="timeline-place">Main Entrance</div>
        </div>
        """
    if not timeline_html:
        timeline_html = "<div class='empty'>No timeline events yet.</div>"

    watchlist_html = ""
    watch_rows = get_watchlist()
    if watch_rows:
        person_id, reason, watch_risk, created_at = watch_rows[0]
        watchlist_html = f"""
        <div class="watch-profile">
            <img src="/video_feed" class="watch-photo">
            <div>
                <h3>Person ID: {person_id}</h3>
                <p>Reason: {reason}</p>
                <p>Risk: {watch_risk}</p>
                <p>Added: {created_at}</p>
            </div>
        </div>
        """
    else:
        watchlist_html = "<div class='empty'>Watchlist is empty.</div>"

    ai_answer_html = ""
    if ai_answer:
        ai_answer_html = f"""
        <div class="ai-answer">
            <div class="ai-answer-title">🛡️ VisionGuard AI Response</div>
            <div>{ai_answer}</div>
            <div class="confidence">Confidence: 98%</div>
        </div>
        """

    status_text = "All Systems Online" if last_event else "Waiting For Events"
    result_title = "Filtered Results" if search_text or start_date or end_date else "Recent Events"
    
    db_status = check_database_status()
    db_class = "online" if db_status == "Online" else "offline"
    
    camera_status = check_camera_status()
    camera_text = "1 / 1" if camera_status == "Online" else "0 / 1"
    camera_class = "online" if camera_status == "Online" else "offline"
   
    storage_percent = check_storage_status()
    
    ai_status = check_ai_engine()
    ai_class = "online" if ai_status == "Online" else "offline"

    yolo_status = check_yolo_status()
    yolo_class = "online" if yolo_status == "Online" else "offline"

    rows_html = ""
    for event in events:
        event_time, event_name, snapshot, video_clip, person_id, shirt_color, camera_name = event
        rows_html += f"""
        <tr>
            <td>{event_time}</td>
            <td>{event_badge(event_name)}</td>
            <td>{person_label(person_id)}</td>
            <td>{shirt_label(shirt_color)}</td>
            <td>{camera_name or "-"}</td>
            <td>{evidence_link(snapshot, video_clip)}</td>
        </tr>
        """

    if not rows_html:
        rows_html = "<tr><td colspan='6'>No events found.</td></tr>"

    quick_buttons = ["What happened?", "Who loitered?", "Person ID 1", "Show latest evidence", "Show watchlist"]
    quick_html = "".join([f"<a class='quick-btn' href='/?ask={b}'>{b}</a>" for b in quick_buttons])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VisionGuard AI Dashboard</title>
        <meta http-equiv="refresh" content="10">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: Arial, Helvetica, sans-serif;
                background: radial-gradient(circle at top left, #1b1240 0, #080d18 36%, #05070d 100%);
                color: #f8fafc;
            }}
            a {{ color: inherit; }}
            .app {{ display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }}
            .sidebar {{
                background: rgba(7, 12, 24, 0.88);
                border-right: 1px solid rgba(148, 163, 184, 0.12);
                padding: 22px 18px;
                position: sticky;
                top: 0;
                height: 100vh;
            }}
            .brand {{ display: flex; gap: 12px; align-items: center; margin-bottom: 28px; }}
            .logo {{
                width: 48px; height: 48px; border-radius: 16px;
                display:flex; align-items:center; justify-content:center;
                background: linear-gradient(135deg, #7c3aed, #2563eb);
                font-size: 28px;
                box-shadow: 0 0 28px rgba(124,58,237,.45);
            }}
            .brand h1 {{ margin:0; font-size: 25px; }}
            .brand p {{ margin:3px 0 0; color:#cbd5e1; font-size: 14px; }}
            .nav a {{
                display:flex; align-items:center; gap:13px;
                color:#dbeafe; text-decoration:none;
                padding: 15px 16px; margin: 7px 0;
                border-radius: 12px;
            }}
            .nav a.active, .nav a:hover {{ background: linear-gradient(90deg, #5b21b6, #4338ca); }}
            .health {{
                text-decoration:none;
                display:block;
                position:absolute; left:18px; right:18px; bottom:52px;
                background: rgba(15,23,42,.82); border:1px solid rgba(148,163,184,.14);
                border-radius: 14px; padding: 16px;
            }}
            .health-row {{ display:flex; justify-content:space-between; padding:9px 0; color:#cbd5e1; border-bottom:1px solid rgba(148,163,184,.09); }}
            .online {{ color:#22c55e; }} .blue-text {{ color:#60a5fa; }}
            .offline {{ color:#ef4444; }}
            .content {{ padding: 20px 24px 28px; }}
            .topbar {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:18px; }}
            .timebox {{ text-align:right; color:#e5e7eb; line-height:1.45; }}
            .status-pill {{
                display:flex; align-items:center; gap:10px; padding:13px 18px; border-radius: 12px;
                background: rgba(15,23,42,.86); border:1px solid rgba(148,163,184,.12);
                color:#22c55e; font-weight:bold;
            }}
            .status-dot {{ width:14px; height:14px; border-radius:50%; background:#22c55e; box-shadow:0 0 18px #22c55e; }}
            .grid-main {{ display:grid; grid-template-columns: 1.02fr 1.15fr; gap:16px; }}
            .panel {{
                background: rgba(15, 23, 42, .78);
                border: 1px solid rgba(148, 163, 184, .14);
                box-shadow: 0 14px 45px rgba(0,0,0,.35);
                border-radius: 14px;
                padding: 16px;
                overflow:hidden;
            }}
            .panel-title {{ display:flex; justify-content:space-between; align-items:center; margin:0 0 12px; font-size:15px; letter-spacing:.02em; }}
            .live-wrap {{ position:relative; border-radius:12px; overflow:hidden; background:#020617; }}
            .camera {{ width:100%; height:310px; object-fit:cover; display:block; }}
            .live-chip {{ position:absolute; top:12px; right:12px; background:#ef4444; border-radius:8px; padding:8px 10px; font-size:12px; font-weight:bold; }}
            .cam-label {{ position:absolute; left:14px; bottom:14px; background:rgba(2,6,23,.75); padding:8px 12px; border-radius:9px; font-size:13px; }}
            .cam-strip {{ display:grid; grid-template-columns: repeat(5, 1fr); gap:10px; margin-top:10px; }}
            .cam-tile {{ height:70px; border-radius:10px; overflow:hidden; background:#111827; border:1px solid rgba(148,163,184,.16); position:relative; text-align:center; }}
            .cam-tile img {{ width:100%; height:100%; object-fit:cover; opacity:.8; }}
            .cam-tile span {{ position:absolute; bottom:7px; left:0; right:0; font-size:12px; }}
            .quick-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }}
            .quick-btn {{ text-decoration:none; background:rgba(30,41,59,.85); padding:13px 17px; border-radius:9px; color:#e5e7eb; border:1px solid rgba(148,163,184,.12); }}
            .ask-form {{ display:flex; gap:10px; margin-bottom:15px; }}
            input {{ background:rgba(2,6,23,.5); border:1px solid rgba(148,163,184,.18); color:#f8fafc; padding:14px; border-radius:10px; font-size:15px; }}
            .ask-input {{ flex:1; }}
            button, .purple-btn {{ background:linear-gradient(135deg,#7c3aed,#4f46e5); color:white; border:0; border-radius:10px; padding:12px 19px; font-weight:bold; cursor:pointer; text-decoration:none; display:inline-block; }}
            .ai-answer {{ background:rgba(6,78,59,.22); border:1px solid rgba(34,197,94,.23); border-radius:12px; padding:15px; color:#dcfce7; line-height:1.55; }}
            .ai-answer-title {{ color:#4ade80; font-weight:bold; margin-bottom:8px; }}
            .confidence {{ text-align:right; color:#86efac; font-size:13px; margin-top:8px; }}
            .stats {{ display:grid; grid-template-columns: repeat(5, 1fr); gap:14px; margin:16px 0; }}
            .stat-card {{ display:flex; gap:15px; align-items:center; }}
            .stat-icon {{ width:54px; height:54px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:25px; background:rgba(124,58,237,.18); }}
            .stat-value {{ font-size:30px; font-weight:bold; color:#60a5fa; }}
            .stat-label {{ color:#d1d5db; }}
            .stat-note {{ color:#94a3b8; font-size:13px; margin-top:4px; }}
            .lower-grid {{ display:grid; grid-template-columns: 1.05fr 1.45fr 1.05fr; gap:16px; }}
            .alert-row {{ display:grid; grid-template-columns: 1fr auto 92px; align-items:center; gap:12px; background:rgba(127,29,29,.18); border-radius:10px; padding:10px; margin-bottom:10px; }}
            .alert-row b {{ color:#f87171; }} .alert-row span, .muted-small, .empty {{ color:#94a3b8; font-size:13px; }}
            .alert-thumb {{ width:92px; height:52px; border-radius:8px; object-fit:cover; }}
            .alert-time {{ font-weight:bold; font-size:13px; }}
            .evidence-grid {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; }}
            .evidence-card {{ background:rgba(15,23,42,.88); border-radius:12px; padding:9px; border:1px solid rgba(148,163,184,.12); }}
            .evidence-img-wrap {{ position:relative; height:118px; border-radius:10px; overflow:hidden; }}
            .evidence-img {{ width:100%; height:100%; object-fit:cover; }}
            .time-chip {{ position:absolute; left:7px; bottom:7px; background:rgba(2,6,23,.8); padding:5px 7px; border-radius:7px; font-size:12px; }}
            .evidence-title {{ margin-top:9px; font-weight:bold; font-size:13px; }}
            .mini-actions .view-link, .mini-actions .video-link {{ font-size:11px; padding:5px 7px; margin-top:8px; }}
            .watch-profile {{ display:flex; gap:14px; align-items:flex-start; }}
            .watch-photo {{ width:92px; height:138px; object-fit:cover; border-radius:10px; background:#111827; }}
            .watch-profile h3 {{ color:#f87171; margin:0 0 10px; }} .watch-profile p {{ margin:7px 0; color:#e5e7eb; }}
            .timeline-panel {{ margin-top:16px; }}
            .timeline {{ display:flex; align-items:flex-start; justify-content:space-between; border-top:3px solid #475569; margin-top:35px; padding-top:0; }}
            .timeline-item {{ position:relative; min-width:135px; transform:translateY(-25px); }}
            .timeline-dot {{ width:50px; height:50px; border-radius:50%; display:flex; align-items:center; justify-content:center; background:#1e293b; border:3px solid #475569; font-size:23px; }}
            .timeline-item.green .timeline-dot {{ border-color:#22c55e; }} .timeline-item.orange .timeline-dot {{ border-color:#fb923c; }} .timeline-item.blue .timeline-dot {{ border-color:#38bdf8; }} .timeline-item.red .timeline-dot {{ border-color:#f87171; }}
            .timeline-time {{ margin-top:8px; color:#cbd5e1; font-size:13px; }}
            .timeline-name {{ font-weight:bold; font-size:13px; }}
            .timeline-place {{ color:#94a3b8; font-size:12px; }}
            .table-panel {{ margin-top:16px; }}
            .filter-form {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:13px; }}
            .text-input {{ min-width:260px; flex:1; }}
            table {{ width:100%; border-collapse:collapse; overflow:hidden; border-radius:12px; }}
            th {{ background:rgba(79,70,229,.85); color:white; padding:12px; text-align:left; }}
            td {{ padding:11px 12px; border-bottom:1px solid rgba(148,163,184,.1); color:#e5e7eb; }}
            tr:hover {{ background:rgba(30,41,59,.45); }}
            .badge {{ padding:6px 10px; border-radius:999px; font-weight:bold; font-size:12px; display:inline-block; }}
            .green {{ background:rgba(22,163,74,.18); color:#86efac; }} .blue {{ background:rgba(37,99,235,.18); color:#93c5fd; }} .orange {{ background:rgba(234,88,12,.18); color:#fdba74; }} .red {{ background:rgba(220,38,38,.18); color:#fca5a5; }} .gray {{ background:rgba(107,114,128,.25); color:#d1d5db; }}
            .view-link {{ background:#16a34a; color:white; padding:6px 9px; border-radius:7px; text-decoration:none; display:inline-block; margin:2px; font-size:12px; }}
            .video-link {{ background:#7c3aed; color:white; padding:6px 9px; border-radius:7px; text-decoration:none; display:inline-block; margin:2px; font-size:12px; }}
            @media (max-width: 1200px) {{ .app {{ grid-template-columns:1fr; }} .sidebar {{ position:relative; height:auto; }} .health {{ position:relative; left:auto; right:auto; bottom:auto; margin-top:20px; }} .grid-main, .lower-grid {{ grid-template-columns:1fr; }} .stats {{ grid-template-columns: repeat(2,1fr); }} }}
        </style>
    </head>
    <body>
        <div class="app">
            <aside class="sidebar">
                <div class="brand"><div class="logo">🛡️</div><div><h1>VisionGuard AI</h1><p>Intelligent CCTV Monitoring</p></div></div>
                <nav class="nav">
                    <a class="active" href="/">🏠 Dashboard</a>
                    <a href="/live-cameras">📹 Live Cameras</a>
                    <a href="/persons">👥 Person Intelligence</a>
                    <a href="/ai-assistant">🤖 AI Assistant</a>
                    <a href="/alerts">🔔 Alerts</a>
                    <a href="/watchlist">⭐ Watchlist</a>
                    <a href="/event-search">🔎 Event Search</a>
                    <a href="/analytics">📊 Analytics</a>
                    <a href="/settings">⚙️ Settings</a>
                </nav>
            </aside>

            <main class="content">
                <div class="topbar">
                    <div></div>
                    <div style="display:flex;gap:18px;align-items:center;">
                        <div class="timebox"><b>{now_time}</b><br>{now_date}</div>
                        <div class="status-pill"><span class="status-dot"></span><div>System Status<br>{status_text}</div></div>
                    </div>
                </div>

                <section class="grid-main">
                    <div class="panel">
                        <h2 class="panel-title">📹 LIVE CAMERA FEED <a class="purple-btn" href="/">Refresh</a></h2>
                        <div class="live-wrap">
                            <img class="camera" src="/video_feed">
                            <div class="live-chip">● LIVE</div>
                            <div class="cam-label">CAM 01 · Main Entrance</div>
                        </div>
                        <div class="cam-strip">
                            <div class="cam-tile"><img src="/video_feed"><span>CAM 01</span></div>
                            <div class="cam-tile"><img src="/video_feed"><span>CAM 02</span></div>
                            <div class="cam-tile"><img src="/video_feed"><span>CAM 03</span></div>
                            <div class="cam-tile"><img src="/video_feed"><span>CAM 04</span></div>
                            <div class="cam-tile"><span style="top:25px;bottom:auto;">+8<br>More Cameras</span></div>
                        </div>
                    </div>

                    <div class="panel">
                        <h2 class="panel-title">🧠 VISIONGUARD AI ASSISTANT</h2>
                        <div class="quick-row">{quick_html}</div>
                        <form class="ask-form" method="get" action="/">
                            <input class="ask-input" name="ask" placeholder="Ask anything about the CCTV..." value="{ai_question}">
                            <button type="submit">Ask</button>
                        </form>
                        {ai_answer_html if ai_answer_html else '<div class="ai-answer"><div class="ai-answer-title">🛡️ VisionGuard AI Response</div>Ask a question or click a quick action above.</div>'}
                    </div>
                </section>

                <section class="stats">
                    <div class="panel stat-card"><div class="stat-icon">🚨</div><div><div class="stat-label">Active Alerts</div><div class="stat-value">{active_alerts}</div><div class="stat-note">View all alerts →</div></div></div>
                    <div class="panel stat-card"><div class="stat-icon">👥</div><div><div class="stat-label">People Present</div><div class="stat-value">{present_today}</div><div class="stat-note">Currently in view</div></div></div>
                    <div class="panel stat-card"><div class="stat-icon">🚶</div><div><div class="stat-label">Loitering</div><div class="stat-value" style="color:#fb923c;">{loitering_today}</div><div class="stat-note">Requires attention</div></div></div>
                    <div class="panel stat-card"><div class="stat-icon">📋</div><div><div class="stat-label">Total Events</div><div class="stat-value">{total_events}</div><div class="stat-note">All recorded events</div></div></div>
                    <div class="panel stat-card"><div class="stat-icon">📈</div><div><div class="stat-label">Risk Level</div><div class="stat-value" style="color:#facc15;">{risk_level}</div><div class="stat-note">Monitor carefully</div></div></div>
                </section>

                <section class="lower-grid">
                    <div class="panel"><h2 class="panel-title">⚠️ ACTIVE ALERTS <span>View all</span></h2>{alert_cards}</div>
                    <div class="panel"><h2 class="panel-title">🎞️ RECENT EVIDENCE <span>View all</span></h2><div class="evidence-grid">{evidence_cards}</div></div>
                    <div class="panel"><h2 class="panel-title">🚩 WATCHLIST <span>View all</span></h2>{watchlist_html}</div>
                </section>

                <section class="panel timeline-panel">
                    <h2 class="panel-title">⏱️ EVENT TIMELINE <span>View full timeline</span></h2>
                    <div class="timeline">{timeline_html}</div>
                </section>

                <section class="panel table-panel">
                    <h2 class="panel-title">{result_title}</h2>
                    <form class="filter-form" method="get" action="/">
                        <input class="text-input" name="q" placeholder="Search: PERSON_ENTERED, LOITERING, PERSON_LEFT..." value="{search_text}">
                        <input type="date" name="start_date" value="{start_date}">
                        <input type="date" name="end_date" value="{end_date}">
                        <button type="submit">Filter</button>
                        <a class="purple-btn" href="/">Clear</a>
                    </form>
                    <table>
                        <tr><th>Time</th><th>Event</th><th>Person ID</th><th>Shirt Color</th><th>Camera</th><th>Evidence</th></tr>
                        {rows_html}
                    </table>
                </section>
            </main>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html)

@app.get("/ai-answer")
def ai_answer_api(
    ask: str = Query(default=""),
    history: str = Query(default="")
):
    if not ask.strip():
        return {"answer": "Please ask a CCTV question."}

    answer = ask_visionguard(ask)

    return {"answer": answer}


@app.get("/ai-assistant", response_class=HTMLResponse)
def ai_assistant():
    ensure_chat_tables()
    chat_sessions = get_chat_sessions()

    chat_history_html = ""
    for session_id, title in chat_sessions:
        chat_history_html += f'<button class="side-btn">💬 {title}</button>'

    html = """
<!DOCTYPE html>
<html>
<head>
<title>VisionGuard AI Assistant</title>
<style>
*{box-sizing:border-box}
body{
    margin:0;
    background:#000;
    color:#ececec;
    font-family:Arial, sans-serif;
}
.app{
    display:flex;
    height:100vh;
    overflow:visible;
}
.sidebar{
    width:260px;
    background:#171717;
    border-right:1px solid #222;
    padding:18px 12px;
    display:flex;
    flex-direction:column;
    overflow-y:auto;
}
.brand{
    font-size:22px;
    font-weight:700;
    padding:10px 12px 20px;
}
.side-btn{
    display:block;
    width:100%;
    text-align:left;
    background:transparent;
    color:#ececec;
    border:0;
    padding:13px 14px;
    border-radius:10px;
    font-size:15px;
    cursor:pointer;
    text-decoration:none;
}
.side-btn:hover,.active{
    background:#2a2a2a;
}
.side-title{
    color:#9ca3af;
    font-size:13px;
    margin:20px 14px 8px;
}
.user-box{
    margin-top:20px;
    display:flex;
    gap:10px;
    align-items:center;
    padding:12px;
    border-radius:12px;
}
.user-box:hover{background:#1f1f1f}
.avatar-small{
    width:34px;
    height:34px;
    border-radius:50%;
    background:#7c3aed;
    display:flex;
    align-items:center;
    justify-content:center;
}
.main{
    flex:1;
    display:flex;
    flex-direction:column;
    background:#212121;
}
.topbar{
    height:64px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 24px;
    border-bottom:1px solid #202020;
}
.title{
    font-size:20px;
    font-weight:600;
}
.top-actions{
    display:flex;
    gap:12px;
    align-items:center;
}
.icon-btn{
    background:transparent;
    color:#ececec;
    border:1px solid #333;
    border-radius:999px;
    padding:8px 14px;
    cursor:pointer;
    text-decoration:none;
}
.chat-area{
    flex:1;
    overflow-y:auto;
    padding:30px 20px 150px;
}
.chat-inner{
    width:100%;
    max-width:720px;
    margin:0 auto;
    padding-left:0;
    padding-right:0;
}
.welcome{
    text-align:center;
    margin-top:70px;
}
.welcome h1{
    font-size:32px;
    margin-bottom:10px;
}
.welcome p{
    color:#aaa;
}
.quick-grid{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:12px;
    margin-top:28px;
}
.quick-card{
    background:#1b1b1b;
    color:#ececec;
    border:1px solid #333;
    border-radius:16px;
    padding:16px;
    text-align:left;
    cursor:pointer;
}
.quick-card:hover{background:#262626}
.message{
    display:flex;
    gap:14px;
    margin:22px 0;
    align-items:flex-start;
}

.user-msg{
    justify-content:flex-end;
}

.user-msg .msg-avatar{
    order:2;
}

.user-msg .msg-content{
    order:1;
    max-width:520px;
}
.msg-avatar{
    width:32px;
    height:32px;
    border-radius:50%;
    flex-shrink:0;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:17px;
}
.user-avatar{background:#2563eb}
.ai-avatar{background:#7c3aed}
.msg-content{
    line-height:1.7;
    font-size:15px;
    max-width:620px;
}
.user-msg .msg-content{
    background:#2f2f2f;
    padding:12px 16px;
    border-radius:18px;
}
.ai-msg .msg-content{
    color:#f4f4f5;
    background:none;
    padding:0;
}
.input-wrap{
    position:fixed;
    left:260px;
    right:0;
    bottom:0;
    padding:26px 20px 18px;
    background:linear-gradient(to top,#0b0b0b 75%,rgba(11,11,11,0));
}
.input-box{
    max-width:720px;
    height:60px;
    margin:0 auto;
    background:#2b2b2b;
    border:1px solid #444;
    border-radius:28px;
    display:flex;
    align-items:flex-end;
    padding:10px 10px 10px 18px;
    gap:10px;
}
textarea{
    flex:1;
    background:transparent;
    color:#ececec;
    border:0;
    outline:0;
    resize:none;
    font-size:16px;
    font-family:Arial,sans-serif;
    min-height:38px;
    max-height:160px;
    padding:9px 0;
}
.send{
    width:42px;
    height:42px;
    border-radius:50%;
    border:0;
    background:#fff;
    color:#000;
    font-size:20px;
    cursor:pointer;
}
.send:hover{background:#ddd}
.disclaimer{
    text-align:center;
    color:#8a8a8a;
    font-size:12px;
    margin-top:10px;
}
.typing{
    color:#aaa;
}

.chat-menu{
    background:#2b2b2b;
    border:1px solid #444;
    border-radius:14px;
    padding:8px;
    min-width:180px;
    z-index:9999;
    box-shadow:0 12px 30px rgba(0,0,0,.45);
}

.chat-menu div{
    padding:12px 14px;
    border-radius:10px;
    cursor:pointer;
    color:#ececec;
}

.chat-menu div:hover{
    background:#3a3a3a;
}

.chat-menu .delete{
    color:#ff5c5c;
}

@media(max-width:800px){
    .sidebar{display:none}
    .input-wrap{left:0}
    .quick-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>

<div class="app">
    <aside class="sidebar">
        <div class="title">🤖 VisionGuard AI Assistant</div>

        <button class="side-btn" onclick="clearChat()">✎ New chat</button>
        <a class="side-btn" href="/">⌂ Dashboard</a>

        <div class="side-title">Pinned</div>
        <button class="side-btn active">📁 VisionGuard AI</button>

        <div class="side-title" id="historyTitle">Chat History</div>
        <div id="historyList"></div>

        <div class="side-title">CCTV prompts</div>
        <button class="side-btn" onclick="askQuick('what happened today')">What happened today?</button>
        <button class="side-btn" onclick="askQuick('who loitered the most')">Who loitered most?</button>
        <button class="side-btn" onclick="askQuick('tell me about person id 1')">Person ID 1</button>
        <button class="side-btn" onclick="askQuick('when is the latest loitering')">Latest loitering</button>

        <div class="user-box">
            <div class="avatar-small">🤖</div>
            <div>
                <div>VisionGuard</div>
                <small style="color:#aaa">Local AI</small>
            </div>
        </div>
    </aside>

    <main class="main">
        <div class="topbar">
            <div class="title">📁 VisionGuard AI Assistant</div>
            <div class="top-actions">
                <a class="icon-btn" href="/">Dashboard</a>
                <button class="icon-btn" onclick="clearChat()">Clear</button>
            </div>
        </div>

        <div class="chat-area" id="chatArea">
            <div class="chat-inner" id="messages"></div>
        </div>

        <div class="input-wrap">
            <form class="input-box" onsubmit="sendMessage(); return false;">
                <textarea id="question" rows="1" placeholder="Ask anything"></textarea>
                <button class="send" type="submit">↑</button>
            </form>
            <div class="disclaimer">VisionGuard can make mistakes. Check important CCTV evidence.</div>
        </div>
    </main>
</div>

<script>

let chats = JSON.parse(localStorage.getItem("vg_chats") || "[]");
let currentChat = null;

function saveChats(){
    localStorage.setItem("vg_chats", JSON.stringify(chats));
}

function createNewChat(){
    currentChat = {
        id: Date.now(),
        title: "New Chat",
        messages: []
    };
    chats.unshift(currentChat);
    saveChats();
    renderChat();
    renderHistory();
}

function renderHistory(){
    document.querySelectorAll(".history-chat").forEach(x => x.remove());
    document.querySelectorAll(".chat-menu").forEach(x => x.remove());

    const historyTitle = document.getElementById("historyTitle");
    
    const sortedChats = [...chats].sort((a, b) => {
        if (a.pinned && !b.pinned) return -1;
        if (!a.pinned && b.pinned) return 1;
        return b.id - a.id;
    }); 

    sortedChats.forEach(chat => {
        const item = document.createElement("div");
        item.className = "side-btn history-chat";
        item.style.display = "flex";
        item.style.justifyContent = "space-between";
        item.style.alignItems = "center";
        item.style.gap = "8px";

        const title = document.createElement("span");
        title.innerHTML = (chat.pinned ? "📌 " : "💬 ") + chat.title;
        title.style.overflow = "hidden";
        title.style.whiteSpace = "nowrap";
        title.style.textOverflow = "ellipsis";

        const dots = document.createElement("button");
        dots.innerHTML = "⋯";
        dots.style.background = "transparent";
        dots.style.color = "#ececec";
        dots.style.border = "0";
        dots.style.cursor = "pointer";
        dots.style.fontSize = "20px";

        item.onclick = function(){
            currentChat = chat;
            renderChat();
        };

        dots.onclick = function(e){
            e.stopPropagation();
            openChatMenu(chat, dots);
        };

        item.appendChild(title);
        item.appendChild(dots);

        document.getElementById("historyList").appendChild(item);
    });
}

function openChatMenu(chat, button){
    document.querySelectorAll(".chat-menu").forEach(x => x.remove());

    const menu = document.createElement("div");
    menu.className = "chat-menu";
    menu.innerHTML = `
        <div onclick="renameChat(${chat.id})">✏️ Rename</div>
        <div onclick="pinChat(${chat.id})">${chat.pinned ? "📍 Unpin chat" : "📌 Pin chat"}</div>
        <div class="delete" onclick="deleteChat(${chat.id})">🗑 Delete</div>
    `;

    document.body.appendChild(menu);

    const rect = button.getBoundingClientRect();
    menu.style.position = "fixed";
    menu.style.left = rect.left + "px";
    menu.style.top = rect.bottom + 6 + "px";

        }

function renameChat(id){
    const chat = chats.find(c => c.id === id);
    const newTitle = prompt("Rename chat:", chat.title);
    if(newTitle){
        chat.title = newTitle;
        saveChats();
        renderHistory();
    }
}

function pinChat(id){
    const chat = chats.find(c => c.id === id);

    if(chat){
        chat.pinned = !chat.pinned;

        saveChats();
        renderHistory();
    }
}

function deleteChat(id){
    chats = chats.filter(c => c.id !== id);
    if(currentChat && currentChat.id === id){
        currentChat = null;
    }
    saveChats();
    renderHistory();
    renderChat();
}

function escapeHTML(text){
    return String(text)
        .replace(/&/g,"&amp;")
        .replace(/</g,"&lt;")
        .replace(/>/g,"&gt;");
}

function renderChat(){
    const box = document.getElementById("messages");
    box.innerHTML = "";

    if(!currentChat || currentChat.messages.length===0){
        box.innerHTML = `
        <div class="welcome">
            <h1>What can I help with?</h1>
            <p>Ask about CCTV events.</p>
        </div>`;
        return;
    }

    currentChat.messages.forEach(m=>{
        const row=document.createElement("div");
        row.className="message "+(m.role==="user"?"user-msg":"ai-msg");

        const avatar=document.createElement("div");
        avatar.className="msg-avatar "+(m.role==="user"?"user-avatar":"ai-avatar");
        avatar.innerHTML=m.role==="user"?"A":"🤖";

        const content=document.createElement("div");
        content.className="msg-content";
        content.innerHTML=m.text;

        row.appendChild(avatar);
        row.appendChild(content);
        box.appendChild(row);
    });

    document.getElementById("chatArea").scrollTop =
        document.getElementById("chatArea").scrollHeight;
}

async function sendMessage(text=null){
    const input=document.getElementById("question");
    const q=text || input.value.trim();

    if(!q) return;

    if(!currentChat){
        createNewChat();
    }

    if(currentChat.title==="New Chat"){
        currentChat.title=q.substring(0,30);
    }

    currentChat.messages.push({
        role:"user",
        text:escapeHTML(q)
    });

    currentChat.messages.push({
        role:"ai",
        text:"Thinking..."
    });

    saveChats();
    renderChat();
    renderHistory();

    input.value="";

    try{
        const res=await fetch("/ai-answer?ask="+encodeURIComponent(q));
        const data=await res.json();

        currentChat.messages.pop();
        currentChat.messages.push({
            role:"ai",
            text:data.answer
        });

        saveChats();
        renderChat();

    } catch(e){
        currentChat.messages.pop();
        currentChat.messages.push({
            role:"ai",
            text:"AI error."
        });

        renderChat();
    }
}

function askQuick(q){
    sendMessage(q);
}

function clearChat(){
    createNewChat();
}

document.getElementById("question").addEventListener("keydown", function(e){
    if(e.key==="Enter" && !e.shiftKey){
        e.preventDefault();
        sendMessage();
    }
});

renderHistory();
renderChat();

</script>

</body>
</html>
"""
    return HTMLResponse(html)

@app.get("/analytics", response_class=HTMLResponse)
def analytics():
    return HTMLResponse("""
    <html>
    <body style="background:#111827;color:white;font-family:Arial;padding:40px">
        <h1>📊 Analytics Page</h1>
        <p>Analytics page is working.</p>
        <a href="/" style="color:#60a5fa">Back</a>
    </body>
    </html>
    """)

@app.get("/alerts", response_class=HTMLResponse)
def alerts():
    return HTMLResponse("""
    <html>
    <body style="background:#111827;color:white;font-family:Arial;padding:40px">
        <h1>🔔 Alerts Page</h1>
        <p>Alerts page is working.</p>
        <a href="/" style="color:#60a5fa">Back</a>
    </body>
    </html>
    """)

@app.get("/watchlist", response_class=HTMLResponse)
def watchlist_page():
    return HTMLResponse("""
    <html>
    <body style="background:#111827;color:white;font-family:Arial;padding:40px">
        <h1>⭐ Watchlist Page</h1>
        <p>Watchlist page is working.</p>
        <a href="/" style="color:#60a5fa">Back</a>
    </body>
    </html>
    """)

@app.get("/event-search", response_class=HTMLResponse)
def event_search():
    return HTMLResponse("""
    <html>
    <body style="background:#111827;color:white;font-family:Arial;padding:40px">
        <h1>🔎 Event Search Page</h1>
        <p>Event Search page is working.</p>
        <a href="/" style="color:#60a5fa">Back</a>
    </body>
    </html>
    """)