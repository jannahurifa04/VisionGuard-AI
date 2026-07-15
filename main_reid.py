from ultralytics import YOLO
import cv2
import requests
import numpy as np
from datetime import datetime
import sqlite3
import time
import os
from collections import deque
import imageio.v2 as imageio
from sklearn.cluster import KMeans
from reid import get_stable_person_id
from watchlist import add_watchlist
from identity_fusion import is_same_person

# =========================
# SETTINGS
# =========================

MODEL_PATH = "yolo11n.pt"

CAMERA_ID = "cam_1"
CAMERA_NAME = "Front Camera"
CAMERA_URL = "http://192.168.18.134:8080/shot.jpg"

DATABASE_NAME = "events.db"

CONFIDENCE_LIMIT = 0.4
PERSON_PRESENT_SECONDS = 30
LOITERING_SECONDS = 5
MISSING_GRACE_SECONDS = 3

ROTATE_FRAME = True

SNAPSHOT_FOLDER = "snapshots"
CLIP_FOLDER = "clips"

CLIP_SECONDS = 10
CLIP_FPS = 10

os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
os.makedirs(CLIP_FOLDER, exist_ok=True)

# =========================
# LOAD YOLO MODEL
# =========================

model = YOLO(MODEL_PATH)

# =========================
# DATABASE SETUP
# =========================

conn = sqlite3.connect(DATABASE_NAME)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time TEXT NOT NULL,
    object_name TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    snapshot TEXT,
    video_clip TEXT,
    person_id INTEGER,
    shirt_color TEXT,
    camera_id TEXT,
    camera_name TEXT
)
""")

# =========================
# PERSON INTELLIGENCE TABLE
# =========================

cursor.execute("""
CREATE TABLE IF NOT EXISTS persons (
    person_id INTEGER PRIMARY KEY,
    display_name TEXT,
    status TEXT DEFAULT 'Unknown',
    first_seen TEXT,
    last_seen TEXT,
    total_seen INTEGER DEFAULT 0,
    visit_count INTEGER DEFAULT 1,
    last_shirt_color TEXT,
    last_camera_id TEXT,
    last_camera_name TEXT,
    risk_level TEXT DEFAULT 'Low',
    avg_visit_seconds INTEGER DEFAULT 0,
    longest_visit_seconds INTEGER DEFAULT 0,
    loiter_count INTEGER DEFAULT 0,
    last_loiter TEXT,
    current_visit_start TEXT,
    notes TEXT
)
""")

conn.commit()

# Add missing columns to old database
for column_sql in [
    "ALTER TABLE events ADD COLUMN snapshot TEXT",
    "ALTER TABLE events ADD COLUMN video_clip TEXT",
    "ALTER TABLE events ADD COLUMN person_id INTEGER",
    "ALTER TABLE events ADD COLUMN shirt_color TEXT",
    "ALTER TABLE events ADD COLUMN camera_id TEXT",
    "ALTER TABLE events ADD COLUMN camera_name TEXT",
    "ALTER TABLE persons ADD COLUMN visit_count INTEGER DEFAULT 1",
    "ALTER TABLE persons ADD COLUMN avg_visit_seconds INTEGER DEFAULT 0",
    "ALTER TABLE persons ADD COLUMN longest_visit_seconds INTEGER DEFAULT 0",
    "ALTER TABLE persons ADD COLUMN loiter_count INTEGER DEFAULT 0",
    "ALTER TABLE persons ADD COLUMN last_loiter TEXT",
    "ALTER TABLE persons ADD COLUMN current_visit_start TEXT",
]:
    try:
        cursor.execute(column_sql)
        conn.commit()
    except sqlite3.OperationalError:
        pass

# =========================
# PERSON PROFILE UPDATE
# =========================

def calculate_person_risk(person_id):
    cursor.execute("""
        SELECT COUNT(*)
        FROM events
        WHERE person_id = ?
        AND object_name = 'LOITERING'
    """, (person_id,))

    loiter_count = cursor.fetchone()[0]

    if loiter_count >= 5:
        return "High"
    elif loiter_count >= 3:
        return "Medium"
    else:
        return "Low"


def update_person_profile(person_id, shirt_color=None):
    if person_id is None:
        return

    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        SELECT person_id
        FROM persons
        WHERE person_id = ?
    """, (person_id,))

    existing_person = cursor.fetchone()

    risk_level = calculate_person_risk(person_id)

    if existing_person:
        cursor.execute("""
            UPDATE persons
            SET last_seen = ?,
                last_shirt_color = ?,
                last_camera_id = ?,
                last_camera_name = ?,
                
                status = CASE
                    WHEN visit_count >= 20 THEN 'VIP'
                    WHEN visit_count >= 10 THEN 'Frequent'
                    ELSE 'Visitor'
                END,

                risk_level = ?
            WHERE person_id = ?
        """, (
            now_time,
            shirt_color,
            CAMERA_ID,
            CAMERA_NAME,
            risk_level,
            person_id
        ))

    else:
        cursor.execute("""
            INSERT INTO persons (
                person_id,
                display_name,
                status,
                first_seen,
                last_seen,
                total_seen,
                visit_count,
                last_shirt_color,
                last_camera_id,
                last_camera_name,
                risk_level,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            person_id,
            None,
            "Unknown",
            now_time,
            now_time,

            1,      # total_seen
            0,  # visit_count

            shirt_color,
            CAMERA_ID,
            CAMERA_NAME,
            risk_level,
            None
        ))

    conn.commit()

    print(f"[PERSON PROFILE] Person ID {person_id} updated | Risk={risk_level}")

def save_event(event_name, confidence=1.0, snapshot=None, video_clip=None, person_id=None, shirt_color=None):
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO events (
            event_time, object_name, confidence,
            snapshot, video_clip, person_id, shirt_color,
            camera_id, camera_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_time,
            event_name,
            round(confidence, 2),
            snapshot,
            video_clip,
            person_id,
            shirt_color,
            CAMERA_ID,
            CAMERA_NAME,
        ),
    )
    conn.commit()

    if event_name == "PERSON_ENTERED" and person_id is not None:
        cursor.execute("""
            UPDATE persons
            SET visit_count = visit_count + 1,
                current_visit_start = ?
            WHERE person_id = ?
        """, (
            event_time,
            person_id
        ))

        conn.commit()

    if event_name == "PERSON_LEFT" and person_id is not None:
        cursor.execute("""
            SELECT current_visit_start
            FROM persons
            WHERE person_id = ?
        """, (person_id,))

        visit_row = cursor.fetchone()

        if visit_row and visit_row[0]:
            visit_start = datetime.strptime(
                visit_row[0],
                "%Y-%m-%d %H:%M:%S"
            )

            visit_seconds = int(
                (datetime.now() - visit_start).total_seconds()
            )

            cursor.execute("""
                UPDATE persons
                SET avg_visit_seconds =
                        CASE
                            WHEN visit_count <= 1 THEN ?
                            ELSE (
                                (avg_visit_seconds * (visit_count - 1)) + ?
                            ) / visit_count
                        END,
                    longest_visit_seconds =
                        MAX(longest_visit_seconds, ?),
                    current_visit_start = NULL
                WHERE person_id = ?
            """, (
                visit_seconds,
                visit_seconds,
                visit_seconds,
                person_id
            ))

            conn.commit()

            print(
                f"[VISIT DURATION] Person ID {person_id} "
                f"stayed {visit_seconds} seconds"
            )

    if event_name == "LOITERING" and person_id is not None:
        cursor.execute("""
            UPDATE persons
            SET loiter_count = loiter_count + 1,
                last_loiter = ?
            WHERE person_id = ?
        """, (
            event_time,
            person_id
        ))

        conn.commit()

        cursor.execute("""
            SELECT COUNT(*)
            FROM events
            WHERE object_name = 'LOITERING'
            AND person_id = ?
        """, (person_id,))

        loiter_count = cursor.fetchone()[0]

        if loiter_count >= 3:
            add_watchlist(
                person_id,
                f"Auto watchlist: repeated loitering {loiter_count} times",
                "Medium"
            )

            print(
                f"[AUTO WATCHLIST] Person ID {person_id} added. "
                f"Loitering count: {loiter_count}"
            )

    print(
        f"[SMART EVENT] {event_time} - {event_name} - "
        f"{CAMERA_NAME} - Person ID {person_id}"
    )

    if shirt_color:
        print(f"[SHIRT COLOR] {shirt_color}")

    if snapshot:
        print(f"[EVENT SNAPSHOT LINKED] {snapshot}")

    if video_clip:
        print(f"[EVENT VIDEO CLIP LINKED] {video_clip}")

def save_snapshot(frame, event_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{event_name}.jpg"
    filepath = os.path.join(SNAPSHOT_FOLDER, filename)

    cv2.imwrite(filepath, frame)

    print(f"[SNAPSHOT SAVED] {filepath}")

    return filename


def save_video_clip(frame_buffer, event_name):
    if not frame_buffer:
        print("[VIDEO CLIP ERROR] Frame buffer is empty")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{event_name}.mp4"
    filepath = os.path.join(CLIP_FOLDER, filename)

    frames = list(frame_buffer)

    if len(frames) == 0:
        print("[VIDEO CLIP ERROR] No frames found")
        return None

    first_frame = frames[0]
    height, width = first_frame.shape[:2]

    rgb_frames = []

    for frame in frames:
        if frame is None:
            continue

        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frames.append(rgb_frame)

    if len(rgb_frames) == 0:
        print("[VIDEO CLIP ERROR] No valid frames to write")
        return None

    try:
        imageio.mimsave(
            filepath,
            rgb_frames,
            fps=CLIP_FPS,
            codec="libx264",
            quality=8,
            macro_block_size=16
        )

        print(f"[VIDEO CLIP SAVED] {filepath}")
        return filename

    except Exception as e:
        print("[VIDEO CLIP ERROR]", e)
        return None


def detect_shirt_color(person_crop):
    try:
        if person_crop is None or person_crop.size == 0:
            return "Unknown"

        h, w = person_crop.shape[:2]

        if h < 20 or w < 20:
            return "Unknown"

        shirt_region = person_crop[
            int(h * 0.45):int(h * 0.75),
            int(w * 0.30):int(w * 0.70)
        ]

        if shirt_region.size == 0:
            return "Unknown"

        shirt_region = cv2.resize(shirt_region, (80, 80))
        hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)

        pixels = hsv.reshape((-1, 3))
        pixels = pixels[(pixels[:, 2] > 40) & (pixels[:, 2] < 245)]

        if len(pixels) < 50:
            return "Unknown"

        h_avg = int(np.median(pixels[:, 0]))
        s_avg = int(np.median(pixels[:, 1]))
        v_avg = int(np.median(pixels[:, 2]))

        if v_avg < 60:
            return "Black"

        if s_avg < 35 and v_avg > 190:
            return "White"

        if s_avg < 45:
            return "Gray"

        if h_avg < 10 or h_avg > 165:
            if s_avg < 120 and v_avg > 120:
                return "Pink"
            return "Red"

        if 10 <= h_avg < 25:
            return "Orange"

        if 25 <= h_avg < 35:
            return "Yellow"

        if 35 <= h_avg < 85:
            return "Green"

        if 85 <= h_avg < 130:
            return "Blue"

        if 130 <= h_avg <= 165:
            return "Pink"

        return "Other"

    except Exception as e:
        print("Color detection error:", e)
        return "Unknown"


def crop_person_from_box(frame, box):
    try:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    except Exception as e:
        print("Person crop error:", e)
        return None


# =========================
# MULTI-PERSON STATE
# =========================

person_states = {}

track_to_stable_id = {}

max_buffer_frames = CLIP_SECONDS * CLIP_FPS
frame_buffer = deque(maxlen=max_buffer_frames)


# =========================
# MAIN LOOP
# =========================

try:
    while True:
        try:
            response = requests.get(CAMERA_URL, timeout=8)

            if response.status_code != 200:
                print("Camera error:", response.status_code)
                time.sleep(1)
                continue

            img_array = np.array(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                print("Cannot decode image")
                time.sleep(1)
                continue

            if ROTATE_FRAME:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            results = model.track(
                frame,
                conf=CONFIDENCE_LIMIT,
                persist=True,
                tracker="bytetrack.yaml",
                device=0,
                imgsz=320,
                verbose=False
            )

            now = datetime.now()
            detections = []
            current_seen_ids = set()

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])

                if class_name != "person":
                    continue

                temp_track_id = None
                if box.id is not None:
                    temp_track_id = int(box.id[0])

                person_crop = crop_person_from_box(frame, box)
                shirt_color = detect_shirt_color(person_crop)

                # Use ByteTrack ID first
                if temp_track_id is not None and temp_track_id in track_to_stable_id:
                    stable_id = track_to_stable_id[temp_track_id]
                else:
                    stable_id = get_stable_person_id(person_crop, shirt_color)

                    if stable_id is None:
                        continue
                    
                    if temp_track_id is not None:
                        track_to_stable_id[temp_track_id] = stable_id

                update_person_profile(
                    stable_id,
                    shirt_color
                )

                detections.append({
                    "stable_id": stable_id,
                    "temp_track_id": temp_track_id,
                    "confidence": confidence,
                    "shirt_color": shirt_color,
                    "box": box,
                    "crop": person_crop
                })

                current_seen_ids.add(stable_id)

                print(
                    f"[REID TRACK] Camera={CAMERA_NAME} "
                    f"Stable ID={stable_id} "
                    f"Temp ID={temp_track_id} "
                    f"Conf={confidence:.2f} "
                    f"Shirt={shirt_color}"
                )

            annotated_frame = frame.copy()

            for det in detections:
                box = det["box"]
                stable_id = det["stable_id"]
                shirt_color = det["shirt_color"]
                confidence = det["confidence"]

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                label = (
                    f"Person ID {stable_id} | "
                    f"{shirt_color} | "
                    f"{confidence:.2f}"
                )
      
                # Label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, max(0, y1 - 30)),
                    (x1 + 320, y1),
                    (0, 255, 255),
                    -1
                )

                # Label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    2
            )

            cv2.putText(
                annotated_frame,
                f"{CAMERA_NAME} | People Count: {len(detections)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            y_text = 80
            for det in detections:
                cv2.putText(
                    annotated_frame,
                    f"Stable ID {det['stable_id']} | Shirt: {det['shirt_color']}",
                    (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                y_text += 35

            frame_buffer.append(annotated_frame.copy())

            for det in detections:
                stable_id = det["stable_id"]
                confidence = det["confidence"]
                shirt_color = det["shirt_color"]

                if stable_id not in person_states:
                    person_states[stable_id] = {
                        "present": False,
                        "enter_time": None,
                        "last_seen": now,
                        "present_logged": False,
                        "loitering_logged": False,
                        "shirt_color": shirt_color,
                        "confidence": confidence
                    }

                state = person_states[stable_id]
                state["last_seen"] = now
                state["shirt_color"] = shirt_color
                state["confidence"] = confidence

                if not state["present"]:
                    save_event(
                        "PERSON_ENTERED",
                        confidence,
                        person_id=stable_id,
                        shirt_color=shirt_color
                    )

                    state["present"] = True
                    state["enter_time"] = now
                    state["present_logged"] = False
                    state["loitering_logged"] = False

                if state["present"] and state["enter_time"] is not None and not state["present_logged"]:
                    seconds_present = (now - state["enter_time"]).total_seconds()

                    if seconds_present >= PERSON_PRESENT_SECONDS:
                        save_event(
                            "PERSON_PRESENT",
                            confidence,
                            person_id=stable_id,
                            shirt_color=shirt_color
                        )

                        state["present_logged"] = True

                if state["present"] and state["enter_time"] is not None and not state["loitering_logged"]:
                    seconds_present = (now - state["enter_time"]).total_seconds()

                    if seconds_present >= LOITERING_SECONDS:
                        snapshot_file = save_snapshot(
                            annotated_frame,
                            f"{CAMERA_ID}_LOITERING_PERSON_{stable_id}"
                        )

                        video_clip_file = save_video_clip(
                            frame_buffer,
                            f"{CAMERA_ID}_LOITERING_PERSON_{stable_id}"
                        )

                        save_event(
                            "LOITERING",
                            confidence,
                            snapshot_file,
                            video_clip_file,
                            person_id=stable_id,
                            shirt_color=shirt_color
                        )

                        state["loitering_logged"] = True

            for stable_id, state in list(person_states.items()):
                if state["present"] and stable_id not in current_seen_ids:
                    missing_seconds = (now - state["last_seen"]).total_seconds()

                    if missing_seconds >= MISSING_GRACE_SECONDS:
                        save_event(
                            "PERSON_LEFT",
                            1.0,
                            person_id=stable_id,
                            shirt_color=state["shirt_color"]
                        )

                        state["present"] = False
                        state["enter_time"] = None
                        state["present_logged"] = False
                        state["loitering_logged"] = False

            cv2.imshow("AI CCTV - Multi Person ReID", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(1 / CLIP_FPS)

        except requests.exceptions.RequestException as e:
            print("Camera connection error:", e)
            time.sleep(1)

        except Exception as e:
            print("Error:", e)
            time.sleep(1)

finally:
    conn.close()
    cv2.destroyAllWindows()