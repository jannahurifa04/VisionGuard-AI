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
from memory import save_incident_memory
from watchlist import add_watchlist

# =========================
# SETTINGS
# =========================

MODEL_PATH = "yolo11n.pt"
CAMERA_URL = "http://192.168.18.134:8080/shot.jpg"
DATABASE_NAME = "events.db"

CONFIDENCE_LIMIT = 0.4
PERSON_PRESENT_SECONDS = 30
LOITERING_SECONDS = 5           # testing: 5 seconds
MISSING_GRACE_SECONDS = 3       # prevent flickering enter/left

ROTATE_FRAME = True

SNAPSHOT_FOLDER = "snapshots"
CLIP_FOLDER = "clips"

CLIP_SECONDS = 10               # save latest 10 seconds as evidence
CLIP_FPS = 10                   # video output FPS

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
    shirt_color TEXT
)
""")
conn.commit()

# Add missing columns to old database if they do not exist yet
try:
    cursor.execute("ALTER TABLE events ADD COLUMN snapshot TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass

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
    cursor.execute("ALTER TABLE events ADD COLUMN shirt_color TEXT")
    conn.commit()
except sqlite3.OperationalError:
    pass


def save_event(event_name, confidence=1.0, snapshot=None, video_clip=None, person_id=None, shirt_color=None):
    event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO events (event_time, object_name, confidence, snapshot, video_clip, person_id, shirt_color)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (event_time, event_name, round(confidence, 2), snapshot, video_clip, person_id, shirt_color),
    )
    conn.commit()

    event_id = cursor.lastrowid

    memory_text = f"""
Event: {event_name}
Person ID: {person_id}
Shirt Color: {shirt_color}
Time: {event_time}
Snapshot: {snapshot}
Video Clip: {video_clip}
"""

    try:
        save_incident_memory(
            event_id,
            memory_text,
            {
                "event": str(event_name),
                "person_id": str(person_id),
                "shirt_color": str(shirt_color),
                "time": str(event_time),
                "snapshot": str(snapshot),
                "video_clip": str(video_clip),
            },
        )
    except Exception as e:
        print("[MEMORY ERROR]", e)


    # Auto Watchlist: add person if repeated loitering
    if event_name == "LOITERING" and person_id is not None:
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
            print(f"[AUTO WATCHLIST] Person ID {person_id} added. Loitering count: {loiter_count}")

    if person_id is not None:
        print(f"[SMART EVENT] {event_time} - {event_name} - Person ID {person_id}")
    else:
        print(f"[SMART EVENT] {event_time} - {event_name}")

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

        print(f"[VIDEO CLIP SAVED - BROWSER COMPATIBLE] {filepath}")
        return filename

    except Exception as e:
        print("[VIDEO CLIP ERROR]", e)
        return None


def detect_shirt_color(person_crop):
    """
    Upper-body shirt color detector.
    Returns: Black, White, Red, Pink, Green, Blue, Yellow, Gray, Other, Unknown.
    """
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

        # remove very dark/very bright pixels
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

        # OpenCV hue range is 0-179
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
# PERSON STATE
# =========================

person_present = False
person_enter_time = None
person_present_logged = False
loitering_logged = False

last_person_seen_time = None
active_person_id = None
active_shirt_color = None

# Keep recent frames in memory for evidence clips
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
                tracker="bytetrack.yaml"
            )

            current_person_found = False
            best_person_confidence = 0.0
            person_count = 0
            current_track_id = None
            current_shirt_color = None

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])

                if class_name == "person":

                    if box.id is not None:
                        current_track_id = int(box.id[0])

                    person_crop = crop_person_from_box(frame, box)
                    current_shirt_color = detect_shirt_color(person_crop)

                    current_person_found = True
                    person_count += 1
                    best_person_confidence = max(best_person_confidence, confidence)

                    print(
                        f"[TRACK] Person ID={current_track_id} "
                        f"Conf={confidence:.2f} "
                        f"Shirt={current_shirt_color}"
                    )

            now = datetime.now()

            annotated_frame = results[0].plot()

            cv2.putText(
                annotated_frame,
                f"People Count: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            if current_track_id is not None:
                cv2.putText(
                    annotated_frame,
                    f"Stable Person ID: {active_person_id}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

            if current_shirt_color:
                cv2.putText(
                    annotated_frame,
                    f"Shirt Color: {current_shirt_color}",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )

            # Add latest annotated frame into clip buffer
            frame_buffer.append(annotated_frame.copy())

            if current_person_found:
                last_person_seen_time = now

                if active_person_id is None and person_crop is not None:
                    active_person_id = get_stable_person_id(person_crop, current_shirt_color)

                if active_shirt_color is None and current_shirt_color:
                    active_shirt_color = current_shirt_color

            # PERSON ENTERED
            if current_person_found and not person_present:
                if active_person_id is None and person_crop is not None:
                    active_person_id = get_stable_person_id(person_crop, current_shirt_color)
                active_shirt_color = current_shirt_color
                                
                save_event(
                    "PERSON_ENTERED",
                    best_person_confidence,
                    person_id=active_person_id,
                    shirt_color=active_shirt_color
                )

                person_present = True
                person_enter_time = now
                person_present_logged = False
                loitering_logged = False

            # PERSON PRESENT
            if person_present and person_enter_time is not None and not person_present_logged:
                seconds_present = (now - person_enter_time).total_seconds()

                if seconds_present >= PERSON_PRESENT_SECONDS:
                    save_event(
                        "PERSON_PRESENT",
                        best_person_confidence,
                        person_id=active_person_id,
                        shirt_color=active_shirt_color
                    )
                    person_present_logged = True

            # LOITERING
            if person_present and person_enter_time is not None and not loitering_logged:
                seconds_present = (now - person_enter_time).total_seconds()

                if seconds_present >= LOITERING_SECONDS:
                    snapshot_file = save_snapshot(annotated_frame, "LOITERING")
                    video_clip_file = save_video_clip(frame_buffer, "LOITERING")

                    save_event(
                        "LOITERING",
                        best_person_confidence,
                        snapshot_file,
                        video_clip_file,
                        person_id=active_person_id,
                        shirt_color=active_shirt_color
                    )

                    loitering_logged = True

            # PERSON LEFT with grace period
            if not current_person_found and person_present:
                if last_person_seen_time is not None:
                    missing_seconds = (now - last_person_seen_time).total_seconds()

                    if missing_seconds >= MISSING_GRACE_SECONDS:
                        save_event(
                            "PERSON_LEFT",
                            1.0,
                            person_id=active_person_id,
                            shirt_color=active_shirt_color
                        )

                        person_present = False
                        person_enter_time = None
                        person_present_logged = False
                        loitering_logged = False
                        last_person_seen_time = None
                        active_person_id = None
                        active_shirt_color = None

            cv2.imshow("AI CCTV", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Keeps capture speed close to CLIP_FPS
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
