from ultralytics import YOLO
import cv2
from fashn_human_parser import FashnHumanParser
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
import requests
import numpy as np
from datetime import datetime
import sqlite3
import time
import os
import json
from collections import deque
import imageio.v2 as imageio
from reid import get_stable_person_id, extract_embedding, clear_pending_identity
from watchlist import add_watchlist
from person_profile import ensure_person_profile
from camera_manager import ensure_camera_table, get_cameras
from face_memory import (
    recognize_face,
    register_face_embedding,
    update_verified_face_gallery,
    get_face_display_name,
    get_face_profile,
)

def gray_world_white_balance(pil_image, mask=None):
    """Gray-world white balance - normalizes lighting color cast.
    Must match exactly what train_clothing_color.py applies during training.
    If mask is given (2D array, same H/W as image, >0 = clothing pixels),
    only those pixels are used to estimate the correction, so large
    neutral-gray filler areas don't dilute the estimate."""
    img = np.asarray(pil_image).astype(np.float32)

    if mask is not None and np.count_nonzero(mask) > 50:
        region = img[mask > 0]
        avg_r = np.mean(region[:, 0])
        avg_g = np.mean(region[:, 1])
        avg_b = np.mean(region[:, 2])
    else:
        avg_r = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_b = np.mean(img[:, :, 2])

    avg_gray = (avg_r + avg_g + avg_b) / 3.0

    img[:, :, 0] *= (avg_gray / (avg_r + 1e-6))
    img[:, :, 1] *= (avg_gray / (avg_g + 1e-6))
    img[:, :, 2] *= (avg_gray / (avg_b + 1e-6))

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

# =========================
# SETTINGS
# =========================

MODEL_PATH = "yolo11n.pt"

CAMERA_ID = "cam_1"
CAMERA_NAME = "Front Camera"
CAMERA_URL = "http://192.168.18.134:8080/shot.jpg"

DATABASE_NAME = "events.db"

ensure_camera_table()

def get_enabled_camera_configs():
    camera_configs = []

    for (
        camera_id,
        camera_code,
        camera_name,
        camera_url,
        enabled,
    ) in get_cameras():

        if enabled:
            camera_configs.append(
                {
                    "id": camera_id,
                    "code": camera_code,
                    "name": camera_name,
                    "url": camera_url,
                }
            )

    return camera_configs

def make_track_key(camera_code, temp_track_id):
    return f"{camera_code}:{temp_track_id}"

CONFIDENCE_LIMIT = 0.4
PERSON_PRESENT_SECONDS = 30
LOITERING_SECONDS = 5
MISSING_GRACE_SECONDS = 10
MIN_REID_FRAMES = 5

# Reconnect a new ByteTrack ID to a person seen moments earlier.
TRACK_RECOVERY_SECONDS = 30
TRACK_RECOVERY_THRESHOLD = 0.68
TRACK_RECOVERY_MARGIN = 0.03
TRACK_MAPPING_EXPIRE_SECONDS = 45

# A new permanent BODY identity cannot be created immediately.
# During this grace period the visible box says "Identifying..." while
# VisionGuard keeps trying face recognition and recent-track recovery.
NEW_ID_FACE_GRACE_SECONDS = 6.0

# Face checks for a pending/new track.
PENDING_FACE_CHECK_INTERVAL_SECONDS = 0.35

# A BODY identity can be overridden by a verified familiar face.
FACE_OVERRIDE_MIN_SCORE = 0.58
FACE_OVERRIDE_CONFIRMATIONS = 2

# Do not create a face profile from one unmatched face frame.
UNKNOWN_FACE_REGISTRATION_MIN_FRAMES = 4
UNKNOWN_FACE_BUFFER_SIZE = 8

# Avoid writing the same person profile to SQLite every frame.
PROFILE_UPDATE_INTERVAL_SECONDS = 5

# Face recognition is the primary permanent identity source.
# Locked tracks are rechecked periodically to improve face memory.
LOCKED_FACE_CHECK_INTERVAL_SECONDS = 0.5

# Shirt-colour detection: all-colours + lighting-aware version.
# Shirt-colour detection uses several frames instead of trusting one frame.
# Deep-learning upper-clothing isolation.
USE_DEEP_CLOTHING_MASK = True
CLOTHING_TOP_CLASS_ID = 3
CLOTHING_MASK_MIN_PIXELS = 250
CLOTHING_PARSER_INTERVAL_SECONDS = 1.0
CLOTHING_MASK_CACHE_SECONDS = 8.0
SHIRT_COLOR_HISTORY_SIZE = 30
SHIRT_COLOR_HISTORY_MAX_AGE_SECONDS = 12
SHIRT_COLOR_MIN_RAW_CONFIDENCE = 0.38

# Do not display a guessed colour immediately.
SHIRT_COLOR_WARMUP_SECONDS = 1.8
SHIRT_COLOR_MIN_VOTES = 6
SHIRT_COLOR_STABLE_RATIO = 0.64

# Once a colour is locked for the current visit, changing it requires
# much stronger evidence.
SHIRT_COLOR_SWITCH_MIN_VOTES = 12
SHIRT_COLOR_SWITCH_RATIO = 0.82

# Keep a confirmed colour through a short leave-and-return sequence.
# It expires later so clothing can still change on a future visit.
SHIRT_COLOR_MEMORY_SECONDS = 90

ROTATE_FRAME = True

SNAPSHOT_FOLDER = "snapshots"
CLIP_FOLDER = "clips"

CLIP_SECONDS = 10
CLIP_FPS = 10

os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
os.makedirs(CLIP_FOLDER, exist_ok=True)

# Shared processed frame used by the FastAPI dashboard.
RUNTIME_FOLDER = "runtime"
PROCESSED_FRAME_PATH = os.path.join(
    RUNTIME_FOLDER,
    "latest_annotated.jpg",
)
LIVE_STATUS_PATH = os.path.join(
    RUNTIME_FOLDER,
    "live_status.json",
)
PROCESSED_FRAME_JPEG_QUALITY = 88

os.makedirs(RUNTIME_FOLDER, exist_ok=True)

def get_camera_processed_frame_path(camera_code):
    return os.path.join(
        RUNTIME_FOLDER,
        f"{camera_code}_latest_annotated.jpg",
    )


def get_camera_live_status_path(camera_code):
    return os.path.join(
        RUNTIME_FOLDER,
        f"{camera_code}_live_status.json",
    )

# =========================
# LOAD YOLO MODEL
# =========================


camera_models = {}

def get_camera_model(camera_code):
    if camera_code not in camera_models:
        camera_models[camera_code] = YOLO(MODEL_PATH)
        print(f"[CAMERA AI] YOLO tracker created for {camera_code}")

    return camera_models[camera_code]

clothing_parser = FashnHumanParser()
print("[CLOTHING PARSER] FASHN model loaded")

# =========================
# DEEP CLOTHING COLOR MODEL
# =========================

COLOR_MODEL_PATH = "clothing_color_classifier.pth"
COLOR_UNKNOWN_THRESHOLD = 0.70

COLOR_DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

color_checkpoint = torch.load(
    COLOR_MODEL_PATH,
    map_location=COLOR_DEVICE,
    weights_only=False,
)

COLOR_CLASSES = color_checkpoint["classes"]

COLOR_IMAGE_SIZE = color_checkpoint.get(
    "image_size",
    224,
)

clothing_color_model = mobilenet_v3_small(
    weights=None
)

color_input_features = (
    clothing_color_model.classifier[3].in_features
)

clothing_color_model.classifier[3] = torch.nn.Linear(
    color_input_features,
    len(COLOR_CLASSES),
)

clothing_color_model.load_state_dict(
    color_checkpoint["state_dict"]
)

clothing_color_model = clothing_color_model.to(
    COLOR_DEVICE
)

clothing_color_model.eval()

color_transform = transforms.Compose(
    [
        transforms.Resize(
            (
                COLOR_IMAGE_SIZE,
                COLOR_IMAGE_SIZE,
            )
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)

print(
    f"[CLOTHING COLOR AI] Model loaded | "
    f"Device={COLOR_DEVICE} | "
    f"Classes={COLOR_CLASSES}"
)

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


def classify_neutral_shirt(
    value_median,
    saturation_median=0.0,
    value_high=None,
    value_peak=None,
    bright_fraction=0.0,
    very_bright_fraction=0.0,
):
    """
    Classify Black / Gray / White using several brightness statistics.

    A white shirt can have a medium median value when one side is in shadow.
    Using the brighter reliable pixels as well as the median prevents white
    fabric from being labelled Gray while still keeping real gray and black
    clothing separate.
    """
    v50 = float(value_median)
    s50 = float(saturation_median)
    v75 = float(value_high if value_high is not None else v50)
    v90 = float(value_peak if value_peak is not None else v75)
    bright_fraction = float(bright_fraction)
    very_bright_fraction = float(very_bright_fraction)

    # Very dark fabric remains dark even in its brighter pixels.
    if v75 < 48 and v90 < 78:
        return "Black"

    if v75 < 82 and v90 < 118:
        return "Charcoal"

    # White / off-white fabric often contains both bright highlights and
    # darker folds. Require low saturation plus enough genuinely bright
    # pixels, rather than requiring the whole crop to be bright.
    # Lighting-aware White detection.
    # A white shirt may look gray when the room is dark, but its brighter
    # folds and highlights still provide evidence that the fabric is white.

    white_evidence = (
        s50 <= 68
        and (
            # Normally illuminated white clothing.
            v50 >= 165

            # White clothing with medium shadows.
            or (
                v75 >= 145
                and v90 >= 185
                and bright_fraction >= 0.08
            )

            # Strong highlight evidence from heavily shadowed white clothing.
            or (
                v75 >= 135
                and v90 >= 200
                and very_bright_fraction >= 0.03
            )

            # Uneven indoor lighting: dark median but bright upper pixels.
            or (
                v50 >= 120
                and v75 >= 150
                and v90 >= 195
                and (v90 - v50) >= 38
            )
        )
    )

    if white_evidence:
        if (
            v50 >= 188
            or bright_fraction >= 0.46
            or very_bright_fraction >= 0.24
        ):
            return "White"

        return "Off White"

    if v50 < 108 or v75 < 126:
        return "Dark Gray"

    if v50 < 154 or v75 < 172:
        return "Gray"

    if s50 <= 52 and v90 >= 198:
        return "Off White"

    return "Light Gray"


def classify_extended_colour(hue, saturation, value, chroma):
    """
    Lighting-aware broad clothing-colour classifier.

    Hue is used as the main signal because it changes less than brightness
    under normal indoor lighting. Saturation, value and Lab chroma are used
    only to separate pale colours, brown, pink and neutral clothing.

    OpenCV hue range:
        0..179
    """
    h = float(hue)
    s = float(saturation)
    v = float(value)
    c = float(chroma)

    if s < 24 and c < 8:
        return "Unknown"

    # Pale warm fabric: cream, beige, tan and khaki.
    if 7 <= h < 46 and s < 88 and v >= 102:
        return "Beige"

    # Red / pink. Pink is normally brighter and less saturated than red.
    if h < 12 or h >= 168:
        pink_like = (
            v >= 100
            and (
                (s <= 150 and c <= 48)
                or (v >= 145 and s <= 182 and c <= 58)
                or (s >= 100 and v >= 140)
            )
        )

        if pink_like:
            return "Pink"

        return "Red"

    # Warm red-orange shades.
    if 12 <= h < 24:
        if v < 78 and s < 175:
            return "Brown"

        if s < 92 and v >= 100:
            return "Beige"

        return "Orange"

    # Orange / yellow / brown.
    if 24 <= h < 42:
        if v < 68 and s < 145:
            return "Brown"

        if s < 82 and v >= 100:
            return "Beige"

        return "Yellow"

    if 42 <= h < 88:
        return "Green"

    if 88 <= h < 103:
        return "Teal"

    if 103 <= h < 138:
        return "Blue"

    if 138 <= h < 168:
        return "Purple"

    return "Unknown"


def canonicalize_shirt_colour(colour):
    """
    Convert related shades into stable dashboard labels.

    The dashboard intentionally shows broad colour families. This avoids
    flickering between similar names such as Off White / White, Navy / Blue,
    or Maroon / Red when lighting changes.
    """
    groups = {
        "Black": {
            "Black",
        },
        "Gray": {
            "Charcoal", "Dark Gray", "Gray",
            "Silver", "Light Gray",
        },
        "White": {
            "Off White", "White", "Ivory", "Cream",
        },
        "Beige": {
            "Beige", "Tan", "Khaki", "Camel",
        },
        "Brown": {
            "Brown", "Dark Brown", "Rust",
        },
        "Pink": {
            "Pink", "Rose", "Magenta",
            "Coral", "Salmon", "Peach",
        },
        "Red": {
            "Red", "Dark Red", "Maroon", "Burgundy",
        },
        "Orange": {
            "Orange",
        },
        "Yellow": {
            "Yellow", "Mustard", "Gold",
        },
        "Green": {
            "Green", "Dark Green", "Lime",
            "Mint", "Sage", "Yellow Green",
            "Olive",
        },
        "Teal": {
            "Teal", "Dark Teal", "Turquoise", "Cyan",
        },
        "Blue": {
            "Sky Blue", "Blue", "Deep Blue",
            "Royal Blue", "Navy", "Indigo",
        },
        "Purple": {
            "Purple", "Dark Purple",
            "Violet", "Lavender",
        },
        "Multicolor": {
            "Multicolor",
        },
    }

    for visible_colour, members in groups.items():
        if colour in members:
            return visible_colour

    return colour


def build_person_foreground_mask(person_crop):
    """
    Separate the person from the background using GrabCut.

    The bounding box contains kitchen walls, cupboards and chairs.
    Those background pixels were the main reason a pink shirt became Gray.
    """
    height, width = person_crop.shape[:2]

    if height < 45 or width < 25:
        return np.ones(
            (height, width),
            dtype=np.uint8,
        ) * 255

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )
    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    margin_x = max(2, int(width * 0.04))
    margin_y = max(2, int(height * 0.03))

    rectangle = (
        margin_x,
        margin_y,
        max(2, width - (2 * margin_x)),
        max(2, height - (2 * margin_y)),
    )

    try:
        cv2.grabCut(
            person_crop,
            mask,
            rectangle,
            background_model,
            foreground_model,
            2,
            cv2.GC_INIT_WITH_RECT,
        )

        foreground = np.where(
            (mask == cv2.GC_FGD)
            | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        if np.count_nonzero(foreground) < (height * width * 0.12):
            raise ValueError("Foreground mask too small.")

        return foreground

    except Exception:
        # Safe fallback: use a central body-shaped ellipse rather than
        # accepting the complete rectangular background.
        foreground = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        cv2.ellipse(
            foreground,
            (width // 2, int(height * 0.48)),
            (
                max(4, int(width * 0.39)),
                max(6, int(height * 0.47)),
            ),
            0,
            0,
            360,
            255,
            -1,
        )

        return foreground

def build_deep_upper_clothing_mask(person_crop, cache_key=None):
    """
    Use FASHN human parsing to isolate only the upper clothing.
    """
    if not USE_DEEP_CLOTHING_MASK:
        return None

    try:
        if person_crop is None or person_crop.size == 0:
            return None

        rgb_crop = cv2.cvtColor(
            person_crop,
            cv2.COLOR_BGR2RGB,
        )

        segmentation = clothing_parser.predict(rgb_crop)

        clothing_mask = np.where(
            segmentation == CLOTHING_TOP_CLASS_ID,
            255,
            0,
        ).astype(np.uint8)

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        clothing_mask = cv2.morphologyEx(
            clothing_mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        clothing_mask = cv2.morphologyEx(
            clothing_mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        if (
            cv2.countNonZero(clothing_mask)
            < CLOTHING_MASK_MIN_PIXELS
        ):
            return None

        if cache_key is not None:
            clothing_mask_cache[cache_key] = clothing_mask.copy()
            clothing_mask_cache_time[cache_key] = time.time()

        return clothing_mask

    except Exception as error:
        print(f"[CLOTHING PARSER ERROR] {error}")
        return None

def predict_deep_clothing_color(person_crop, clothing_mask):
    """
    Predict clothing colour using the trained MobileNet classifier.

    Returns:
        (colour_name, confidence)
    """
    try:
        if (
            person_crop is None
            or person_crop.size == 0
            or clothing_mask is None
        ):
            return "Unknown", 0.0

        crop_height, crop_width = person_crop.shape[:2]

        # Make sure the mask matches the current person crop.
        if clothing_mask.shape[:2] != (crop_height, crop_width):
            clothing_mask = cv2.resize(
                clothing_mask,
                (crop_width, crop_height),
                interpolation=cv2.INTER_NEAREST,
            )

        points = cv2.findNonZero(clothing_mask)

        if points is None:
            return "Unknown", 0.0

        x, y, w, h = cv2.boundingRect(points)

        if w < 30 or h < 40:
            return "Unknown", 0.0

        clothing_crop = person_crop[
            y:y + h,
            x:x + w,
        ]

        clothing_mask_crop = clothing_mask[
            y:y + h,
            x:x + w,
        ]

        # Same neutral background used during training.
        classifier_image = np.full_like(
            clothing_crop,
            127,
        )

        classifier_image[
            clothing_mask_crop > 0
        ] = clothing_crop[
            clothing_mask_crop > 0
        ]

        rgb_image = cv2.cvtColor(
            classifier_image,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(
            rgb_image
        )

        pil_image = gray_world_white_balance(pil_image, mask=clothing_mask_crop)

        tensor = color_transform(
            pil_image
        ).unsqueeze(0)

        tensor = tensor.to(
            COLOR_DEVICE
        )

        with torch.no_grad():
            output = clothing_color_model(
                tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1,
            )

            confidence, class_index = torch.max(
                probabilities,
                dim=1,
            )

        confidence = float(
            confidence.item()
        )

        colour_name = COLOR_CLASSES[
            int(class_index.item())
        ]

        if confidence < COLOR_UNKNOWN_THRESHOLD:
            return "Unknown", confidence

        return colour_name, confidence

    except Exception as error:
        print(
            f"[CLOTHING COLOR AI ERROR] {error}"
        )
        return "Unknown", 0.0

def detect_shirt_color(person_crop, cache_key=None):
    """
    Detect shirt colour from the central upper torso.

    Improvements in this version:
      * samples the chest instead of the complete person box;
      * removes most background, hair and skin pixels;
      * uses hue for coloured clothing so shadows change brightness, not colour;
      * uses brightness percentiles for Black / Gray / White;
      * supports pale colours such as pink, beige and light blue;
      * returns one of the broad stable dashboard colour families.

    Returns:
        (colour_name, confidence)
    """
    try:
        if person_crop is None or person_crop.size == 0:
            return "Unknown", 0.0

        original_height, original_width = person_crop.shape[:2]

        if original_height < 90 or original_width < 40:
            return "Unknown", 0.0

        working_height = 256
        working_width = max(
            96,
            min(
                192,
                int(
                    working_height
                    * original_width
                    / max(original_height, 1)
                ),
            ),
        )

        resized_person = cv2.resize(
            person_crop,
            (working_width, working_height),
            interpolation=cv2.INTER_AREA,
        )

        deep_clothing_mask = build_deep_upper_clothing_mask(
            person_crop,
            cache_key=cache_key,
        )
        
        
        # Use the current FASHN mask for the deep colour classifier.
        classifier_mask = deep_clothing_mask

        # If FASHN misses this frame, use the recent cached FASHN mask.
        if (
            classifier_mask is None
            and cache_key is not None
            and cache_key in clothing_mask_cache
            and cache_key in clothing_mask_cache_time
            and (
                time.time()
                - clothing_mask_cache_time[cache_key]
            ) <= CLOTHING_MASK_CACHE_SECONDS
        ):
            classifier_mask = clothing_mask_cache[cache_key]

        # Deep-learning colour prediction is now the PRIMARY detector.
        if classifier_mask is not None:
            ai_colour, ai_confidence = predict_deep_clothing_color(
                person_crop,
                classifier_mask,
            )

            if ai_colour != "Unknown":
                print(
                    f"[CLOTHING COLOR AI] "
                    f"{ai_colour} Conf={ai_confidence:.2f}"
                )
                return ai_colour, ai_confidence

            # AI wasn't confident enough - fall through to the HSV-based
            # analysis below instead of getting stuck on Unknown forever.

        if deep_clothing_mask is not None:
            print("[CLOTHING MASK] Using FASHN deep mask")

            foreground_mask = cv2.resize(
                deep_clothing_mask,
                (working_width, working_height),
                interpolation=cv2.INTER_NEAREST,
            )

        elif (
            cache_key is not None
            and cache_key in clothing_mask_cache
            and cache_key in clothing_mask_cache_time
            and (
                time.time()
                - clothing_mask_cache_time[cache_key]
            ) <= CLOTHING_MASK_CACHE_SECONDS
        ):
            print("[CLOTHING MASK] Using cached FASHN mask")

            foreground_mask = cv2.resize(
                clothing_mask_cache[cache_key],
                (working_width, working_height),
                interpolation=cv2.INTER_NEAREST,
            )

        else:
            print("[CLOTHING MASK] FASHN failed - using fallback")

            foreground_mask = build_person_foreground_mask(
                resized_person
            )

        # Central chest area. Starting lower than the old version removes
        # more face, neck and hair while still keeping enough shirt pixels.
        y1 = int(working_height * 0.30)
        y2 = int(working_height * 0.69)
        x1 = int(working_width * 0.20)
        x2 = int(working_width * 0.80)

        shirt_region = resized_person[y1:y2, x1:x2]
        foreground_region = foreground_mask[y1:y2, x1:x2]

        if shirt_region is None or shirt_region.size == 0:
            return "Unknown", 0.0

        region_height, region_width = shirt_region.shape[:2]

        shirt_region = cv2.bilateralFilter(
            shirt_region,
            7,
            38,
            38,
        )

        hsv = cv2.cvtColor(
            shirt_region,
            cv2.COLOR_BGR2HSV,
        )
        ycrcb = cv2.cvtColor(
            shirt_region,
            cv2.COLOR_BGR2YCrCb,
        )
        lab = cv2.cvtColor(
            shirt_region,
            cv2.COLOR_BGR2LAB,
        )

        h_channel, s_channel, v_channel = cv2.split(hsv)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        _, a_channel, b_channel = cv2.split(lab)

        # Prefer the centre of the chest. The sides of a person box often
        # contain hair, arms, chair or background.
        centre_mask = np.zeros(
            (region_height, region_width),
            dtype=np.uint8,
        )

        cv2.ellipse(
            centre_mask,
            (
                region_width // 2,
                int(region_height * 0.52),
            ),
            (
                max(5, int(region_width * 0.43)),
                max(5, int(region_height * 0.46)),
            ),
            0,
            0,
            360,
            255,
            -1,
        )

        valid_brightness = (
            (v_channel >= 16)
            & (v_channel <= 252)
        )

        # Skin is removed only in likely neck/arm areas. Applying a skin
        # filter to the entire chest can incorrectly remove pink fabric.
        skin_hue = (
            (h_channel <= 24)
            | (h_channel >= 176)
        )

        likely_skin = (
            (cr_channel >= 133)
            & (cr_channel <= 180)
            & (cb_channel >= 76)
            & (cb_channel <= 135)
            & (y_channel >= 38)
            & skin_hue
            & (s_channel >= 18)
            & (s_channel <= 175)
        )

        skin_zone = np.zeros(
            (region_height, region_width),
            dtype=bool,
        )

        top_zone_end = max(
            1,
            int(region_height * 0.30),
        )
        side_zone_width = max(
            1,
            int(region_width * 0.13),
        )

        skin_zone[
            :top_zone_end,
            int(region_width * 0.20):int(region_width * 0.80),
        ] = True
        skin_zone[:, :side_zone_width] = True
        skin_zone[:, region_width - side_zone_width:] = True

        likely_skin = likely_skin & skin_zone

        valid_mask = (
            (foreground_region > 0)
            & (centre_mask > 0)
            & valid_brightness
            & (~likely_skin)
        )

        if int(np.count_nonzero(valid_mask)) < 220:
            valid_mask = (
                (foreground_region > 0)
                & (centre_mask > 0)
                & valid_brightness
            )

        valid_count = int(np.count_nonzero(valid_mask))

        if valid_count < 160:
            return "Unknown", 0.0

        hue_values = h_channel[valid_mask].astype(np.float32)
        saturation_values = s_channel[valid_mask].astype(np.float32)
        value_values = v_channel[valid_mask].astype(np.float32)

        a_values = (
            a_channel[valid_mask].astype(np.float32)
            - 128.0
        )
        b_values = (
            b_channel[valid_mask].astype(np.float32)
            - 128.0
        )

        chroma_values = np.sqrt(
            (a_values ** 2)
            + (b_values ** 2)
        )

        # Spatial weights make the middle of the chest more important.
        yy, xx = np.mgrid[
            0:region_height,
            0:region_width,
        ].astype(np.float32)

        dx = (
            (xx - (region_width / 2.0))
            / max(region_width / 2.0, 1.0)
        )
        dy = (
            (yy - (region_height * 0.52))
            / max(region_height * 0.52, 1.0)
        )

        distance = np.sqrt(
            (dx ** 2)
            + (dy ** 2)
        )

        spatial_weight_map = np.clip(
            1.15 - (0.38 * distance),
            0.62,
            1.15,
        )
        spatial_values = spatial_weight_map[
            valid_mask
        ].astype(np.float32)

        strong_chromatic = (
            (saturation_values >= 45)
            & (chroma_values >= 12)
        )
        soft_chromatic = (
            (saturation_values >= 24)
            & (chroma_values >= 8)
        )

        strong_fraction = float(
            np.mean(strong_chromatic)
        )
        soft_fraction = float(
            np.mean(soft_chromatic)
        )

        def neutral_result():
            neutral_pixels = (
                (saturation_values < 58)
                | (chroma_values < 18)
            )

            if int(np.count_nonzero(neutral_pixels)) < 80:
                neutral_pixels = np.ones(
                    value_values.shape,
                    dtype=bool,
                )

            neutral_values = value_values[
                neutral_pixels
            ]
            neutral_saturation = saturation_values[
                neutral_pixels
            ]

            v50 = float(
                np.percentile(neutral_values, 50)
            )
            v75 = float(
                np.percentile(neutral_values, 75)
            )
            v90 = float(
                np.percentile(neutral_values, 90)
            )
            s50 = float(
                np.percentile(neutral_saturation, 50)
            )

            bright_fraction = float(
                np.mean(neutral_values >= 180)
            )
            very_bright_fraction = float(
                np.mean(neutral_values >= 215)
            )

            colour = classify_neutral_shirt(
                value_median=v50,
                saturation_median=s50,
                value_high=v75,
                value_peak=v90,
                bright_fraction=bright_fraction,
                very_bright_fraction=very_bright_fraction,
            )

            neutral_consistency = float(
                np.mean(neutral_pixels)
            )

            confidence = min(
                0.98,
                0.56
                + (neutral_consistency * 0.28)
                + (
                    min(valid_count / 1200.0, 1.0)
                    * 0.10
                ),
            )

            return (
                canonicalize_shirt_colour(colour),
                float(confidence),
            )

        # Warm indoor lighting can make a white shirt look yellow/beige.
        # Check for warm white before accepting Beige.
        overall_s50 = float(
            np.percentile(saturation_values, 50)
        )
        overall_v75 = float(
            np.percentile(value_values, 75)
        )
        overall_v90 = float(
            np.percentile(value_values, 90)
        )
        overall_b50 = float(
            np.percentile(b_values, 50)
        )

        warm_white_evidence = (
            overall_s50 <= 58
            and overall_v75 >= 145
            and overall_v90 >= 190
            and strong_fraction < 0.18
            and overall_b50 < 18
        )

        if warm_white_evidence:
            return neutral_result()

        # A real beige shirt must have stronger and more consistent
        # warm colour than a white shirt under yellow lighting.
        warm_beige_pixels = (
            (hue_values >= 7)
            & (hue_values < 46)
            & (saturation_values >= 38)
            & (saturation_values < 105)
            & (value_values >= 85)
            & (value_values <= 235)
            & (b_values >= 7)
            & (chroma_values >= 11)
        )

        warm_beige_fraction = float(
            np.mean(warm_beige_pixels)
        )

        if (
            warm_beige_fraction >= 0.34
            and strong_fraction >= 0.08
            and strong_fraction < 0.35
        ):
            beige_confidence = min(
                0.95,
                0.58
                + (warm_beige_fraction * 0.55),
            )
            return "Beige", float(beige_confidence)

        use_chromatic_pixels = (
            strong_fraction >= 0.09
            or soft_fraction >= 0.24
        )

        if not use_chromatic_pixels:
            return neutral_result()

        voting_indices = soft_chromatic

        selected_hues = hue_values[
            voting_indices
        ]
        selected_saturations = saturation_values[
            voting_indices
        ]
        selected_values = value_values[
            voting_indices
        ]
        selected_chroma = chroma_values[
            voting_indices
        ]
        selected_spatial = spatial_values[
            voting_indices
        ]

        colour_weights = {}
        colour_counts = {}

        for (
            hue,
            saturation,
            value,
            chroma,
            spatial_weight,
        ) in zip(
            selected_hues,
            selected_saturations,
            selected_values,
            selected_chroma,
            selected_spatial,
        ):
            colour = classify_extended_colour(
                hue,
                saturation,
                value,
                chroma,
            )
            colour = canonicalize_shirt_colour(
                colour
            )

            if colour in (
                "Unknown",
                None,
                "",
            ):
                continue

            saturation_weight = (
                0.45
                + (
                    0.55
                    * min(
                        float(saturation) / 150.0,
                        1.0,
                    )
                )
            )

            chroma_weight = (
                0.55
                + (
                    0.45
                    * min(
                        float(chroma) / 42.0,
                        1.0,
                    )
                )
            )

            brightness_weight = 1.0

            if value < 28:
                brightness_weight = 0.62
            elif value < 52:
                brightness_weight = 0.82
            elif value > 246:
                brightness_weight = 0.76
            elif value > 232:
                brightness_weight = 0.90

            weight = (
                saturation_weight
                * chroma_weight
                * brightness_weight
                * float(spatial_weight)
            )

            colour_weights[colour] = (
                colour_weights.get(colour, 0.0)
                + weight
            )
            colour_counts[colour] = (
                colour_counts.get(colour, 0)
                + 1
            )

        if not colour_weights:
            return neutral_result()

        ranked_colours = sorted(
            colour_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_colour, best_weight = ranked_colours[0]
        total_weight = float(
            sum(colour_weights.values())
        )
        dominance = (
            best_weight / total_weight
            if total_weight > 0
            else 0.0
        )

        # Low-saturation colour is accepted only when its hue is coherent.
        # Otherwise the small hue changes are probably neutral camera noise.
        if (
            strong_fraction < 0.09
            and dominance < 0.62
        ):
            return neutral_result()

        # When two related families are almost tied, prefer the broad family
        # supported by more pixels rather than reacting to a tiny highlight.
        if len(ranked_colours) >= 2:
            second_colour, second_weight = ranked_colours[1]
            second_ratio = (
                second_weight / total_weight
                if total_weight > 0
                else 0.0
            )

            if (
                dominance - second_ratio < 0.05
                and colour_counts.get(
                    second_colour,
                    0,
                )
                > colour_counts.get(
                    best_colour,
                    0,
                )
            ):
                best_colour = second_colour
                best_weight = second_weight
                dominance = second_ratio

        confidence = min(
            0.98,
            0.34
            + (dominance * 0.50)
            + (min(strong_fraction, 0.50) * 0.22)
            + (min(valid_count / 1400.0, 1.0) * 0.08),
        )

        if confidence < 0.38:
            return "Unknown", float(confidence)

        return best_colour, float(confidence)

    except Exception as error:
        print(f"[SHIRT COLOR ERROR] {error}")
        return "Unknown", 0.0


def reset_shirt_colour_visit(person_id):
    """
    End the current colour-analysis visit without forgetting a confirmed
    colour immediately.

    This keeps Pink stable when a person leaves for a few seconds and
    returns from another angle. The confirmed colour expires later.
    """
    shirt_color_histories.pop(person_id, None)
    shirt_color_visit_started.pop(person_id, None)
    shirt_color_last_seen[person_id] = datetime.now()


def stabilize_shirt_color(
    person_id,
    raw_colour,
    raw_confidence,
    now,
):
    """
    Analyse first, then lock one visible colour for the current visit.
    """
    if person_id is None:
        return "Analyzing"

    previous_last_seen = shirt_color_last_seen.get(person_id)

    if previous_last_seen is not None:
        memory_age = (
            now - previous_last_seen
        ).total_seconds()

        if memory_age > SHIRT_COLOR_MEMORY_SECONDS:
            stable_shirt_colors.pop(person_id, None)
            shirt_color_histories.pop(person_id, None)
            shirt_color_visit_started.pop(person_id, None)

    shirt_color_last_seen[person_id] = now

    visible_colour = canonicalize_shirt_colour(
        raw_colour
    )

    history = shirt_color_histories.setdefault(
        person_id,
        deque(maxlen=SHIRT_COLOR_HISTORY_SIZE),
    )

    visit_started = shirt_color_visit_started.setdefault(
        person_id,
        now,
    )

    while history:
        age_seconds = (
            now - history[0][2]
        ).total_seconds()

        if age_seconds <= SHIRT_COLOR_HISTORY_MAX_AGE_SECONDS:
            break

        history.popleft()

    if (
        visible_colour not in (
            None,
            "",
            "Unknown",
            "Analyzing",
        )
        and raw_confidence >= SHIRT_COLOR_MIN_RAW_CONFIDENCE
    ):
        history.append(
            (
                visible_colour,
                float(raw_confidence),
                now,
            )
        )

    elapsed = (
        now - visit_started
    ).total_seconds()

    # Never expose a one-frame guess.
    if (
        elapsed < SHIRT_COLOR_WARMUP_SECONDS
        or len(history) < SHIRT_COLOR_MIN_VOTES
    ):
        return stable_shirt_colors.get(
            person_id,
            "Analyzing",
        )

    weighted_votes = {}
    vote_counts = {}

    for index, (
        colour,
        confidence,
        _,
    ) in enumerate(history):
        recency_weight = (
            0.70
            + (
                0.30
                * ((index + 1) / len(history))
            )
        )

        weight = confidence * recency_weight

        weighted_votes[colour] = (
            weighted_votes.get(colour, 0.0)
            + weight
        )
        vote_counts[colour] = (
            vote_counts.get(colour, 0)
            + 1
        )

    candidate = max(
        weighted_votes,
        key=weighted_votes.get,
    )

    total_weight = sum(
        weighted_votes.values()
    )
    candidate_ratio = (
        weighted_votes[candidate] / total_weight
        if total_weight > 0
        else 0.0
    )

    locked_colour = stable_shirt_colors.get(
        person_id
    )

    if locked_colour is None:
        enough_initial_evidence = (
            vote_counts.get(candidate, 0)
            >= SHIRT_COLOR_MIN_VOTES
            and candidate_ratio
            >= SHIRT_COLOR_STABLE_RATIO
        )

        if not enough_initial_evidence:
            return "Analyzing"

        stable_shirt_colors[person_id] = candidate

        print(
            f"[SHIRT LOCKED] Person ID {person_id} "
            f"Colour={candidate} "
            f"Votes={vote_counts[candidate]} "
            f"Ratio={candidate_ratio:.2f}"
        )

        return candidate

    if candidate == locked_colour:
        return locked_colour

    neutral_colours = {
        "Black",
        "Gray",
        "White",
        "Cream",
        "Beige",
    }

    locked_is_neutral = locked_colour in neutral_colours
    candidate_is_neutral = candidate in neutral_colours

    candidate_votes = vote_counts.get(candidate, 0)

    # A bright coloured shirt should be allowed to correct an earlier
    # neutral/background mistake relatively quickly.
    if locked_is_neutral and not candidate_is_neutral:
        required_votes = 8
        required_ratio = 0.72
        required_seconds = 2.5

    # A real clothing change to Black/Gray/White must be recognised.
    # The previous values were intentionally very strict to block gray
    # background mistakes, but they also kept an old Pink result after the
    # person returned wearing a black shirt.
    elif not locked_is_neutral and candidate_is_neutral:
        if candidate == "Black":
            required_votes = 9
            required_ratio = 0.76
            required_seconds = 2.8
        else:
            required_votes = 12
            required_ratio = 0.82
            required_seconds = 3.8

    else:
        required_votes = 12
        required_ratio = 0.82
        required_seconds = 3.8

    strong_switch_evidence = (
        elapsed >= required_seconds
        and candidate_votes >= required_votes
        and candidate_ratio >= required_ratio
    )

    if strong_switch_evidence:
        print(
            f"[SHIRT CORRECTED] Person ID {person_id} "
            f"{locked_colour} -> {candidate} "
            f"Votes={candidate_votes} "
            f"Ratio={candidate_ratio:.2f}"
        )

        stable_shirt_colors[person_id] = candidate
        return candidate

    return locked_colour


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
# SHORT-TERM TRACK RECOVERY
# =========================

def normalize_recovery_embedding(embedding):
    if embedding is None:
        return None

    embedding = np.asarray(embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(embedding)

    if norm <= 0:
        return None

    return embedding / norm


def find_recent_track_match(
    embedding,
    now,
    excluded_stable_ids=None,
):
    """Find a recently visible stable ID for a new ByteTrack track."""
    embedding = normalize_recovery_embedding(embedding)

    if embedding is None:
        return None, -1.0, -1.0

    excluded_stable_ids = excluded_stable_ids or set()
    candidate_scores = []

    for stable_id, recent_data in recent_stable_tracks.items():
        if stable_id in excluded_stable_ids:
            continue

        age_seconds = (
            now - recent_data["last_seen"]
        ).total_seconds()

        if age_seconds < 0 or age_seconds > TRACK_RECOVERY_SECONDS:
            continue

        old_embedding = normalize_recovery_embedding(
            recent_data.get("embedding")
        )

        if old_embedding is None:
            continue

        score = float(np.dot(embedding, old_embedding))
        candidate_scores.append((stable_id, score, age_seconds))

    candidate_scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    if not candidate_scores:
        return None, -1.0, -1.0

    best_stable_id, best_score, best_age = candidate_scores[0]
    second_best_score = (
        candidate_scores[1][1]
        if len(candidate_scores) >= 2
        else -1.0
    )

    score_margin = (
        best_score - second_best_score
        if second_best_score >= 0
        else 1.0
    )

    if (
        best_score >= TRACK_RECOVERY_THRESHOLD
        and score_margin >= TRACK_RECOVERY_MARGIN
    ):
        print(
            f"[TRACK RECOVERY MATCH] Stable ID={best_stable_id} "
            f"Score={best_score:.3f} "
            f"Margin={score_margin:.3f} "
            f"Age={best_age:.1f}s"
        )
        return best_stable_id, best_score, score_margin

    print(
        f"[TRACK RECOVERY REJECTED] "
        f"Best ID={best_stable_id} "
        f"Score={best_score:.3f} "
        f"Margin={score_margin:.3f} "
        f"Age={best_age:.1f}s"
    )

    return None, best_score, score_margin


# =========================
# MULTI-PERSON STATE
# =========================

person_states = {}

def make_person_state_key(camera_code, stable_id):
    return f"{camera_code}:{stable_id}"

track_to_stable_id = {}
track_embedding_buffers = {}
track_reference_embeddings = {}
track_last_seen = {}
recent_stable_tracks = {}
last_profile_update = {}
track_identity_source = {}
track_last_face_check = {}
track_first_seen = {}
track_face_override_votes = {}
track_unknown_face_embeddings = {}

# Per-person temporal shirt-colour memory.
shirt_color_histories = {}
stable_shirt_colors = {}
shirt_color_visit_started = {}
shirt_color_last_seen = {}

# Remember the last good FASHN mask for each live track.
clothing_mask_cache = {}
clothing_mask_cache_time = {}

max_buffer_frames = CLIP_SECONDS * CLIP_FPS
frame_buffer = deque(maxlen=max_buffer_frames)

camera_frame_buffers = {}

def get_camera_frame_buffer(camera_code):
    if camera_code not in camera_frame_buffers:
        camera_frame_buffers[camera_code] = deque(
            maxlen=max_buffer_frames
        )

    return camera_frame_buffers[camera_code]


# =========================
# DASHBOARD FRAME PUBLISHER
# =========================

def publish_processed_frame(frame, camera_code=None):
    """
    Publish the newest annotated AI frame for dashboard.py

    A temporary file is written first and then atomically replaced,
    preventing the dashboard from reading a half-written JPEG.
    """
    if frame is None or frame.size == 0:
        return False

    destination_path = (
        get_camera_processed_frame_path(camera_code)
        if camera_code
        else PROCESSED_FRAME_PATH
    )

    try:
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                PROCESSED_FRAME_JPEG_QUALITY,
            ],
        )

        if not success:
            return False

        temporary_path = (
            destination_path
            + f".{os.getpid()}.tmp.jpg"
        )

        with open(temporary_path, "wb") as file:
            file.write(encoded.tobytes())

        # Windows can briefly lock the destination while the dashboard
        # reads it. A few short retries avoid interrupting the AI loop.
        for _ in range(3):
            try:
                os.replace(
                    temporary_path,
                    destination_path,
                )
                return True
            except PermissionError:
                time.sleep(0.01)

        try:
            os.remove(temporary_path)
        except OSError:
            pass

        return False

    except Exception as error:
        print(f"[PROCESSED FEED ERROR] {error}")
        return False

def publish_live_status(people_count, camera_code=None):
    """
    Publish the immediate number of people visible in the latest frame.
    """

    destination_path = (
        get_camera_live_status_path(camera_code)
        if camera_code
        else LIVE_STATUS_PATH
    )

    try:
        status_data = {
            "people_count": max(0, int(people_count)),
            "updated_at": datetime.now().isoformat(),
        }

        temporary_path = (
            destination_path
            + f".{os.getpid()}.tmp"
        )

        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(status_data, file)

        for _ in range(3):
            try:
                os.replace(
                    temporary_path,
                    destination_path,
                )
                return True
            except PermissionError:
                time.sleep(0.01)

        try:
            os.remove(temporary_path)
        except OSError:
            pass

        return False

    except Exception as error:
        print(f"[LIVE STATUS ERROR] {error}")
        return False

# =========================
# MAIN LOOP
# =========================

camera_configs = get_enabled_camera_configs()

if not camera_configs:
    raise RuntimeError("No enabled cameras found in cameras.db")

print(f"[MULTI CAMERA] Enabled cameras: {[camera['code'] for camera in camera_configs]}")

camera_index = 0

try:
    while True:

        active_camera = camera_configs[
            camera_index % len(camera_configs)
        ]

        CAMERA_ID = active_camera["code"]
        CAMERA_NAME = active_camera["name"]
        CAMERA_URL = active_camera["url"]

        model = get_camera_model(CAMERA_ID)

        active_frame_buffer = get_camera_frame_buffer(
            CAMERA_ID
        )

        camera_index += 1

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

            # Stable IDs already represented by a current ByteTrack box.
            # This prevents two simultaneous boxes from sharing one ID.
            reserved_stable_ids = set()

            for current_box in results[0].boxes:
                if current_box.id is None:
                    continue

                current_temp_id = int(current_box.id[0])

                current_track_key = make_track_key(
                    CAMERA_ID,
                    current_temp_id,
                )

                if current_track_key in track_to_stable_id:
                    reserved_stable_ids.add(
                        track_to_stable_id[current_track_key]
                    )

            visual_detections = []

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                confidence = float(box.conf[0])

                if class_name != "person":
                    continue

                temp_track_id = None

                if box.id is not None:
                    temp_track_id = int(box.id[0])
               
                track_key = (
                    make_track_key(CAMERA_ID, temp_track_id)
                    if temp_track_id is not None
                    else None
                )

                person_crop = crop_person_from_box(frame, box)

                (
                    raw_shirt_color,
                    raw_shirt_confidence,
                ) = detect_shirt_color(
                    person_crop,
                    cache_key=track_key,
                )

                shirt_color = raw_shirt_color

                # Every YOLO person box is visible immediately, even before
                # the permanent identity has been resolved.
                visual_detection = {
                    "stable_id": None,
                    "temp_track_id": temp_track_id,
                    "confidence": confidence,
                    "shirt_color": "Analyzing",
                    "box": box,
                    "crop": person_crop,
                    "display_name": None,
                    "identity_source": "IDENTIFYING",
                    "status": "IDENTIFYING",
                }
                visual_detections.append(visual_detection)

                if temp_track_id is None:
                    visual_detection["identity_source"] = "TRACKING"
                    continue

                track_last_seen[track_key] = now
                track_first_seen.setdefault(
                    track_key,
                    now,
                )

                stable_id = None
                face_result = None
                identity_source = track_identity_source.get(
                    track_key,
                    "IDENTIFYING",
                )

                # -----------------------------------------------------
                # Existing ByteTrack session
                # -----------------------------------------------------
                if track_key in track_to_stable_id:
                    stable_id = track_to_stable_id[track_key]

                    previous_face_check = track_last_face_check.get(
                        track_key
                    )

                    should_check_face = (
                        previous_face_check is None
                        or (
                            now - previous_face_check
                        ).total_seconds()
                        >= LOCKED_FACE_CHECK_INTERVAL_SECONDS
                    )

                    if should_check_face:
                        excluded_face_ids = (
                            current_seen_ids | reserved_stable_ids
                        ) - {stable_id}

                        face_result = recognize_face(
                            person_crop,
                            excluded_person_ids=excluded_face_ids,
                        )
                        track_last_face_check[track_key] = now

                        face_match_id = face_result.get("person_id")
                        face_score = float(
                            face_result.get("score", -1.0)
                        )

                        if face_match_id == stable_id:
                            identity_source = "FACE"
                            track_identity_source[track_key] = "FACE"
                            track_face_override_votes.pop(
                                track_key,
                                None,
                            )

                            update_verified_face_gallery(
                                person_id=stable_id,
                                embedding=face_result.get("embedding"),
                                verified_match_score=face_score,
                            )

                        elif (
                            face_match_id is not None
                            and face_match_id != stable_id
                        ):
                            current_source = track_identity_source.get(
                                track_key,
                                identity_source,
                            )

                            # A confirmed familiar face is allowed to correct
                            # BODY / RECENT BODY / BODY PROVISIONAL.
                            if (
                                current_source != "FACE"
                                and face_score >= FACE_OVERRIDE_MIN_SCORE
                                and face_match_id not in (
                                    current_seen_ids
                                    | reserved_stable_ids
                                )
                            ):
                                vote_data = track_face_override_votes.get(
                                    track_key,
                                    {
                                        "person_id": face_match_id,
                                        "count": 0,
                                    },
                                )

                                if vote_data["person_id"] != face_match_id:
                                    vote_data = {
                                        "person_id": face_match_id,
                                        "count": 0,
                                    }

                                vote_data["count"] += 1
                                track_face_override_votes[
                                    track_key
                                ] = vote_data

                                print(
                                    f"[FACE OVERRIDE CHECK] Temp ID="
                                    f"{temp_track_id} BODY ID={stable_id} "
                                    f"Face ID={face_match_id} "
                                    f"Score={face_score:.3f} "
                                    f"Votes={vote_data['count']}/"
                                    f"{FACE_OVERRIDE_CONFIRMATIONS}"
                                )

                                if (
                                    vote_data["count"]
                                    >= FACE_OVERRIDE_CONFIRMATIONS
                                ):
                                    old_stable_id = stable_id
                                    stable_id = face_match_id
                                    identity_source = "FACE"

                                    old_person_state_key = make_person_state_key(
                                        CAMERA_ID,
                                        old_stable_id,
                                    )

                                    new_person_state_key = make_person_state_key(
                                        CAMERA_ID,
                                        stable_id,
                                    )

                                    track_to_stable_id[
                                        track_key
                                    ] = stable_id
                                    track_identity_source[
                                        track_key
                                    ] = "FACE"
                                    track_face_override_votes.pop(
                                        track_key,
                                        None,
                                    )

                                    reserved_stable_ids.discard(
                                        old_stable_id
                                    )
                                    reserved_stable_ids.add(stable_id)

                                    # Move temporary runtime state if the old
                                    # provisional BODY ID had already started.
                                    if (
                                        old_person_state_key in person_states
                                        and new_person_state_key not in person_states
                                    ):
                                        person_states[new_person_state_key] = (
                                            person_states.pop(
                                                old_person_state_key
                                            )
                                        )
                                        person_states[
                                        new_person_state_key
                                        ]["stable_id"] = stable_id

                                    elif old_person_state_key in person_states:
                                        person_states.pop(
                                            old_person_state_key,
                                            None,
                                        )

                                    if (
                                        old_stable_id
                                        in stable_shirt_colors
                                        and stable_id
                                        not in stable_shirt_colors
                                    ):
                                        stable_shirt_colors[stable_id] = (
                                            stable_shirt_colors.pop(
                                                old_stable_id
                                            )
                                        )

                                    update_verified_face_gallery(
                                        person_id=stable_id,
                                        embedding=face_result.get(
                                            "embedding"
                                        ),
                                        verified_match_score=face_score,
                                    )

                                    print(
                                        f"[FACE ID OVERRIDE] Temp ID="
                                        f"{temp_track_id} corrected "
                                        f"Person ID {old_stable_id} -> "
                                        f"{stable_id}"
                                    )
                            else:
                                print(
                                    f"[FACE CONFLICT IGNORED] Temp ID="
                                    f"{temp_track_id} locked to Person ID "
                                    f"{stable_id}; face suggested "
                                    f"Person ID {face_match_id}."
                                )

                # -----------------------------------------------------
                # New ByteTrack session: FACE FIRST
                # -----------------------------------------------------
                else:
                    previous_face_check = track_last_face_check.get(
                        track_key
                    )

                    should_check_face = (
                        previous_face_check is None
                        or (
                            now - previous_face_check
                        ).total_seconds()
                        >= PENDING_FACE_CHECK_INTERVAL_SECONDS
                    )

                    if should_check_face:
                        excluded_face_ids = (
                            current_seen_ids | reserved_stable_ids
                        )

                        face_result = recognize_face(
                            person_crop,
                            excluded_person_ids=excluded_face_ids,
                        )
                        track_last_face_check[track_key] = now
                    else:
                        face_result = {
                            "person_id": None,
                            "score": -1.0,
                            "margin": -1.0,
                            "embedding": None,
                            "face_detected": False,
                        }

                    face_match_id = face_result.get("person_id")

                    if face_match_id is not None:
                        stable_id = face_match_id
                        identity_source = "FACE"

                        clear_pending_identity(track_key)
                        track_embedding_buffers.pop(
                            track_key,
                            None,
                        )
                        track_unknown_face_embeddings.pop(
                            track_key,
                            None,
                        )

                        body_reference = extract_embedding(person_crop)

                        if body_reference is not None:
                            track_reference_embeddings[
                                track_key
                            ] = body_reference

                        update_verified_face_gallery(
                            person_id=stable_id,
                            embedding=face_result.get("embedding"),
                            verified_match_score=face_result.get(
                                "score",
                                -1.0,
                            ),
                        )

                        print(
                            f"[FACE ID RESTORED] Temp ID={temp_track_id} "
                            f"matched permanent Person ID={stable_id} "
                            f"Score={face_result.get('score', -1.0):.3f}"
                        )

                    else:
                        unknown_face_embedding = face_result.get(
                            "embedding"
                        )

                        if unknown_face_embedding is not None:
                            unknown_buffer = (
                                track_unknown_face_embeddings.setdefault(
                                    track_key,
                                    [],
                                )
                            )
                            unknown_buffer.append(
                                unknown_face_embedding
                            )

                            if (
                                len(unknown_buffer)
                                > UNKNOWN_FACE_BUFFER_SIZE
                            ):
                                track_unknown_face_embeddings[
                                    track_key
                                ] = unknown_buffer[
                                    -UNKNOWN_FACE_BUFFER_SIZE:
                                ]

                        current_embedding = extract_embedding(
                            person_crop
                        )

                        if current_embedding is None:
                            visual_detection[
                                "identity_source"
                            ] = "IDENTIFYING"
                            continue

                        buffer = track_embedding_buffers.setdefault(
                            track_key,
                            [],
                        )
                        buffer.append(current_embedding)

                        if len(buffer) > MIN_REID_FRAMES:
                            track_embedding_buffers[
                                track_key
                            ] = buffer[-MIN_REID_FRAMES:]
                            buffer = track_embedding_buffers[
                                track_key
                            ]

                        collected_frames = len(buffer)

                        if collected_frames < MIN_REID_FRAMES:
                            visual_detection[
                                "identity_source"
                            ] = (
                                f"IDENTIFYING "
                                f"{collected_frames}/"
                                f"{MIN_REID_FRAMES}"
                            )
                            continue

                        average_embedding = np.mean(
                            buffer,
                            axis=0,
                        )
                        average_embedding = (
                            normalize_recovery_embedding(
                                average_embedding
                            )
                        )

                        if average_embedding is None:
                            track_embedding_buffers.pop(
                                track_key,
                                None,
                            )
                            continue

                        excluded_ids = (
                            current_seen_ids | reserved_stable_ids
                        )

                        recovered_id, _, _ = find_recent_track_match(
                            average_embedding,
                            now,
                            excluded_stable_ids=excluded_ids,
                        )

                        if recovered_id is not None:
                            stable_id = recovered_id
                            identity_source = "RECENT BODY"
                            clear_pending_identity(track_key)

                            print(
                                f"[TRACK RECOVERED] Temp ID="
                                f"{temp_track_id} reconnected to "
                                f"Stable ID={stable_id}"
                            )

                        else:
                            pending_seconds = (
                                now
                                - track_first_seen[track_key]
                            ).total_seconds()

                            if (
                                pending_seconds
                                < NEW_ID_FACE_GRACE_SECONDS
                            ):
                                visual_detection[
                                    "identity_source"
                                ] = (
                                    f"IDENTIFYING "
                                    f"{pending_seconds:.1f}s"
                                )

                                print(
                                    f"[IDENTITY GRACE] Temp ID="
                                    f"{temp_track_id} waiting for face "
                                    f"or recent-body recovery "
                                    f"({pending_seconds:.1f}/"
                                    f"{NEW_ID_FACE_GRACE_SECONDS:.1f}s)"
                                )
                                continue

                            # Only after the grace period may persistent
                            # body ReID match or create a permanent ID.
                            stable_id = get_stable_person_id(
                                person_crop=None,
                                shirt_color=shirt_color,
                                embedding=average_embedding,
                                track_key=track_key,
                            )
                            identity_source = "BODY PROVISIONAL"

                        if stable_id is None:
                            continue

                        track_reference_embeddings[
                            track_key
                        ] = average_embedding

                        # Register an unknown person's face only after
                        # several clear face embeddings agree over time.
                        unknown_buffer = (
                            track_unknown_face_embeddings.get(
                                track_key,
                                [],
                            )
                        )

                        if (
                            get_face_profile(stable_id) is None
                            and len(unknown_buffer)
                            >= UNKNOWN_FACE_REGISTRATION_MIN_FRAMES
                        ):
                            averaged_face_embedding = np.mean(
                                np.asarray(
                                    unknown_buffer,
                                    dtype=np.float32,
                                ),
                                axis=0,
                            )
                            face_norm = np.linalg.norm(
                                averaged_face_embedding
                            )

                            if face_norm > 0:
                                averaged_face_embedding = (
                                    averaged_face_embedding / face_norm
                                )

                                registered = register_face_embedding(
                                    person_id=stable_id,
                                    embedding=averaged_face_embedding,
                                )

                                if registered:
                                    print(
                                        f"[FACE MEMORY CREATED] "
                                        f"Person ID {stable_id} "
                                        f"registered from "
                                        f"{len(unknown_buffer)} "
                                        "face frames"
                                    )

                    if stable_id is None:
                        continue

                    track_to_stable_id[
                        track_key
                    ] = stable_id
                    track_identity_source[
                        track_key
                    ] = identity_source
                    track_embedding_buffers.pop(
                        track_key,
                        None,
                    )
                    reserved_stable_ids.add(stable_id)

                    print(
                        f"[IDENTITY LOCKED] Temp ID={temp_track_id} "
                        f"Permanent ID={stable_id} "
                        f"Source={identity_source}"
                    )

                if stable_id is None:
                    continue

                shirt_color = stabilize_shirt_color(
                    person_id=stable_id,
                    raw_colour=raw_shirt_color,
                    raw_confidence=raw_shirt_confidence,
                    now=now,
                )

                reference_embedding = (
                    track_reference_embeddings.get(
                        track_key
                    )
                )

                if reference_embedding is not None:
                    recent_stable_tracks[stable_id] = {
                        "embedding": reference_embedding,
                        "last_seen": now,
                        "temp_track_id": temp_track_id,
                    }

                previous_profile_update = last_profile_update.get(
                    stable_id
                )

                should_update_profile = (
                    previous_profile_update is None
                    or (
                        now - previous_profile_update
                    ).total_seconds()
                    >= PROFILE_UPDATE_INTERVAL_SECONDS
                )

                if should_update_profile:
                    update_person_profile(
                        stable_id,
                        shirt_color,
                    )

                    ensure_person_profile(
                        person_id=stable_id,
                        shirt_color=shirt_color,
                        camera_id=CAMERA_ID,
                        camera_name=CAMERA_NAME,
                    )

                    last_profile_update[stable_id] = now

                display_name = get_face_display_name(
                    stable_id
                )
                identity_source = track_identity_source.get(
                    track_key,
                    identity_source,
                )

                resolved_detection = {
                    "stable_id": stable_id,
                    "temp_track_id": temp_track_id,
                    "confidence": confidence,
                    "shirt_color": shirt_color,
                    "box": box,
                    "crop": person_crop,
                    "display_name": display_name,
                    "identity_source": identity_source,
                    "status": "IDENTIFIED",
                }

                detections.append(resolved_detection)
                visual_detection.update(resolved_detection)
                current_seen_ids.add(stable_id)

                print(
                    f"[IDENTITY TRACK] Camera={CAMERA_NAME} "
                    f"Permanent ID={stable_id} "
                    f"Temp ID={temp_track_id} "
                    f"Source={identity_source} "
                    f"Conf={confidence:.2f} "
                    f"Shirt={shirt_color}"
                )

            # Remove old ByteTrack mappings so a recycled temporary ID
            # cannot inherit another person's stable identity later.
            for old_temp_id, last_seen_time in list(track_last_seen.items()):
                inactive_seconds = (
                    now - last_seen_time
                ).total_seconds()

                if inactive_seconds > TRACK_MAPPING_EXPIRE_SECONDS:
                    track_last_seen.pop(old_temp_id, None)
                    track_to_stable_id.pop(old_temp_id, None)
                    track_embedding_buffers.pop(old_temp_id, None)
                    track_reference_embeddings.pop(old_temp_id, None)
                    track_identity_source.pop(old_temp_id, None)
                    track_last_face_check.pop(old_temp_id, None)
                    track_first_seen.pop(old_temp_id, None)
                    track_face_override_votes.pop(old_temp_id, None)
                    track_unknown_face_embeddings.pop(
                        old_temp_id,
                        None,
                    )
                    clear_pending_identity(old_temp_id)

                    print(
                        f"[TRACK CLEANUP] Removed Temp ID={old_temp_id} "
                        f"after {inactive_seconds:.1f}s"
                    )

            # Recent stable tracks are only needed for the short recovery window.
            for old_stable_id, recent_data in list(
                recent_stable_tracks.items()
            ):
                age_seconds = (
                    now - recent_data["last_seen"]
                ).total_seconds()

                if age_seconds > TRACK_RECOVERY_SECONDS:
                    recent_stable_tracks.pop(old_stable_id, None)

            annotated_frame = frame.copy()

            for det in visual_detections:
                box = det["box"]
                stable_id = det.get("stable_id")
                confidence = det["confidence"]
                status = det.get("status", "IDENTIFYING")

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                if stable_id is None:
                    box_colour = (0, 165, 255)
                    identity_source = det.get(
                        "identity_source",
                        "IDENTIFYING",
                    )
                    label = (
                        f"Identifying... | {identity_source} | "
                        f"{confidence:.2f}"
                    )
                else:
                    box_colour = (0, 255, 255)
                    shirt_color = det["shirt_color"]
                    display_name = det.get("display_name")
                    identity_source = det.get(
                        "identity_source",
                        "BODY",
                    )

                    identity_text = (
                        f"{display_name} | ID {stable_id}"
                        if display_name
                        else f"Person ID {stable_id}"
                    )

                    label = (
                        f"{identity_text} | {identity_source} | "
                        f"{shirt_color} | {confidence:.2f}"
                    )

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    box_colour,
                    2,
                )

                (
                    label_width,
                    label_height,
                ), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    2,
                )

                label_top = max(
                    0,
                    y1 - label_height - baseline - 12,
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, label_top),
                    (x1 + label_width + 12, y1),
                    box_colour,
                    -1,
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 6, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 0, 0),
                    2,
                )

            cv2.putText(
                annotated_frame,
                f"{CAMERA_NAME} | People Count: {len(visual_detections)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            y_text = 80
            for det in detections:
                display_name = det.get("display_name")
                identity_name = (
                    display_name
                    if display_name
                    else f"Person {det['stable_id']}"
                )

                cv2.putText(
                    annotated_frame,
                    f"{identity_name} | {det.get('identity_source', 'BODY')} "
                    f"| Shirt: {det['shirt_color']}",
                    (20, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                y_text += 35

            # Make the exact annotated AI frame available to FastAPI.
            publish_processed_frame(annotated_frame)
            publish_processed_frame(
                annotated_frame,
                camera_code=CAMERA_ID,
            )

            publish_live_status(
                len(visual_detections)
            )
            publish_live_status(
                len(visual_detections),
                camera_code=CAMERA_ID,
            )

            active_frame_buffer.append(
                annotated_frame.copy()
            )

            for det in detections:
                stable_id = det["stable_id"]
                confidence = det["confidence"]
                shirt_color = det["shirt_color"]
      
                person_state_key = make_person_state_key(
                    CAMERA_ID,
                    stable_id,
                )

                if person_state_key not in person_states:
                    person_states[person_state_key] = {
                        "camera_id": CAMERA_ID,
                        "stable_id": stable_id,
                        "present": False,
                        "enter_time": None,
                        "last_seen": now,
                        "present_logged": False,
                        "loitering_logged": False,
                        "shirt_color": shirt_color,
                        "confidence": confidence
                    }

                state = person_states[person_state_key]
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
                            active_frame_buffer,
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

            for person_state_key, state in list(person_states.items()):
                if state["camera_id"] != CAMERA_ID:
                    continue

                state_stable_id = state["stable_id"]
                if state["present"] and state_stable_id not in current_seen_ids:

                    missing_seconds = (now - state["last_seen"]).total_seconds()

                    if missing_seconds >= MISSING_GRACE_SECONDS:
                        save_event(
                            "PERSON_LEFT",
                            1.0,
                            person_id=state_stable_id,
                            shirt_color=state["shirt_color"]
                        )

                        state["present"] = False
                        state["enter_time"] = None
                        state["present_logged"] = False
                        state["loitering_logged"] = False

                        # A later visit may use different clothing.
                        reset_shirt_colour_visit(state_stable_id)

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