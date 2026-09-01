import os
import time
import sqlite3
from datetime import datetime

import cv2
from ultralytics import YOLOWorld


MODEL_FILE = "yolov8s-worldv2.pt"
CAMERA_SOURCE = 0
CONFIDENCE_THRESHOLD = 0.35
ALERT_COOLDOWN_SECONDS = 60

SNAPSHOT_FOLDER = "object_alerts"
DATABASE_FILE = "object_alerts.db"

OBJECTS_TO_DETECT = [
    "person",
    "backpack",
    "cell phone",
    "bottle",
    "cardboard box",
]

ALERT_OBJECTS = {
    "backpack",
    "cell phone",
    "cardboard box",
}


# -----------------------------
# Create snapshot folder
# -----------------------------

os.makedirs(
    SNAPSHOT_FOLDER,
    exist_ok=True,
)


# -----------------------------
# SQLite database
# -----------------------------

db = sqlite3.connect(
    DATABASE_FILE,
)

cursor = db.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS object_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        object_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        snapshot_path TEXT NOT NULL
    )
    """
)

db.commit()

print(
    f"[DATABASE] Ready: {DATABASE_FILE}"
)


# -----------------------------
# YOLO-World
# -----------------------------

model = YOLOWorld(
    MODEL_FILE,
)

model.set_classes(
    OBJECTS_TO_DETECT,
)


# -----------------------------
# Camera
# -----------------------------

camera = cv2.VideoCapture(
    CAMERA_SOURCE,
)

if not camera.isOpened():
    db.close()

    raise RuntimeError(
        "Could not open the camera."
    )


last_alert_time = {}

print("[YOLO-WORLD ALERTS] Running")
print("[YOLO-WORLD ALERTS] Press Q to stop")


# -----------------------------
# Main loop
# -----------------------------

while True:

    success, frame = camera.read()

    if not success:
        print(
            "Could not read the camera frame."
        )
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    result = results[0]

    display_frame = result.plot()

    current_time = time.time()

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            object_name = result.names[
                class_id
            ]

            if object_name not in ALERT_OBJECTS:
                continue

            previous_alert = last_alert_time.get(
                object_name,
                0,
            )

            if (
                current_time - previous_alert
                < ALERT_COOLDOWN_SECONDS
            ):
                continue


            # -----------------------------
            # Save snapshot
            # -----------------------------

            timestamp = datetime.now()

            timestamp_file = timestamp.strftime(
                "%Y%m%d_%H%M%S"
            )

            timestamp_db = timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            safe_name = object_name.replace(
                " ",
                "_",
            )

            snapshot_path = os.path.join(
                SNAPSHOT_FOLDER,
                f"{safe_name}_{timestamp_file}.jpg",
            )

            cv2.imwrite(
                snapshot_path,
                display_frame,
            )


            # -----------------------------
            # Save alert to SQLite
            # -----------------------------

            cursor.execute(
                """
                INSERT INTO object_alerts (
                    timestamp,
                    object_name,
                    confidence,
                    snapshot_path
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    timestamp_db,
                    object_name,
                    confidence,
                    snapshot_path,
                ),
            )

            db.commit()


            # -----------------------------
            # Console alert
            # -----------------------------

            print(
                f"[OBJECT ALERT] "
                f"{object_name} detected | "
                f"Confidence={confidence:.2f} | "
                f"Snapshot={snapshot_path}"
            )

            print(
                "[DATABASE] Alert saved"
            )

            last_alert_time[
                object_name
            ] = current_time


    cv2.imshow(
        "VisionGuard - YOLO World Alerts",
        display_frame,
    )

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):
        break


# -----------------------------
# Cleanup
# -----------------------------

camera.release()

cv2.destroyAllWindows()

db.close()

print(
    "[YOLO-WORLD ALERTS] Stopped"
)