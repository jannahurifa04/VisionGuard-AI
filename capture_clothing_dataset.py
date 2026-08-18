import os
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO
from fashn_human_parser import FashnHumanParser


# =========================
# SETTINGS
# =========================

CAMERA_URL = "http://192.168.18.134:8080/shot.jpg"

MODEL_PATH = "yolo11n.pt"

OUTPUT_ROOT = Path("clothing_color_raw")

PERSON_CONFIDENCE = 0.40

TOP_CLASS_ID = 3

CAPTURE_INTERVAL_SECONDS = 1.0

DEFAULT_CAPTURE_COUNT = 30

ROTATE_FRAME = True


# =========================
# MODELS
# =========================

print("[DATASET] Loading YOLO...")
yolo_model = YOLO(MODEL_PATH)

print("[DATASET] Loading FASHN...")
clothing_parser = FashnHumanParser()

print("[DATASET] Models ready.")


# =========================
# HELPERS
# =========================

def get_camera_frame():
    try:
        response = requests.get(
            CAMERA_URL,
            timeout=8,
        )

        response.raise_for_status()

        image_array = np.frombuffer(
            response.content,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            return None

        if ROTATE_FRAME:
            frame = cv2.rotate(
                frame,
                cv2.ROTATE_90_CLOCKWISE,
            )

        return frame

    except Exception as error:
        print(f"[CAMERA ERROR] {error}")
        return None


def detect_person(frame):
    result = yolo_model(
        frame,
        classes=[0],
        conf=PERSON_CONFIDENCE,
        verbose=False,
    )[0]

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return None

    best_index = int(
        result.boxes.conf.argmax().item()
    )

    x1, y1, x2, y2 = (
        result.boxes.xyxy[best_index]
        .cpu()
        .numpy()
        .astype(int)
    )

    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    person_crop = frame[
        y1:y2,
        x1:x2,
    ]

    if person_crop.size == 0:
        return None

    return person_crop


def isolate_upper_clothing(person_crop):
    rgb = cv2.cvtColor(
        person_crop,
        cv2.COLOR_BGR2RGB,
    )

    segmentation = clothing_parser.predict(rgb)

    mask = np.where(
        segmentation == TOP_CLASS_ID,
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    points = cv2.findNonZero(mask)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    if w < 30 or h < 40:
        return None

    clothing = person_crop[
        y:y + h,
        x:x + w,
    ]

    clothing_mask = mask[
        y:y + h,
        x:x + w,
    ]

    # Use the same neutral background for every colour.
    # This prevents the room/background from teaching
    # the classifier the wrong thing.
    output = np.full_like(
        clothing,
        127,
    )

    output[
        clothing_mask > 0
    ] = clothing[
        clothing_mask > 0
    ]

    return output


def capture_dataset(
    colour_name,
    capture_count,
):
    output_folder = (
        OUTPUT_ROOT
        / colour_name
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0
    attempts = 0

    print()
    print(
        f"[DATASET] Capturing {colour_name}"
    )
    print(
        f"[DATASET] Target: {capture_count} images"
    )
    print(
        "[DATASET] Move slightly between captures."
    )
    print()

    while saved < capture_count:

        attempts += 1

        frame = get_camera_frame()

        if frame is None:
            time.sleep(1)
            continue

        person_crop = detect_person(frame)

        if person_crop is None:
            print(
                "[SKIP] No person detected"
            )
            time.sleep(
                CAPTURE_INTERVAL_SECONDS
            )
            continue

        try:
            clothing = isolate_upper_clothing(
                person_crop
            )

        except Exception as error:
            print(
                f"[SKIP] Parser error: {error}"
            )
            time.sleep(
                CAPTURE_INTERVAL_SECONDS
            )
            continue

        if clothing is None:
            print(
                "[SKIP] Upper clothing not clear"
            )
            time.sleep(
                CAPTURE_INTERVAL_SECONDS
            )
            continue

        timestamp = int(
            time.time() * 1000
        )

        filename = (
            output_folder
            / (
                f"{colour_name.lower()}_"
                f"{timestamp}.jpg"
            )
        )

        success = cv2.imwrite(
            str(filename),
            clothing,
        )

        if success:
            saved += 1

            print(
                f"[SAVED] {saved}/{capture_count} "
                f"{filename.name}"
            )

        time.sleep(
            CAPTURE_INTERVAL_SECONDS
        )

    print()
    print(
        f"[DONE] {saved} {colour_name} "
        f"clothing images saved."
    )


# =========================
# MAIN
# =========================

VALID_COLOURS = [
    "Black",
    "White",
    "Gray",
    "Beige",
    "Brown",
    "Red",
    "Pink",
    "Orange",
    "Yellow",
    "Green",
    "Teal",
    "Blue",
    "Purple",
]


print()
print("Available colours:")
print(", ".join(VALID_COLOURS))
print()

colour = input(
    "Colour you are wearing: "
).strip().title()

if colour not in VALID_COLOURS:
    print(
        f"Invalid colour: {colour}"
    )
    raise SystemExit(1)


count_text = input(
    f"Number of images "
    f"[Enter = {DEFAULT_CAPTURE_COUNT}]: "
).strip()

if count_text:
    capture_count = int(count_text)
else:
    capture_count = (
        DEFAULT_CAPTURE_COUNT
    )


capture_dataset(
    colour,
    capture_count,
)