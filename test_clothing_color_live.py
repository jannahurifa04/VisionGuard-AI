import time

import cv2
import numpy as np
import requests
import torch

from PIL import Image

from ultralytics import YOLO
from fashn_human_parser import FashnHumanParser

from torchvision import transforms
from torchvision.models import mobilenet_v3_small


# =========================
# SETTINGS
# =========================

CAMERA_URL = "http://192.168.18.134:8080/shot.jpg"

YOLO_MODEL_PATH = "yolo11n.pt"

COLOR_MODEL_PATH = "clothing_color_classifier.pth"

PERSON_CONFIDENCE = 0.40

TOP_CLASS_ID = 3

ROTATE_FRAME = True

UNKNOWN_THRESHOLD = 0.70


# =========================
# DEVICE
# =========================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"[TEST] Device: {DEVICE}")


# =========================
# LOAD YOLO
# =========================

print("[TEST] Loading YOLO...")

yolo_model = YOLO(
    YOLO_MODEL_PATH
)


# =========================
# LOAD FASHN
# =========================

print("[TEST] Loading FASHN...")

clothing_parser = FashnHumanParser()


# =========================
# LOAD COLOR MODEL
# =========================

print("[TEST] Loading color classifier...")

checkpoint = torch.load(
    COLOR_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

CLASSES = checkpoint["classes"]

IMAGE_SIZE = checkpoint.get(
    "image_size",
    224,
)

color_model = mobilenet_v3_small(
    weights=None
)

input_features = (
    color_model.classifier[3].in_features
)

color_model.classifier[3] = torch.nn.Linear(
    input_features,
    len(CLASSES),
)

color_model.load_state_dict(
    checkpoint["state_dict"]
)

color_model = color_model.to(
    DEVICE
)

color_model.eval()

print(
    f"[TEST] Classes: {CLASSES}"
)


# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
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


# =========================
# CAMERA
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

        print(
            f"[CAMERA ERROR] {error}"
        )

        return None


# =========================
# PERSON DETECTION
# =========================

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

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return crop


# =========================
# FASHN CLOTHING MASK
# =========================

def isolate_upper_clothing(person_crop):

    rgb = cv2.cvtColor(
        person_crop,
        cv2.COLOR_BGR2RGB,
    )

    segmentation = (
        clothing_parser.predict(rgb)
    )

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

    points = cv2.findNonZero(
        mask
    )

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(
        points
    )

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

    # Same neutral background used
    # during dataset capture.
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


# =========================
# COLOR PREDICTION
# =========================

def predict_color(clothing_image):

    rgb = cv2.cvtColor(
        clothing_image,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        rgb
    )

    tensor = transform(
        pil_image
    ).unsqueeze(0)

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        output = color_model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, index = torch.max(
            probabilities,
            dim=1,
        )

    confidence = float(
        confidence.item()
    )

    prediction = CLASSES[
        int(index.item())
    ]

    if confidence < UNKNOWN_THRESHOLD:
        return (
            "Unknown",
            confidence,
        )

    return (
        prediction,
        confidence,
    )


# =========================
# LIVE TEST
# =========================

print()
print(
    "[TEST] Live clothing color test started."
)

print(
    "[TEST] Press Q in the camera window to stop."
)

print()


while True:

    frame = get_camera_frame()

    if frame is None:
        time.sleep(1)
        continue


    person_crop = detect_person(
        frame
    )

    if person_crop is None:

        print(
            "[TEST] No person detected"
        )

        time.sleep(1)
        continue


    try:

        clothing = isolate_upper_clothing(
            person_crop
        )

    except Exception as error:

        print(
            f"[PARSER ERROR] {error}"
        )

        time.sleep(1)
        continue


    if clothing is None:

        print(
            "[TEST] Clothing not clear"
        )

        time.sleep(1)
        continue


    color, confidence = predict_color(
        clothing
    )


    print(
        f"[COLOR] {color} | "
        f"Confidence={confidence:.2f}"
    )


    preview = clothing.copy()

    cv2.putText(
        preview,
        f"{color} | {confidence:.2f}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )


    cv2.imshow(
        "VisionGuard Clothing Color Test",
        preview,
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


    time.sleep(0.7)


cv2.destroyAllWindows()

print(
    "[TEST] Finished."
)