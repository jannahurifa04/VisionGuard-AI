import cv2
from ultralytics import YOLOWorld

MODEL_FILE = "yolov8s-worldv2.pt"
CAMERA_SOURCE = 0

OBJECTS_TO_DETECT = [
    "person",
    "backpack",
    "cell phone",
    "bottle",
    "cardboard box",
]

model = YOLOWorld(MODEL_FILE)
model.set_classes(OBJECTS_TO_DETECT)

camera = cv2.VideoCapture(CAMERA_SOURCE)

if not camera.isOpened():
    raise RuntimeError("Could not open the camera.")

print("[YOLO-WORLD] Running")
print("[YOLO-WORLD] Press Q to stop")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read the camera frame.")
        break

    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False,
    )

    display_frame = results[0].plot()

    cv2.imshow(
        "VisionGuard - YOLO World Test",
        display_frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()