import torch
import torchreid
import cv2
import numpy as np
from datetime import datetime

# =========================
# OSNet ReID Model
# =========================

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)

model.eval()

# =========================
# Stable Person Memory
# =========================

known_persons = {}
next_stable_person_id = 1

REID_MATCH_THRESHOLD = 0.55


def extract_embedding(person_crop):
    if person_crop is None:
        return None

    try:
        image = cv2.resize(person_crop, (128, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        tensor = torch.tensor(image).float().unsqueeze(0)

        with torch.no_grad():
            features = model(tensor)

        embedding = features.numpy().flatten()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    except Exception as e:
        print("[REID ERROR]", e)
        return None


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0

    return float(np.dot(a, b))


def get_stable_person_id(person_crop, shirt_color="Unknown"):
    global next_stable_person_id, known_persons

    embedding = extract_embedding(person_crop)

    if embedding is None:
        person_id = next_stable_person_id
        next_stable_person_id += 1

        known_persons[person_id] = {
            "embedding": None,
            "shirt_color": shirt_color,
            "last_seen": datetime.now()
        }

        return person_id

    best_person_id = None
    best_score = 0.0

    for person_id, data in known_persons.items():
        old_embedding = data.get("embedding")

        if old_embedding is None:
            continue

        score = cosine_similarity(embedding, old_embedding)

        if score > best_score:
            best_score = score
            best_person_id = person_id

    if best_person_id is not None and best_score >= REID_MATCH_THRESHOLD:
        old_embedding = known_persons[best_person_id]["embedding"]
        known_persons[best_person_id]["embedding"] = (old_embedding * 0.7) + (embedding * 0.3)
        known_persons[best_person_id]["shirt_color"] = shirt_color
        known_persons[best_person_id]["last_seen"] = datetime.now()

        print(f"[REID MATCH] Stable Person ID {best_person_id} Score={best_score:.2f}")
        return best_person_id

    person_id = next_stable_person_id
    next_stable_person_id += 1

    known_persons[person_id] = {
        "embedding": embedding,
        "shirt_color": shirt_color,
        "last_seen": datetime.now()
    }

    print(f"[REID NEW] Stable Person ID {person_id}")

    return person_id