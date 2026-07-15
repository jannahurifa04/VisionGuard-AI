import os
import pickle
from datetime import datetime

import cv2
import numpy as np
import torch
import torchreid


# =========================================================
# Configuration
# =========================================================

REID_MEMORY_FILE = "reid_memory.pkl"

# Start slightly stricter to reduce incorrect matches.
REID_MATCH_THRESHOLD = 0.70

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# OSNet ReID model
# =========================================================

model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True,
)

model = model.to(DEVICE)
model.eval()

print(f"[REID] Model running on: {DEVICE}")


# =========================================================
# Persistent person memory
# =========================================================

known_persons = {}
next_stable_person_id = 1


def load_reid_memory():
    global known_persons, next_stable_person_id

    if not os.path.exists(REID_MEMORY_FILE):
        known_persons = {}
        next_stable_person_id = 1
        print("[REID] No previous memory found.")
        return

    try:
        with open(REID_MEMORY_FILE, "rb") as file:
            saved_data = pickle.load(file)

        known_persons = saved_data.get("known_persons", {})
        next_stable_person_id = saved_data.get(
            "next_stable_person_id",
            1,
        )

        print(
            f"[REID] Loaded {len(known_persons)} known person profile(s)."
        )

    except Exception as error:
        print(f"[REID LOAD ERROR] {error}")
        known_persons = {}
        next_stable_person_id = 1


def save_reid_memory():
    try:
        saved_data = {
            "known_persons": known_persons,
            "next_stable_person_id": next_stable_person_id,
        }

        with open(REID_MEMORY_FILE, "wb") as file:
            pickle.dump(saved_data, file)

    except Exception as error:
        print(f"[REID SAVE ERROR] {error}")


# Load saved identities when this file starts
load_reid_memory()


# =========================================================
# Embedding extraction
# =========================================================

def extract_embedding(person_crop):
    if person_crop is None or person_crop.size == 0:
        return None

    try:
        # OSNet normally uses 256 height × 128 width
        image = cv2.resize(person_crop, (128, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))

        tensor = torch.from_numpy(image).float().unsqueeze(0)
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            features = model(tensor)

        embedding = features.detach().cpu().numpy().flatten()

        norm = np.linalg.norm(embedding)

        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    except Exception as error:
        print(f"[REID EMBEDDING ERROR] {error}")
        return None


def cosine_similarity(embedding_a, embedding_b):
    if embedding_a is None or embedding_b is None:
        return 0.0

    return float(np.dot(embedding_a, embedding_b))


# =========================================================
# Stable identity matching
# =========================================================

def get_stable_person_id(person_crop, shirt_color="Unknown"):
    global known_persons, next_stable_person_id

    embedding = extract_embedding(person_crop)

    if embedding is None:
        print("[REID] No valid embedding. Identity not created.")
        return None

    best_person_id = None
    best_score = -1.0

    for person_id, person_data in known_persons.items():
        old_embedding = person_data.get("embedding")

        if old_embedding is None:
            continue

        score = cosine_similarity(
            embedding,
            old_embedding,
        )

        if score > best_score:
            best_score = score
            best_person_id = person_id

    # Existing person matched
    if (
        best_person_id is not None
        and best_score >= REID_MATCH_THRESHOLD
    ):
        old_embedding = known_persons[best_person_id]["embedding"]

        updated_embedding = (
            old_embedding * 0.8
            + embedding * 0.2
        )

        updated_norm = np.linalg.norm(updated_embedding)

        if updated_norm > 0:
            updated_embedding = updated_embedding / updated_norm

        known_persons[best_person_id]["embedding"] = updated_embedding
        known_persons[best_person_id]["shirt_color"] = shirt_color
        known_persons[best_person_id]["last_seen"] = datetime.now().isoformat()

        save_reid_memory()

        print(
            f"[REID MATCH] Person ID {best_person_id} "
            f"Similarity={best_score:.3f}"
        )

        return best_person_id

    # New person
    person_id = next_stable_person_id
    next_stable_person_id += 1

    known_persons[person_id] = {
        "embedding": embedding,
        "shirt_color": shirt_color,
        "first_seen": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat(),
    }

    save_reid_memory()

    print(
        f"[REID NEW] Person ID {person_id} "
        f"Best similarity={best_score:.3f}"
    )

    return person_id