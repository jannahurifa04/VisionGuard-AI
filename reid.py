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

# Immediate confident match.
REID_MATCH_THRESHOLD = 0.82
REID_MATCH_MARGIN = 0.06

# Likely matches wait for more frames instead of creating a new ID.
REID_PENDING_THRESHOLD = 0.73
REID_PENDING_MARGIN = 0.05
PENDING_CONFIRMATIONS_REQUIRED = 3
PENDING_MAX_ATTEMPTS = 6

# Only very strong matches may improve the permanent profile.
REID_GALLERY_UPDATE_THRESHOLD = 0.85
REID_NEW_CLUSTER_THRESHOLD = 0.55

MAX_EMBEDDINGS_PER_PERSON = 8
MIN_CROP_HEIGHT = 120
MIN_CROP_WIDTH = 45

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
# Persistent and temporary memory
# =========================================================

known_persons = {}
next_stable_person_id = 1

# Temporary decisions only; not saved to disk.
pending_tracks = {}


def normalize_embedding(embedding):
    if embedding is None:
        return None

    embedding = np.asarray(embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(embedding)

    if norm <= 0:
        return None

    return embedding / norm


def migrate_person_profile(person_data):
    embeddings = person_data.get("embeddings")

    if embeddings is None:
        old_embedding = person_data.get("embedding")
        embeddings = [old_embedding] if old_embedding is not None else []

    valid_embeddings = []

    for embedding in embeddings:
        normalized = normalize_embedding(embedding)
        if normalized is not None:
            valid_embeddings.append(normalized)

    person_data["embeddings"] = valid_embeddings[-MAX_EMBEDDINGS_PER_PERSON:]
    person_data.pop("embedding", None)

    return person_data


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

        loaded_persons = saved_data.get("known_persons", {})
        known_persons = {}

        for person_id, person_data in loaded_persons.items():
            known_persons[int(person_id)] = migrate_person_profile(person_data)

        next_stable_person_id = saved_data.get("next_stable_person_id", 1)

        if known_persons:
            next_stable_person_id = max(
                next_stable_person_id,
                max(known_persons.keys()) + 1,
            )

        print(f"[REID] Loaded {len(known_persons)} known person profile(s).")

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

        temp_file = REID_MEMORY_FILE + ".tmp"

        with open(temp_file, "wb") as file:
            pickle.dump(saved_data, file)

        os.replace(temp_file, REID_MEMORY_FILE)

    except Exception as error:
        print(f"[REID SAVE ERROR] {error}")


load_reid_memory()


# =========================================================
# Crop quality and embedding extraction
# =========================================================


def is_good_person_crop(person_crop):
    if person_crop is None or person_crop.size == 0:
        return False

    height, width = person_crop.shape[:2]

    if height < MIN_CROP_HEIGHT:
        print(f"[REID QUALITY] Crop too short: {height}px")
        return False

    if width < MIN_CROP_WIDTH:
        print(f"[REID QUALITY] Crop too narrow: {width}px")
        return False

    aspect_ratio = height / max(width, 1)

    if aspect_ratio < 1.1 or aspect_ratio > 5.5:
        print(f"[REID QUALITY] Invalid crop ratio: {aspect_ratio:.2f}")
        return False

    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur_score < 20:
        print(f"[REID QUALITY] Crop is blurred: {blur_score:.1f}")
        return False

    return True


def extract_embedding(person_crop):
    if not is_good_person_crop(person_crop):
        return None

    try:
        image = cv2.resize(person_crop, (128, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))

        tensor = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = model(tensor)

        embedding = features.detach().cpu().numpy().flatten()
        return normalize_embedding(embedding)

    except Exception as error:
        print(f"[REID EMBEDDING ERROR] {error}")
        return None


def cosine_similarity(embedding_a, embedding_b):
    embedding_a = normalize_embedding(embedding_a)
    embedding_b = normalize_embedding(embedding_b)

    if embedding_a is None or embedding_b is None:
        return 0.0

    return float(np.dot(embedding_a, embedding_b))


# =========================================================
# Gallery helpers
# =========================================================


def calculate_person_score(new_embedding, stored_embeddings):
    if not stored_embeddings:
        return 0.0

    scores = [
        cosine_similarity(new_embedding, stored_embedding)
        for stored_embedding in stored_embeddings
    ]

    return max(scores) if scores else 0.0


def add_embedding_to_gallery(person_id, embedding):
    """
    Keep up to MAX_EMBEDDINGS_PER_PERSON distinct appearance clusters
    instead of a FIFO queue, so a person's different outfits can coexist
    instead of the newest captures silently evicting older valid ones.
    """
    person_data = known_persons.get(person_id)

    if person_data is None:
        return False

    normalized = normalize_embedding(embedding)

    if normalized is None:
        return False

    embeddings = person_data.setdefault("embeddings", [])

    if not embeddings:
        embeddings.append(normalized)
        return True

    sims = [cosine_similarity(normalized, e) for e in embeddings]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    if best_sim >= 0.995:
        return False

    if best_sim >= REID_GALLERY_UPDATE_THRESHOLD:
        updated = 0.7 * embeddings[best_idx] + 0.3 * normalized
        embeddings[best_idx] = normalize_embedding(updated)
        return True

    if best_sim < REID_NEW_CLUSTER_THRESHOLD or len(embeddings) < MAX_EMBEDDINGS_PER_PERSON:
        if len(embeddings) >= MAX_EMBEDDINGS_PER_PERSON:
            redundancy = []
            for i, e in enumerate(embeddings):
                other_sims = [cosine_similarity(e, o) for j, o in enumerate(embeddings) if j != i]
                redundancy.append(max(other_sims) if other_sims else 0.0)
            evict_idx = int(np.argmax(redundancy))
            embeddings[evict_idx] = normalized
        else:
            embeddings.append(normalized)
        return True

    return False

def clear_pending_identity(track_key):
    if track_key is None:
        return

    pending_tracks.pop(str(track_key), None)


def _accept_existing_person(
    person_id,
    embedding,
    shirt_color,
    best_score,
    second_best_score,
    score_margin,
    pending_match=False,
):
    person_data = known_persons[person_id]
    person_data["last_seen"] = datetime.now().isoformat()

    if shirt_color not in (None, "", "Unknown"):
        person_data["shirt_color"] = shirt_color

    gallery_updated = False

    if best_score >= REID_GALLERY_UPDATE_THRESHOLD:
        gallery_updated = add_embedding_to_gallery(person_id, embedding)

    save_reid_memory()

    label = "REID PENDING MATCH" if pending_match else "REID MATCH"

    print(
        f"[{label}] Person ID {person_id} "
        f"Best={best_score:.3f} "
        f"Second={second_best_score:.3f} "
        f"Margin={score_margin:.3f} "
        f"Gallery updated={gallery_updated}"
    )

    return person_id


def _create_new_person(
    embedding,
    shirt_color,
    best_person_id,
    best_score,
    second_best_score,
    score_margin,
):
    global next_stable_person_id

    person_id = next_stable_person_id
    next_stable_person_id += 1

    known_persons[person_id] = {
        "embeddings": [embedding],
        "shirt_color": shirt_color,
        "first_seen": datetime.now().isoformat(),
        "last_seen": datetime.now().isoformat(),
    }

    save_reid_memory()

    print(
        f"[REID NEW] Person ID {person_id} "
        f"Best person={best_person_id} "
        f"Best={best_score:.3f} "
        f"Second={second_best_score:.3f} "
        f"Margin={score_margin:.3f}"
    )

    return person_id


# =========================================================
# Stable identity matching with pending state
# =========================================================


def get_stable_person_id(
    person_crop=None,
    shirt_color="Unknown",
    embedding=None,
    track_key=None,
):
    """
    Return a Person ID only after a confident decision.
    Return None while more frames are still needed.
    """
    if embedding is None:
        embedding = extract_embedding(person_crop)
    else:
        embedding = normalize_embedding(embedding)

    if embedding is None:
        print("[REID] No valid embedding. Identity decision postponed.")
        return None

    candidate_scores = []

    for person_id, person_data in known_persons.items():
        stored_embeddings = person_data.get("embeddings", [])
        score = calculate_person_score(embedding, stored_embeddings)
        candidate_scores.append((person_id, score))

    candidate_scores.sort(key=lambda item: item[1], reverse=True)

    best_person_id = None
    best_score = -1.0
    second_best_score = -1.0

    if candidate_scores:
        best_person_id = candidate_scores[0][0]
        best_score = candidate_scores[0][1]

    if len(candidate_scores) >= 2:
        second_best_score = candidate_scores[1][1]

    score_margin = (
        best_score - second_best_score
        if second_best_score >= 0
        else 1.0
    )

    match_is_confident = (
        best_person_id is not None
        and best_score >= REID_MATCH_THRESHOLD
        and score_margin >= REID_MATCH_MARGIN
    )

    if match_is_confident:
        clear_pending_identity(track_key)

        return _accept_existing_person(
            best_person_id,
            embedding,
            shirt_color,
            best_score,
            second_best_score,
            score_margin,
            pending_match=False,
        )

    pending_key = str(track_key) if track_key is not None else "default"

    state = pending_tracks.setdefault(
        pending_key,
        {
            "attempts": 0,
            "candidate_id": None,
            "confirmations": 0,
            "scores": [],
        },
    )

    state["attempts"] += 1

    likely_existing_person = (
        best_person_id is not None
        and best_score >= REID_PENDING_THRESHOLD
        and score_margin >= REID_PENDING_MARGIN
    )

    if likely_existing_person:
        if state["candidate_id"] == best_person_id:
            state["confirmations"] += 1
        else:
            state["candidate_id"] = best_person_id
            state["confirmations"] = 1
            state["scores"] = []

        state["scores"].append(best_score)
        state["scores"] = state["scores"][-PENDING_CONFIRMATIONS_REQUIRED:]

        average_pending_score = float(np.mean(state["scores"]))

        print(
            f"[REID PENDING] Temp ID={track_key} "
            f"Possible Person ID={best_person_id} "
            f"Best={best_score:.3f} "
            f"Margin={score_margin:.3f} "
            f"Confirm={state['confirmations']}/"
            f"{PENDING_CONFIRMATIONS_REQUIRED} "
            f"Attempt={state['attempts']}/{PENDING_MAX_ATTEMPTS}"
        )

        if (
            state["confirmations"] >= PENDING_CONFIRMATIONS_REQUIRED
            and average_pending_score >= REID_PENDING_THRESHOLD
        ):
            clear_pending_identity(track_key)

            return _accept_existing_person(
                best_person_id,
                embedding,
                shirt_color,
                best_score,
                second_best_score,
                score_margin,
                pending_match=True,
            )

    else:
        state["candidate_id"] = None
        state["confirmations"] = 0
        state["scores"] = []

        print(
            f"[REID PENDING] Temp ID={track_key} "
            f"No reliable match yet "
            f"Best person={best_person_id} "
            f"Best={best_score:.3f} "
            f"Margin={score_margin:.3f} "
            f"Attempt={state['attempts']}/{PENDING_MAX_ATTEMPTS}"
        )

    if state["attempts"] < PENDING_MAX_ATTEMPTS:
        return None

    clear_pending_identity(track_key)

    return _create_new_person(
        embedding,
        shirt_color,
        best_person_id,
        best_score,
        second_best_score,
        score_margin,
    )
