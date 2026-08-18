"""
VisionGuard face recognition and persistent face memory.

This module uses InsightFace for:
- face detection
- face embedding extraction
- familiar-person matching
- persistent face galleries

Important:
The Person IDs stored here must be the SAME permanent IDs used by reid.py
and the persons table. Face memory does not create a second ID system.
"""

import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# =========================================================
# Configuration
# =========================================================

FACE_MEMORY_FILE = "face_memory.pkl"
FACE_MODEL_NAME = "buffalo_l"
FACE_DETECTION_SIZE = (640, 640)

# Conservative face recognition settings.
# These should be calibrated later using your real camera.
FACE_MATCH_THRESHOLD = 0.50
FACE_MATCH_MARGIN = 0.08

# Only a strong verified match may automatically improve a face gallery.
FACE_GALLERY_UPDATE_THRESHOLD = 0.65

MAX_FACE_EMBEDDINGS_PER_PERSON = 20
TOP_FACE_SCORES_TO_AVERAGE = 3

MIN_FACE_WIDTH = 50
MIN_FACE_HEIGHT = 50
MIN_FACE_DETECTION_SCORE = 0.65
MIN_FACE_BLUR_SCORE = 25.0

# Prevent almost identical embeddings from filling the gallery.
DUPLICATE_FACE_THRESHOLD = 0.995


# =========================================================
# Lazy InsightFace model
# =========================================================

_face_app = None
_face_model_error_reported = False


def _get_face_app():
    """
    Load InsightFace in GPU-only mode.

    CUDA/cuDNN DLLs are preloaded before InsightFace creates any
    ONNX Runtime sessions. If CUDA is unavailable or a model falls
    back to CPU, initialization fails instead of silently continuing.
    """
    global _face_app, _face_model_error_reported

    if _face_app is not None:
        return _face_app

    try:
        # Import PyTorch first so its CUDA runtime can also be discovered.
        import torch
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        # Load CUDA/cuDNN DLLs installed by:
        # python -m pip install --upgrade "onnxruntime-gpu[cuda,cudnn]"
        try:
            ort.preload_dlls(directory="")
        except TypeError:
            ort.preload_dlls()

        available_providers = ort.get_available_providers()

        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "CUDAExecutionProvider is unavailable. "
                "VisionGuard face recognition is configured for GPU only."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch cannot access the NVIDIA GPU."
            )

        _face_app = FaceAnalysis(
            name=FACE_MODEL_NAME,
            providers=["CUDAExecutionProvider"],
        )

        _face_app.prepare(
            ctx_id=0,
            det_size=FACE_DETECTION_SIZE,
        )

        active_providers = {}

        for model_name, loaded_model in _face_app.models.items():
            session = getattr(loaded_model, "session", None)

            if session is None:
                continue

            model_providers = session.get_providers()
            active_providers[model_name] = model_providers

            if (
                not model_providers
                or model_providers[0] != "CUDAExecutionProvider"
            ):
                raise RuntimeError(
                    f"InsightFace model '{model_name}' fell back to CPU. "
                    f"Actual providers: {model_providers}"
                )

        print(
            f"[FACE GPU] Model={FACE_MODEL_NAME} | "
            f"GPU={torch.cuda.get_device_name(0)}"
        )
        print(f"[FACE GPU] Active providers: {active_providers}")

        return _face_app

    except Exception as error:
        _face_app = None

        if not _face_model_error_reported:
            print("[FACE GPU ERROR] GPU initialization failed.")
            print(f"[FACE GPU ERROR DETAILS] {error}")
            _face_model_error_reported = True

        raise


# =========================================================
# Persistent face memory
# =========================================================

known_faces: Dict[int, Dict[str, Any]] = {}


def normalize_face_embedding(
    embedding: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if embedding is None:
        return None

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).flatten()

    norm = np.linalg.norm(embedding)

    if norm <= 0:
        return None

    return embedding / norm


def _migrate_face_profile(
    person_data: Dict[str, Any],
) -> Dict[str, Any]:
    embeddings = person_data.get("embeddings")

    if embeddings is None:
        old_embedding = person_data.get("embedding")
        embeddings = (
            [old_embedding]
            if old_embedding is not None
            else []
        )

    valid_embeddings: List[np.ndarray] = []

    for embedding in embeddings:
        normalized = normalize_face_embedding(embedding)

        if normalized is not None:
            valid_embeddings.append(normalized)

    person_data["embeddings"] = valid_embeddings[
        -MAX_FACE_EMBEDDINGS_PER_PERSON:
    ]
    person_data.pop("embedding", None)

    person_data.setdefault("display_name", None)
    person_data.setdefault("first_seen", None)
    person_data.setdefault("last_seen", None)

    return person_data


def load_face_memory() -> None:
    global known_faces

    if not os.path.exists(FACE_MEMORY_FILE):
        known_faces = {}
        print("[FACE] No previous face memory found.")
        return

    try:
        with open(FACE_MEMORY_FILE, "rb") as file:
            saved_data = pickle.load(file)

        loaded_faces = saved_data.get(
            "known_faces",
            saved_data if isinstance(saved_data, dict) else {},
        )

        known_faces = {}

        for person_id, person_data in loaded_faces.items():
            known_faces[int(person_id)] = _migrate_face_profile(
                person_data
            )

        print(
            f"[FACE] Loaded {len(known_faces)} "
            "face profile(s)."
        )

    except Exception as error:
        print(f"[FACE LOAD ERROR] {error}")
        known_faces = {}


def save_face_memory() -> None:
    try:
        saved_data = {
            "known_faces": known_faces,
            "saved_at": datetime.now().isoformat(),
        }

        temporary_file = FACE_MEMORY_FILE + ".tmp"

        with open(temporary_file, "wb") as file:
            pickle.dump(saved_data, file)

        os.replace(temporary_file, FACE_MEMORY_FILE)

    except Exception as error:
        print(f"[FACE SAVE ERROR] {error}")


load_face_memory()


# =========================================================
# Face quality and embedding extraction
# =========================================================

def _clamp_face_box(
    bbox: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = np.asarray(bbox).astype(int)

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return x1, y1, x2, y2


def _select_best_face(
    faces: List[Any],
    image_shape: Tuple[int, ...],
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    if not faces:
        return None

    image_height, image_width = image_shape[:2]
    valid_candidates = []

    for face in faces:
        bbox = getattr(face, "bbox", None)
        det_score = float(getattr(face, "det_score", 0.0))

        if bbox is None:
            continue

        x1, y1, x2, y2 = _clamp_face_box(
            bbox,
            image_width,
            image_height,
        )

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width < MIN_FACE_WIDTH:
            continue

        if face_height < MIN_FACE_HEIGHT:
            continue

        if det_score < MIN_FACE_DETECTION_SCORE:
            continue

        area = face_width * face_height
        ranking_score = area * max(det_score, 0.01)

        valid_candidates.append(
            (
                ranking_score,
                face,
                {
                    "bbox": (x1, y1, x2, y2),
                    "det_score": det_score,
                    "width": face_width,
                    "height": face_height,
                },
            )
        )

    if not valid_candidates:
        return None

    valid_candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    _, best_face, face_info = valid_candidates[0]
    return best_face, face_info


def extract_face_embedding(
    person_crop: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Detect the clearest face inside a YOLO person crop.

    Returns:
        (normalized_embedding, face_info)

    face_info includes:
        bbox, det_score, width, height, blur_score
    """
    if person_crop is None or person_crop.size == 0:
        return None, None

    face_app = _get_face_app()

    if face_app is None:
        return None, None

    try:
        faces = face_app.get(person_crop)
        selected = _select_best_face(
            faces,
            person_crop.shape,
        )

        if selected is None:
            return None, None

        face, face_info = selected
        x1, y1, x2, y2 = face_info["bbox"]

        face_crop = person_crop[y1:y2, x1:x2]

        if face_crop is None or face_crop.size == 0:
            return None, None

        gray_face = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2GRAY,
        )

        blur_score = float(
            cv2.Laplacian(
                gray_face,
                cv2.CV_64F,
            ).var()
        )

        face_info["blur_score"] = blur_score

        if blur_score < MIN_FACE_BLUR_SCORE:
            print(
                f"[FACE QUALITY] Face blurred: "
                f"{blur_score:.1f}"
            )
            return None, face_info

        embedding = getattr(face, "embedding", None)
        embedding = normalize_face_embedding(embedding)

        if embedding is None:
            return None, face_info

        return embedding, face_info

    except Exception as error:
        print(f"[FACE EMBEDDING ERROR] {error}")
        return None, None


# =========================================================
# Face matching
# =========================================================

def face_cosine_similarity(
    embedding_a: Optional[np.ndarray],
    embedding_b: Optional[np.ndarray],
) -> float:
    embedding_a = normalize_face_embedding(embedding_a)
    embedding_b = normalize_face_embedding(embedding_b)

    if embedding_a is None or embedding_b is None:
        return 0.0

    return float(np.dot(embedding_a, embedding_b))


def calculate_face_profile_score(
    new_embedding: np.ndarray,
    stored_embeddings: List[np.ndarray],
) -> float:
    if not stored_embeddings:
        return 0.0

    scores = [
        face_cosine_similarity(
            new_embedding,
            stored_embedding,
        )
        for stored_embedding in stored_embeddings
    ]

    scores.sort(reverse=True)

    top_scores = scores[
        :min(
            TOP_FACE_SCORES_TO_AVERAGE,
            len(scores),
        )
    ]

    if not top_scores:
        return 0.0

    return float(np.mean(top_scores))

# =========================================================
# CANONICAL FACE ID ALIASES
# =========================================================

# Multiple face galleries may belong to the same real person.
# Keep their embeddings for recognition, but always resolve
# them to one permanent identity.

FACE_ID_ALIASES = {
    15: 11,
}

def get_canonical_face_id(person_id):
    if person_id is None:
        return None

    current_id = int(person_id)
    visited = set()

    while (
        current_id in FACE_ID_ALIASES
        and current_id not in visited
    ):
        visited.add(current_id)
        current_id = int(
            FACE_ID_ALIASES[current_id]
        )

    return current_id

def find_face_match(
    embedding: Optional[np.ndarray],
    excluded_person_ids: Optional[set] = None,
) -> Tuple[Optional[int], float, float]:
    """
    Compare one face against all persistent face profiles.

    Returns:
        person_id, best_score, score_margin

    person_id is None when the evidence is not strong enough.
    """
    embedding = normalize_face_embedding(embedding)

    if embedding is None:
        return None, -1.0, -1.0

    excluded_person_ids = excluded_person_ids or set()
    excluded_canonical_ids = {
        get_canonical_face_id(person_id)
        for person_id in excluded_person_ids
    }
    canonical_scores = {}

    for person_id, person_data in known_faces.items():
        canonical_id = get_canonical_face_id(person_id)

        if canonical_id in excluded_canonical_ids:
            continue

        stored_embeddings = person_data.get(
            "embeddings",
            [],
        )

        score = calculate_face_profile_score(
            embedding,
            stored_embeddings,
        )

        if (
            canonical_id not in canonical_scores
            or score > canonical_scores[canonical_id]
        ):
            canonical_scores[canonical_id] = score

    candidate_scores = list(
        canonical_scores.items()
    )

    candidate_scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    if not candidate_scores:
        return None, -1.0, -1.0

    best_person_id, best_score = candidate_scores[0]

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

    confident_match = (
        best_score >= FACE_MATCH_THRESHOLD
        and score_margin >= FACE_MATCH_MARGIN
    )

    if confident_match:
        known_faces[best_person_id]["last_seen"] = (
            datetime.now().isoformat()
        )
        save_face_memory()

        print(
            f"[FACE MATCH] Person ID {best_person_id} "
            f"Best={best_score:.3f} "
            f"Second={second_best_score:.3f} "
            f"Margin={score_margin:.3f}"
        )

        return (
            best_person_id,
            best_score,
            score_margin,
        )

    print(
        f"[FACE NO MATCH] Best Person ID={best_person_id} "
        f"Best={best_score:.3f} "
        f"Second={second_best_score:.3f} "
        f"Margin={score_margin:.3f}"
    )

    return None, best_score, score_margin


def recognize_face(
    person_crop: Optional[np.ndarray],
    excluded_person_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Complete helper for main_reid.py.

    A face may be detected even when it does not yet match a known person.
    The returned embedding can later be registered under the final stable ID.
    """
    embedding, face_info = extract_face_embedding(
        person_crop
    )

    result = {
        "face_detected": embedding is not None,
        "person_id": None,
        "display_name": None,
        "score": -1.0,
        "margin": -1.0,
        "embedding": embedding,
        "face_info": face_info,
    }

    if embedding is None:
        return result

    person_id, score, margin = find_face_match(
        embedding,
        excluded_person_ids=excluded_person_ids,
    )

    result["person_id"] = person_id
    result["score"] = score
    result["margin"] = margin

    if person_id is not None:
        result["display_name"] = get_face_display_name(
            person_id
        )

    return result


# =========================================================
# Face gallery management
# =========================================================

def _append_face_embedding(
    person_id: int,
    embedding: np.ndarray,
) -> bool:
    person_data = known_faces.get(person_id)

    if person_data is None:
        return False

    embedding = normalize_face_embedding(embedding)

    if embedding is None:
        return False

    embeddings = person_data.setdefault(
        "embeddings",
        [],
    )

    for old_embedding in embeddings:
        if (
            face_cosine_similarity(
                embedding,
                old_embedding,
            )
            >= DUPLICATE_FACE_THRESHOLD
        ):
            return False

    embeddings.append(embedding)

    if len(embeddings) > MAX_FACE_EMBEDDINGS_PER_PERSON:
        person_data["embeddings"] = embeddings[
            -MAX_FACE_EMBEDDINGS_PER_PERSON:
        ]

    person_data["last_seen"] = (
        datetime.now().isoformat()
    )

    return True


def register_face_embedding(
    person_id: int,
    embedding: Optional[np.ndarray],
    display_name: Optional[str] = None,
) -> bool:
    """
    Register a verified face under an existing permanent Person ID.

    Use this for:
    - manual familiar-person enrollment
    - the first clear face saved for a newly confirmed stable ID
    """
    embedding = normalize_face_embedding(embedding)

    if embedding is None:
        return False

    person_id = int(person_id)
    now_text = datetime.now().isoformat()

    if person_id not in known_faces:
        known_faces[person_id] = {
            "display_name": display_name,
            "embeddings": [],
            "first_seen": now_text,
            "last_seen": now_text,
        }

    elif display_name:
        known_faces[person_id]["display_name"] = (
            display_name
        )

    added = _append_face_embedding(
        person_id,
        embedding,
    )

    if added:
        save_face_memory()

        print(
            f"[FACE REGISTERED] Person ID {person_id} "
            f"Name={known_faces[person_id].get('display_name')} "
            f"Faces={len(known_faces[person_id]['embeddings'])}"
        )

    return added


def register_face_from_crop(
    person_id: int,
    person_crop: Optional[np.ndarray],
    display_name: Optional[str] = None,
) -> bool:
    embedding, _ = extract_face_embedding(
        person_crop
    )

    return register_face_embedding(
        person_id=person_id,
        embedding=embedding,
        display_name=display_name,
    )


def update_verified_face_gallery(
    person_id: int,
    embedding: Optional[np.ndarray],
    verified_match_score: float,
) -> bool:
    """
    Automatically add a new angle only when identity was already verified
    by a strong face match.
    """
    if verified_match_score < FACE_GALLERY_UPDATE_THRESHOLD:
        return False

    return register_face_embedding(
        person_id=person_id,
        embedding=embedding,
    )


def set_face_display_name(
    person_id: int,
    display_name: Optional[str],
) -> bool:
    person_id = int(person_id)

    if person_id not in known_faces:
        return False

    known_faces[person_id]["display_name"] = (
        display_name.strip()
        if display_name
        else None
    )

    save_face_memory()
    return True


def get_face_display_name(
    person_id: Optional[int],
) -> Optional[str]:
    if person_id is None:
        return None

    person_data = known_faces.get(int(person_id))

    if person_data is None:
        return None

    return person_data.get("display_name")


def get_face_profile(
    person_id: int,
) -> Optional[Dict[str, Any]]:
    person_data = known_faces.get(int(person_id))

    if person_data is None:
        return None

    return {
        "person_id": int(person_id),
        "display_name": person_data.get(
            "display_name"
        ),
        "face_count": len(
            person_data.get("embeddings", [])
        ),
        "first_seen": person_data.get(
            "first_seen"
        ),
        "last_seen": person_data.get(
            "last_seen"
        ),
    }


def list_face_profiles() -> List[Dict[str, Any]]:
    return [
        get_face_profile(person_id)
        for person_id in sorted(known_faces)
    ]


def delete_face_profile(person_id: int) -> bool:
    person_id = int(person_id)

    if person_id not in known_faces:
        return False

    del known_faces[person_id]
    save_face_memory()

    print(
        f"[FACE DELETED] Person ID {person_id}"
    )

    return True
