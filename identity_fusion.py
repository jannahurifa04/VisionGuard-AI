from datetime import datetime

# =====================================
# Identity Fusion Engine
# =====================================

SHIRT_SCORE = 0.15
REID_SCORE = 0.70
TIME_SCORE = 0.15

MATCH_THRESHOLD = 0.75


def calculate_identity_score(
        reid_similarity,
        shirt_match,
        seconds_since_seen
):
    """
    Returns score between 0 and 1
    """

    score = 0

    # ReID
    score += reid_similarity * REID_SCORE

    # Shirt color
    if shirt_match:
        score += SHIRT_SCORE

    # Recent visit bonus
    if seconds_since_seen < 60:
        score += TIME_SCORE

    return score


def is_same_person(
        reid_similarity,
        shirt_match,
        seconds_since_seen
):

    score = calculate_identity_score(
        reid_similarity,
        shirt_match,
        seconds_since_seen
    )

    return score >= MATCH_THRESHOLD, score