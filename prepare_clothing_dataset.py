from pathlib import Path
import shutil

import cv2
import numpy as np


# =========================
# SETTINGS
# =========================

RAW_ROOT = Path("clothing_color_raw")
DATASET_ROOT = Path("clothing_color_dataset")
REJECTED_ROOT = Path("clothing_color_rejected")

COLORS = [
    "Black",
    "White",
    "Pink",
    "Gray",
    "Yellow",
]

# With about 30 images per colour:
# 20 train + 5 validation + 5 test
TRAIN_COUNT = 20
VAL_COUNT = 5

# Image quality filtering
MIN_CLOTHING_RATIO = 0.18
MIN_COMPONENT_RATIO = 0.12
MIN_WIDTH = 80
MIN_HEIGHT = 80


# =========================
# QUALITY CHECK
# =========================

def clothing_quality_ok(image):
    """
    Check whether enough real clothing remains in the image.

    The capture script uses a neutral gray background.
    We estimate that background from the image borders
    and measure how much foreground/clothing remains.
    """

    if image is None or image.size == 0:
        return False, "invalid image"

    height, width = image.shape[:2]

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False, "image too small"

    # Get border pixels to estimate the neutral background.
    top = image[0:5, :, :].reshape(-1, 3)
    bottom = image[-5:, :, :].reshape(-1, 3)
    left = image[:, 0:5, :].reshape(-1, 3)
    right = image[:, -5:, :].reshape(-1, 3)

    border_pixels = np.concatenate(
        [top, bottom, left, right],
        axis=0,
    )

    background_color = np.median(
        border_pixels,
        axis=0,
    )

    difference = np.linalg.norm(
        image.astype(np.float32)
        - background_color.astype(np.float32),
        axis=2,
    )

    # Clothing pixels differ from the neutral background.
    foreground = (
        difference > 12
    ).astype(np.uint8)

    # Clean tiny JPEG/noise artifacts.
    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    total_pixels = height * width

    clothing_pixels = int(
        np.count_nonzero(foreground)
    )

    clothing_ratio = (
        clothing_pixels
        / max(total_pixels, 1)
    )

    if clothing_ratio < MIN_CLOTHING_RATIO:
        return (
            False,
            f"too little clothing ({clothing_ratio:.2f})",
        )

    # Check that there is one substantial connected area,
    # rather than many tiny broken pieces.
    number_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8,
        )
    )

    if number_labels <= 1:
        return False, "no clothing component"

    component_areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_component = int(
        component_areas.max()
    )

    largest_ratio = (
        largest_component
        / max(total_pixels, 1)
    )

    if largest_ratio < MIN_COMPONENT_RATIO:
        return (
            False,
            f"fragmented mask ({largest_ratio:.2f})",
        )

    return True, (
        f"good ratio={clothing_ratio:.2f} "
        f"component={largest_ratio:.2f}"
    )


# =========================
# PREPARE DATASET
# =========================

def prepare_colour(colour):
    raw_folder = RAW_ROOT / colour

    if not raw_folder.exists():
        print(
            f"[SKIP] {colour}: raw folder missing"
        )
        return

    files = sorted(
        list(raw_folder.glob("*.jpg"))
        + list(raw_folder.glob("*.jpeg"))
        + list(raw_folder.glob("*.png"))
    )

    accepted = []
    rejected = []

    print()
    print(
        f"========== {colour} =========="
    )

    for file_path in files:
        image = cv2.imread(
            str(file_path)
        )

        good, reason = clothing_quality_ok(
            image
        )

        if good:
            accepted.append(file_path)

            print(
                f"[GOOD] {file_path.name} | {reason}"
            )

        else:
            rejected.append(file_path)

            print(
                f"[REJECT] {file_path.name} | {reason}"
            )

    # Rejected images are COPIED here.
    # Raw originals are never deleted.
    rejected_folder = (
        REJECTED_ROOT
        / colour
    )

    rejected_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file_path in rejected:
        shutil.copy2(
            file_path,
            rejected_folder / file_path.name,
        )

    if len(accepted) < 15:
        print(
            f"[WARNING] Only {len(accepted)} good "
            f"{colour} images."
        )

    # Keep temporal order.
    # Sequential frames are NOT randomly mixed across
    # train/val/test, reducing near-duplicate leakage.
    train_files = accepted[
        :TRAIN_COUNT
    ]

    val_files = accepted[
        TRAIN_COUNT:
        TRAIN_COUNT + VAL_COUNT
    ]

    test_files = accepted[
        TRAIN_COUNT + VAL_COUNT:
    ]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    for split_name, split_files in splits.items():

        destination = (
            DATASET_ROOT
            / split_name
            / colour
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Clean old generated files for this colour.
        for old_file in destination.glob("*"):
            if old_file.is_file():
                old_file.unlink()

        for file_path in split_files:
            shutil.copy2(
                file_path,
                destination / file_path.name,
            )

    print()
    print(
        f"[RESULT] {colour}"
    )
    print(
        f"  Raw:      {len(files)}"
    )
    print(
        f"  Good:     {len(accepted)}"
    )
    print(
        f"  Rejected: {len(rejected)}"
    )
    print(
        f"  Train:    {len(train_files)}"
    )
    print(
        f"  Val:      {len(val_files)}"
    )
    print(
        f"  Test:     {len(test_files)}"
    )


# =========================
# MAIN
# =========================

print(
    "[DATASET] Preparing clothing colour dataset..."
)

for colour in COLORS:
    prepare_colour(colour)

print()
print(
    "[DATASET] Preparation finished."
)
print(
    "[DATASET] Raw images were NOT deleted."
)