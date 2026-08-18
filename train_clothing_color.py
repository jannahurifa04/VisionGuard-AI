from pathlib import Path
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# =========================================================
# SETTINGS
# =========================================================

DATASET_ROOT = Path("clothing_color_dataset")

CLASSES = [
    "Black",
    "White",
    "Pink",
    "Gray",
    "Yellow",
]

MODEL_OUTPUT = "clothing_color_classifier.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25

LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.0001

EARLY_STOPPING_PATIENCE = 6

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_SEED
    )


# =========================================================
# COLOR CONSTANCY PREPROCESSING
# =========================================================
# Gray-world white balance: normalizes color CAST caused by lighting
# (e.g. warm indoor light making white look yellowish), applied
# identically at train time and inference time. This does NOT touch
# brightness/lightness, so it targets hue confusion (Yellow/White)
# without making the Black/White/Gray brightness-based classes harder.
#
# IMPORTANT: this exact function must also be applied in main_reid.py
# to the clothing crop before calling the classifier at inference time.
# If it's only applied during training, it won't help live predictions.

def gray_world_white_balance(pil_image):
    """Apply gray-world assumption white balance. Input/output: PIL RGB image."""
    img = np.asarray(pil_image).astype(np.float32)

    avg_r = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_b = np.mean(img[:, :, 2])
    avg_gray = (avg_r + avg_g + avg_b) / 3.0

    img[:, :, 0] *= (avg_gray / (avg_r + 1e-6))
    img[:, :, 1] *= (avg_gray / (avg_g + 1e-6))
    img[:, :, 2] *= (avg_gray / (avg_b + 1e-6))

    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


class WhiteBalance:
    """Picklable transform wrapper so DataLoader workers can use it."""
    def __call__(self, pil_image):
        return gray_world_white_balance(pil_image)


# =========================================================
# DATASET
# =========================================================

class ClothingColorDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        class_names,
        transform=None,
    ):
        self.transform = transform
        self.samples = []

        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(class_names)
        }

        split_root = (
            Path(root)
            / split
        )

        for class_name in class_names:

            class_folder = (
                split_root
                / class_name
            )

            if not class_folder.exists():
                continue

            image_files = []

            for extension in (
                "*.jpg",
                "*.jpeg",
                "*.png",
            ):
                image_files.extend(
                    class_folder.glob(
                        extension
                    )
                )

            for image_path in sorted(
                image_files
            ):
                self.samples.append(
                    (
                        image_path,
                        self.class_to_idx[
                            class_name
                        ],
                    )
                )

        print(
            f"[DATASET] {split}: "
            f"{len(self.samples)} images"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(
                image
            )

        return image, label


# =========================================================
# TRANSFORMS
# =========================================================

weights = (
    MobileNet_V3_Small_Weights.DEFAULT
)

normalization = transforms.Normalize(
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
)


train_transform = transforms.Compose(
    [
        # Color constancy FIRST, before other augmentation.
        WhiteBalance(),

        transforms.Resize(
            (256, 256)
        ),

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(
                0.88,
                1.0,
            ),
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomAffine(
            degrees=5,
            translate=(
                0.03,
                0.03,
            ),
            scale=(
                0.96,
                1.04,
            ),
        ),

        # Widened from the original (brightness=0.08, contrast=0.06).
        # Black/White/Gray are brightness-separated classes, so the
        # model needs real brightness/contrast variety to generalize
        # across lighting conditions. Saturation kept low since color
        # itself is the label.
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.15,
            saturation=0.05,
        ),

        transforms.ToTensor(),

        normalization,
    ]
)


eval_transform = transforms.Compose(
    [
        # Same white balance step as training, for consistency.
        WhiteBalance(),

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.ToTensor(),

        normalization,
    ]
)


# =========================================================
# LOAD DATA
# =========================================================

train_dataset = ClothingColorDataset(
    DATASET_ROOT,
    "train",
    CLASSES,
    transform=train_transform,
)

val_dataset = ClothingColorDataset(
    DATASET_ROOT,
    "val",
    CLASSES,
    transform=eval_transform,
)

test_dataset = ClothingColorDataset(
    DATASET_ROOT,
    "test",
    CLASSES,
    transform=eval_transform,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,

    # Windows-friendly.
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# =========================================================
# MODEL
# =========================================================

print()
print(
    "[MODEL] Loading pretrained "
    "MobileNetV3-Small..."
)

model = mobilenet_v3_small(
    weights=weights
)


# Freeze pretrained feature extractor first.
for parameter in (
    model.features.parameters()
):
    parameter.requires_grad = False


input_features = (
    model.classifier[3].in_features
)

model.classifier[3] = nn.Linear(
    input_features,
    len(CLASSES),
)

model = model.to(
    DEVICE
)


print(
    f"[MODEL] Device: {DEVICE}"
)

print(
    f"[MODEL] Classes: {CLASSES}"
)


# =========================================================
# LOSS / OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(
        lambda parameter:
        parameter.requires_grad,
        model.parameters(),
    ),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    model,
    loader,
):

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += int(
                (
                    predictions
                    == labels
                )
                .sum()
                .item()
            )

            total += labels.size(0)

    average_loss = (
        total_loss
        / max(total, 1)
    )

    accuracy = (
        correct
        / max(total, 1)
    )

    return (
        average_loss,
        accuracy,
    )


# =========================================================
# TRAIN
# =========================================================

print()
print(
    "[TRAIN] Starting training..."
)
print()


best_val_accuracy = 0.0
epochs_without_improvement = 0


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += int(
            (
                predictions
                == labels
            )
            .sum()
            .item()
        )

        total += labels.size(0)


    train_loss = (
        running_loss
        / max(total, 1)
    )

    train_accuracy = (
        correct
        / max(total, 1)
    )


    val_loss, val_accuracy = (
        evaluate(
            model,
            val_loader,
        )
    )


    print(
        f"[EPOCH {epoch:02d}] "
        f"Train Loss={train_loss:.4f} "
        f"Train Acc={train_accuracy:.2%} | "
        f"Val Loss={val_loss:.4f} "
        f"Val Acc={val_accuracy:.2%}"
    )


    if (
        val_accuracy
        > best_val_accuracy
    ):

        best_val_accuracy = (
            val_accuracy
        )

        epochs_without_improvement = 0


        checkpoint = {
            "model_name":
                "mobilenet_v3_small",

            "classes":
                CLASSES,

            "image_size":
                IMAGE_SIZE,

            "state_dict":
                model.state_dict(),

            "best_val_accuracy":
                best_val_accuracy,

            # Flag so main_reid.py knows this checkpoint expects
            # white-balanced input and can apply the same preprocessing.
            "expects_white_balance":
                True,
        }


        torch.save(
            checkpoint,
            MODEL_OUTPUT,
        )


        print(
            "[MODEL SAVED] "
            f"Val Acc="
            f"{best_val_accuracy:.2%}"
        )


    else:

        epochs_without_improvement += 1


    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print()
        print(
            "[TRAIN] Early stopping."
        )

        break


# =========================================================
# LOAD BEST MODEL
# =========================================================

print()
print(
    "[TEST] Loading best model..."
)


checkpoint = torch.load(
    MODEL_OUTPUT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["state_dict"]
)


# =========================================================
# TEST
# =========================================================

test_loss, test_accuracy = (
    evaluate(
        model,
        test_loader,
    )
)


print()
print(
    f"[TEST] Accuracy: "
    f"{test_accuracy:.2%}"
)

print(
    f"[TEST] Loss: "
    f"{test_loss:.4f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

confusion_matrix = torch.zeros(
    len(CLASSES),
    len(CLASSES),
    dtype=torch.int64,
)


model.eval()

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )

        outputs = model(
            images
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
            .cpu()
        )


        for true_label, prediction in zip(
            labels,
            predictions,
        ):

            confusion_matrix[
                int(true_label),
                int(prediction),
            ] += 1


print()
print(
    "[TEST] Confusion Matrix"
)

print(
    "Rows = actual"
)

print(
    "Columns = predicted"
)

print()

print(
    "        "
    + " ".join(
        f"{name[:5]:>6}"
        for name in CLASSES
    )
)


for index, class_name in enumerate(
    CLASSES
):

    values = " ".join(
        f"{int(value):6d}"
        for value
        in confusion_matrix[index]
    )

    print(
        f"{class_name[:7]:>7} "
        + values
    )


print()
print(
    "[DONE] Deep clothing colour "
    "classifier training finished."
)

print(
    f"[DONE] Model saved as: "
    f"{MODEL_OUTPUT}"
)

print()
print(
    "[REMINDER] Apply gray_world_white_balance() to the clothing "
    "crop in main_reid.py BEFORE running it through this classifier "
    "at inference time - training and inference preprocessing must match."
)