from __future__ import annotations

from pathlib import Path

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
DIRECTORY_NAMES = (
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
)
DISPLAY_CLASS_NAMES = DIRECTORY_NAMES
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_TRAIN_DIR = DEFAULT_DATA_ROOT / "Training"
DEFAULT_TEST_DIR = DEFAULT_DATA_ROOT / "Testing"
DEFAULT_PREDICT_DIR = DEFAULT_DATA_ROOT / "New"

DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 16
DEFAULT_MODEL_NAME = "resnet50"
DATASET_SLUG = "prathamgrover/brain-tumor-classification"
