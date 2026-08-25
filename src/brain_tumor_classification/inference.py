from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from brain_tumor_classification.checkpoints import load_model_from_checkpoint
from brain_tumor_classification.constants import (
    CLASS_NAMES,
    DEFAULT_PREDICT_DIR,
    DISPLAY_CLASS_NAMES,
    IMAGE_EXTENSIONS,
)
from brain_tumor_classification.training import resolve_device
from brain_tumor_classification.transforms import build_eval_transform


def predict_image(
    checkpoint_path: Path,
    image_path: Path,
    *,
    device_name: str = "auto",
) -> dict[str, object]:
    device = resolve_device(device_name)
    model, metadata = load_model_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    transform = build_eval_transform()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        predicted = int(output.argmax(dim=1).item())
        probabilities = torch.softmax(output, dim=1)[0].tolist()

    class_names = metadata.get("class_names", list(CLASS_NAMES))
    display_names = dict(zip(class_names, DISPLAY_CLASS_NAMES, strict=True))
    predicted_class = class_names[predicted]
    return {
        "image_path": str(image_path),
        "predicted_index": predicted,
        "predicted_class": predicted_class,
        "display_class": display_names[predicted_class],
        "probabilities": dict(zip(class_names, probabilities, strict=True)),
    }


def predict_folder(
    checkpoint_path: Path,
    folder_path: Path = DEFAULT_PREDICT_DIR,
    *,
    device_name: str = "auto",
    output_path: Path | None = None,
) -> list[dict[str, object]]:
    predictions = []
    for image_path in sorted(folder_path.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            predictions.append(predict_image(checkpoint_path, image_path, device_name=device_name))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return predictions


def display_prediction(image_path: Path, predicted_label: str) -> None:
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_label}")
    plt.axis("off")
    plt.show()
