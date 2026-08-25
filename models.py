from pathlib import Path

from brain_tumor_classification.checkpoints import save_legacy_state_dict
from brain_tumor_classification.modeling import TransferLearningResNet

model_factory = {"resnet": TransferLearningResNet}


def save_model(model) -> Path:  # type: ignore[no-untyped-def]
    output_path = Path("./resnet_model.pth")
    save_legacy_state_dict(output_path, model)
    print(f"Model saved as {output_path.name}")
    return output_path
