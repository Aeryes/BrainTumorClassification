from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from brain_tumor_classification.constants import DEFAULT_MODEL_NAME, NUM_CLASSES


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    num_classes: int = NUM_CLASSES
    dropout_rate_1: float = 0.5
    dropout_rate_2: float = 0.5
    load_pretrained_weights: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TransferLearningResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate_1: float = 0.5,
        dropout_rate_2: float = 0.5,
        *,
        load_pretrained_weights: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if load_pretrained_weights else None
        self.resnet = resnet50(weights=weights)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        dropout_probability = (dropout_rate_1 + dropout_rate_2) / 2
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.resnet(x)


def build_model(config: ModelConfig) -> TransferLearningResNet:
    if config.model_name != DEFAULT_MODEL_NAME:
        raise ValueError(f"Unsupported model_name: {config.model_name}")

    return TransferLearningResNet(
        num_classes=config.num_classes,
        dropout_rate_1=config.dropout_rate_1,
        dropout_rate_2=config.dropout_rate_2,
        load_pretrained_weights=config.load_pretrained_weights,
    )
