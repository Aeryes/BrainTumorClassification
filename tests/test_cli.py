from __future__ import annotations

from brain_tumor_classification.cli import build_parser


def test_cli_parses_train_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["train"])
    assert args.command == "train"
    assert args.device is None


def test_cli_parses_predict_image() -> None:
    parser = build_parser()
    args = parser.parse_args(["predict", "--checkpoint", "model.pt", "--image", "sample.png"])
    assert args.command == "predict"
    assert str(args.image).endswith("sample.png")
