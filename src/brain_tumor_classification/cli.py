from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from brain_tumor_classification.config import load_toml_file
from brain_tumor_classification.constants import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PREDICT_DIR,
    DEFAULT_SEED,
)
from brain_tumor_classification.data import (
    build_dataset_manifest,
    build_duplicate_report,
    download_kaggle_dataset,
    ensure_split_manifest,
    inspect_dataset,
)
from brain_tumor_classification.evaluation import evaluate_checkpoint
from brain_tumor_classification.inference import predict_folder, predict_image
from brain_tumor_classification.training import TrainConfig, train_model


def _apply_train_config_overrides(args: argparse.Namespace) -> TrainConfig:
    config_values: dict[str, object] = {}
    if args.config is not None:
        config_values.update(load_toml_file(args.config))

    for key in (
        "data_root",
        "output_root",
        "split_manifest",
        "seed",
        "epochs",
        "batch_size",
        "num_workers",
        "learning_rate_backbone",
        "learning_rate_head",
        "weight_decay",
        "validation_fraction",
        "early_stopping_patience",
        "device",
        "use_amp",
        "image_size",
        "dry_run",
        "load_pretrained_weights",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config_values[key] = value

    for path_key in ("data_root", "output_root", "split_manifest"):
        if path_key in config_values:
            config_values[path_key] = Path(str(config_values[path_key]))

    return TrainConfig(
        data_root=cast(Path, config_values.get("data_root", DEFAULT_DATA_ROOT)),
        output_root=cast(Path, config_values.get("output_root", Path("artifacts") / "runs")),
        split_manifest=cast(
            Path,
            config_values.get(
                "split_manifest",
                DEFAULT_DATA_ROOT / "manifests" / "train_validation_split.csv",
            ),
        ),
        seed=cast(int, config_values.get("seed", DEFAULT_SEED)),
        epochs=cast(int, config_values.get("epochs", 16)),
        batch_size=cast(int, config_values.get("batch_size", 32)),
        num_workers=cast(int, config_values.get("num_workers", 0)),
        learning_rate_backbone=cast(float, config_values.get("learning_rate_backbone", 1e-5)),
        learning_rate_head=cast(float, config_values.get("learning_rate_head", 1e-4)),
        weight_decay=cast(float, config_values.get("weight_decay", 1e-4)),
        validation_fraction=cast(float, config_values.get("validation_fraction", 0.2)),
        early_stopping_patience=cast(int, config_values.get("early_stopping_patience", 3)),
        device=cast(str, config_values.get("device", "auto")),
        use_amp=cast(bool, config_values.get("use_amp", True)),
        image_size=cast(int, config_values.get("image_size", 224)),
        dry_run=cast(bool, config_values.get("dry_run", False)),
        load_pretrained_weights=cast(
            bool,
            config_values.get("load_pretrained_weights", True),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-tumor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    download_parser = data_subparsers.add_parser("download")
    download_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    download_parser.add_argument("--manifest-path", type=Path, default=None)
    download_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    inspect_parser = data_subparsers.add_parser("inspect")
    inspect_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    inspect_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    inspect_parser.add_argument("--manifest-path", type=Path, default=None)
    inspect_parser.add_argument("--split-manifest", type=Path, default=None)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, default=None)
    train_parser.add_argument("--data-root", type=Path, default=None)
    train_parser.add_argument("--output-root", type=Path, default=None)
    train_parser.add_argument("--split-manifest", type=Path, default=None)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)
    train_parser.add_argument("--num-workers", type=int, default=None)
    train_parser.add_argument("--learning-rate-backbone", type=float, default=None)
    train_parser.add_argument("--learning-rate-head", type=float, default=None)
    train_parser.add_argument("--weight-decay", type=float, default=None)
    train_parser.add_argument("--validation-fraction", type=float, default=None)
    train_parser.add_argument("--early-stopping-patience", type=int, default=None)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    train_parser.add_argument("--use-amp", action="store_true", default=None)
    train_parser.add_argument("--no-amp", action="store_false", dest="use_amp")
    train_parser.add_argument("--image-size", type=int, default=None)
    train_parser.add_argument("--dry-run", action="store_true", default=None)
    train_parser.add_argument("--load-pretrained-weights", action="store_true", default=None)
    train_parser.add_argument(
        "--no-pretrained-weights",
        action="store_false",
        dest="load_pretrained_weights",
    )

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--test-root", type=Path, default=DEFAULT_DATA_ROOT / "Testing")
    evaluate_parser.add_argument("--output-dir", type=Path, default=None)
    evaluate_parser.add_argument("--batch-size", type=int, default=32)
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate_parser.add_argument(
        "--exclude-leaked-test-images",
        action="store_true",
    )
    evaluate_parser.add_argument("--train-root", type=Path, default=DEFAULT_DATA_ROOT / "Training")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, default=None)
    predict_parser.add_argument("--folder", type=Path, default=DEFAULT_PREDICT_DIR)
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict_parser.add_argument("--json-output", type=Path, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "data" and args.data_command == "download":
        archive_path, manifest_path = download_kaggle_dataset(
            data_root=args.data_root,
            manifest_path=args.manifest_path,
            seed=args.seed,
        )
        print(
            json.dumps(
                {"archive_path": str(archive_path), "manifest_path": str(manifest_path)},
                indent=2,
            )
        )
        return

    if args.command == "data" and args.data_command == "inspect":
        data_root = args.data_root
        split_manifest = ensure_split_manifest(
            train_root=data_root / "Training",
            manifest_path=args.split_manifest,
            seed=args.seed,
        )
        manifest_path = args.manifest_path or Path("configs") / "brain-tumor.dataset.json"
        build_dataset_manifest(
            data_root=data_root,
            seed=args.seed,
            output_path=manifest_path,
        )
        payload = {
            "train": inspect_dataset(data_root / "Training"),
            "test": inspect_dataset(data_root / "Testing"),
            "duplicates": build_duplicate_report(
                train_root=data_root / "Training",
                test_root=data_root / "Testing",
            ),
            "split_manifest": str(split_manifest),
            "dataset_manifest": str(manifest_path),
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "train":
        result = train_model(_apply_train_config_overrides(args))
        printable_result = {
            key: str(value) if isinstance(value, Path) else value for key, value in result.items()
        }
        print(json.dumps(printable_result, indent=2))
        return

    if args.command == "evaluate":
        result = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            test_root=args.test_root,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
            exclude_leaked_test_images=args.exclude_leaked_test_images,
            train_root=args.train_root,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "predict":
        if args.image is not None:
            result = predict_image(args.checkpoint, args.image, device_name=args.device)
            print(json.dumps(result, indent=2))
            return
        results = predict_folder(
            args.checkpoint,
            folder_path=args.folder,
            device_name=args.device,
            output_path=args.json_output,
        )
        print(json.dumps(results, indent=2))
        return


if __name__ == "__main__":
    main()
