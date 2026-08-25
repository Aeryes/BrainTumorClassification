from pathlib import Path

from brain_tumor_classification.evaluation import evaluate_checkpoint


def main() -> None:
    metrics = evaluate_checkpoint(
        checkpoint_path=Path("./resnet_model.pth"),
        output_dir=Path("artifacts") / "legacy-visualize",
    )
    print(metrics)


if __name__ == "__main__":
    main()
