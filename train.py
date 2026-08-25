import argparse
from pathlib import Path

from brain_tumor_classification.training import TrainConfig, train_model


def train(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    output_root = Path(args.log_dir) if args.log_dir else Path("artifacts") / "runs"
    config = TrainConfig(output_root=output_root)
    return train_model(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=None)
    parsed_args = parser.parse_args()
    train(parsed_args)
