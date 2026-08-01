"""Train and evaluate the breast cancer classifier.

Run from the repository root with:

    python src/train.py
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset import create_dataloaders
from engine import evaluate, train_one_epoch
from model import NeuralNetwork, initialize_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "pytorch_classifier.pth",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed the random-number generators used by this prototype."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()
    print(f"Using {device} device")

    train_loader, validation_loader, test_loader, feature_names = (
        create_dataloaders(args.data_dir, args.batch_size, args.seed)
    )
    print(
        f"Samples: train={len(train_loader.dataset)}, "
        f"validation={len(validation_loader.dataset)}, "
        f"test={len(test_loader.dataset)}"
    )

    model = NeuralNetwork(len(feature_names)).to(device)
    model.apply(initialize_parameters)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_validation_f1 = -1.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            train_loader, model, loss_fn, optimizer, device
        )
        validation_metrics = evaluate(
            validation_loader, model, loss_fn, device
        )

        if validation_metrics.f1 > best_validation_f1:
            best_validation_f1 = validation_metrics.f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:>2}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"validation loss: {validation_metrics.loss:.4f} | "
            f"validation F1: {validation_metrics.f1:.3f}"
        )

    if best_model_state is None:
        raise RuntimeError("Training completed without producing a model")

    model.load_state_dict(best_model_state)
    test_metrics = evaluate(test_loader, model, loss_fn, device)

    print(f"\nRestored epoch {best_epoch} (validation F1={best_validation_f1:.3f})")
    print(f"Test loss:      {test_metrics.loss:.4f}")
    print(f"Test accuracy:  {test_metrics.accuracy:.1%}")
    print(f"Test precision: {test_metrics.precision:.3f}")
    print(f"Test recall:    {test_metrics.recall:.3f}")
    print(f"Test F1:        {test_metrics.f1:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "feature_names": feature_names,
            "best_epoch": best_epoch,
            "validation_f1": best_validation_f1,
            "test_metrics": vars(test_metrics),
        },
        args.output,
    )
    print(f"Saved model checkpoint to {args.output}")


if __name__ == "__main__":
    main()
