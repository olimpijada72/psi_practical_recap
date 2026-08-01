"""Reusable training and evaluation loops."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Metrics:
    """Metrics collected over one complete dataset split."""

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one pass over the training Dataset and return mean loss."""
    model.train()
    total_loss = 0.0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        logits = model(features)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> Metrics:
    """Evaluate without gradient updates and return binary metrics."""
    model.eval()
    total_loss = 0.0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correct = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            total_loss += loss_fn(logits, labels).item() * features.size(0)
            predictions = (torch.sigmoid(logits) >= 0.5).float()

            correct += (predictions == labels).sum().item()
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

    size = len(dataloader.dataset)
    accuracy = correct / size
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return Metrics(
        loss=total_loss / size,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )
