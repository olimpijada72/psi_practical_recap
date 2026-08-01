"""Dataset and DataLoader definitions for breast cancer classification."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class BreastCancerDataset(Dataset):
    """A map-style PyTorch Dataset backed by a processed CSV file."""

    def __init__(self, csv_path: Path, target_column: str = "malignant") -> None:
        dataframe = pd.read_csv(csv_path)

        if target_column not in dataframe.columns:
            raise ValueError(f"Missing target column: {target_column}")
        if dataframe.isna().any().any():
            raise ValueError(f"Missing values found in {csv_path}")

        self.feature_names = [
            column for column in dataframe.columns if column != target_column
        ]
        self.features = torch.tensor(
            dataframe[self.feature_names].to_numpy(), dtype=torch.float32
        )
        self.labels = torch.tensor(
            dataframe[target_column].to_numpy(), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create loaders for the prepared train, validation, and test splits."""
    training_data = BreastCancerDataset(data_dir / "train.csv")
    validation_data = BreastCancerDataset(data_dir / "validation.csv")
    test_data = BreastCancerDataset(data_dir / "test.csv")

    if not (
        training_data.feature_names
        == validation_data.feature_names
        == test_data.feature_names
    ):
        raise ValueError("Feature columns differ between the data splits")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return (
        train_loader,
        validation_loader,
        test_loader,
        training_data.feature_names,
    )
