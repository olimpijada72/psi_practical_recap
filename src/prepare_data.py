"""Load, split, scale, and save the breast cancer dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Prepare reproducible train, validation, and test CSV files."""
    dataset = load_breast_cancer(as_frame=True)
    raw_df = dataset.frame.copy()

    # In scikit-learn's dataset, target 0 is malignant and target 1 is benign.
    features = raw_df.drop(columns="target")
    labels = (raw_df["target"] == 0).astype(np.float32).rename("malignant")

    # 60% training, 20% validation, 20% test. Stratification preserves the
    # class proportions in every split.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    # Fit only on training data to avoid leaking validation/test information.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    def make_processed_frame(
        scaled_features: np.ndarray, labels: pd.Series
    ) -> pd.DataFrame:
        frame = pd.DataFrame(scaled_features, columns=features.columns)
        frame["malignant"] = labels.to_numpy()
        return frame

    train_df = make_processed_frame(X_train_scaled, y_train)
    validation_df = make_processed_frame(X_validation_scaled, y_validation)
    test_df = make_processed_frame(X_test_scaled, y_test)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "breast_cancer_data.csv"
    raw_df.to_csv(raw_path, index=False)

    processed_splits = {
        "train.csv": train_df,
        "validation.csv": validation_df,
        "test.csv": test_df,
    }
    for filename, frame in processed_splits.items():
        frame.to_csv(PROCESSED_DIR / filename, index=False)

    print(f"Saved raw data: {raw_path} ({len(raw_df)} rows)")
    for filename, frame in processed_splits.items():
        print(f"Saved {filename}: {len(frame)} rows")


if __name__ == "__main__":
    main()
