"""Finalize the committed aligned data without fitting on test rows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config

LOGGER = logging.getLogger(__name__)


class IdentityScaler:
    """Fallback scaler for already-scaled recovered inputs."""

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)


def _read_split(name: str, split: str) -> pd.DataFrame:
    frame = pd.read_csv(config.aligned_path(name, split))
    expected = config.FEATURES + [config.LABEL]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected columns in {name} {split}: {list(frame.columns)}")
    if frame[config.FEATURES].isna().any().any() or not np.isfinite(frame[config.FEATURES].to_numpy(float)).all():
        raise ValueError(f"Invalid feature values in {name} {split}")
    return frame


def _remove_train_overlap(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keys = set(map(tuple, train[config.FEATURES].to_numpy()))
    keep = [tuple(row) not in keys for row in test[config.FEATURES].to_numpy()]
    return test.loc[keep].reset_index(drop=True), len(test) - sum(keep)


def finalize_dataset(name: str, force: bool = False) -> dict[str, int]:
    """Create raw/log/scaled files from recovered aligned model-space values."""
    config.SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    train = _read_split(name, "train")
    test, removed = _remove_train_overlap(train, _read_split(name, "test"))
    train = train.assign(split="train")
    test = test.assign(split="test")
    combined = pd.concat([train, test], ignore_index=True)
    for representation in ("raw", "log", "scaled"):
        output = config.finalized_path(name, representation)
        if force or not output.exists():
            combined.to_csv(output, index=False)
    scaler_path = config.SCALERS_DIR / f"{config.dataset_key(name)}_scaler.joblib"
    if force or not scaler_path.exists():
        joblib.dump({"scaler": None, "log_features": [], "recovered_aligned": True}, scaler_path)
    return {"train_rows": len(train), "test_rows": len(test), "removed_overlap": removed}


def inverse_transform_to_raw(values: np.ndarray, dataset: str) -> np.ndarray:
    """Return recovered model-space values because original raw units are unavailable."""
    artifact = joblib.load(config.SCALERS_DIR / f"{config.dataset_key(dataset)}_scaler.joblib")
    if artifact.get("recovered_aligned"):
        return np.asarray(values, dtype=float)
    return artifact["scaler"].inverse_transform(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cicids", "unsw", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    names = config.DATASETS if args.dataset == "all" else (args.dataset,)
    for name in names:
        LOGGER.info("%s: %s", name, finalize_dataset(name, force=args.force))


if __name__ == "__main__":
    main()