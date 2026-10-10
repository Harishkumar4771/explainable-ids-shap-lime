"""Leakage-aware loading for finalized IDS representations."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

LOGGER = logging.getLogger(__name__)


def load_scaler(name: str) -> dict:
	return joblib.load(config.SCALERS_DIR / f"{config.dataset_key(name)}_scaler.joblib")


def load_dataset(name: str, split: str, representation: str = "scaled", scaling_mode: str = config.SCALING_MODE) -> tuple[pd.DataFrame, np.ndarray]:
	if scaling_mode not in config.ALLOWED_SCALING_MODES:
		raise ValueError(f"Unsupported scaling mode: {scaling_mode}")
	frame = pd.read_csv(config.finalized_path(name, representation))
	subset = frame.loc[frame["split"] == split]
	return subset[config.FEATURES].reset_index(drop=True), subset[config.LABEL].to_numpy(dtype=int)


def get_train_val(name: str, seed: int, scaling_mode: str = config.SCALING_MODE) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	features, labels = load_dataset(name, "train", scaling_mode=scaling_mode)
	return train_test_split(features.to_numpy(float), labels, test_size=config.VAL_FRACTION, random_state=seed, stratify=labels)


def get_cross_dataset_inputs(source: str, target: str, scaling_mode: str = config.SCALING_MODE) -> pd.DataFrame:
	if scaling_mode == "source":
		LOGGER.warning("Source scaling requested but only recovered aligned inputs exist")
	return load_dataset(target, "test", scaling_mode=scaling_mode)[0]


def print_split_summary() -> None:
	for name in config.DATASETS:
		for split in ("train", "test"):
			_, labels = load_dataset(name, split)
			attacks = int(labels.sum())
			LOGGER.info("%s %s rows=%d attacks=%d prevalence=%.6f", name, split, len(labels), attacks, attacks / len(labels))
			if attacks < config.SMALL_POSITIVE_WARNING:
				LOGGER.warning("%s %s has fewer than %d positives", name, split, config.SMALL_POSITIVE_WARNING)
