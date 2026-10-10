"""Model zoo and prediction adapter for binary intrusion detection."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras

from . import config


def build_mlp_a(input_dim: int = len(config.FEATURES)) -> keras.Model:
	inputs = keras.Input(shape=(input_dim,))
	x = keras.layers.Dense(64, activation="relu")(inputs)
	x = keras.layers.Dense(32, activation="relu")(x)
	x = keras.layers.Dense(16, activation="relu")(x)
	outputs = keras.layers.Dense(1, activation="sigmoid")(x)
	return keras.Model(inputs, outputs, name="mlp_a")


def build_mlp_b(input_dim: int = len(config.FEATURES)) -> keras.Model:
	regularizer = keras.regularizers.l2(1e-4)
	inputs = keras.Input(shape=(input_dim,))
	x = keras.layers.GaussianNoise(0.05)(inputs)
	for units in (64, 32):
		x = keras.layers.Dense(units, kernel_regularizer=regularizer)(x)
		x = keras.layers.LayerNormalization()(x)
		x = keras.layers.Activation("relu")(x)
		x = keras.layers.Dropout(0.3)(x)
	x = keras.layers.Dense(16, activation="relu")(x)
	outputs = keras.layers.Dense(1, activation="sigmoid")(x)
	return keras.Model(inputs, outputs, name="mlp_b")


def build_mlp_c(input_dim: int = len(config.FEATURES)) -> keras.Model:
	inputs = keras.Input(shape=(input_dim,))
	x = keras.layers.Dense(64, activation="relu")(inputs)
	for _ in range(2):
		residual = x
		branch = keras.layers.LayerNormalization()(x)
		branch = keras.layers.Dense(64, activation="relu")(branch)
		branch = keras.layers.Dropout(0.2)(branch)
		branch = keras.layers.Dense(64)(branch)
		x = keras.layers.Add()([residual, branch])
	x = keras.layers.Dense(16, activation="relu")(x)
	outputs = keras.layers.Dense(1, activation="sigmoid")(x)
	return keras.Model(inputs, outputs, name="mlp_c")


def build_rf(seed: int) -> RandomForestClassifier:
	return RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1)


def build_dt(seed: int, max_depth: int = 5) -> DecisionTreeClassifier:
	return DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=seed)


class ModelAdapter:
	"""Uniform prediction and persistence interface for Keras and sklearn models."""

	def __init__(self, name: str, family: str, model: object):
		self.name = name
		self.family = family
		self.model = model

	def predict_attack_proba(self, values: np.ndarray) -> np.ndarray:
		values = np.asarray(values, dtype=float)
		if self.family == "keras":
			probabilities = self.model.predict(values, verbose=0).reshape(-1)
		else:
			probabilities = self.model.predict_proba(values)[:, 1]
		return np.clip(probabilities, 0.0, 1.0)

	def predict_proba(self, values: np.ndarray) -> np.ndarray:
		attack = self.predict_attack_proba(values)
		return np.column_stack((1.0 - attack, attack))

	def predict(self, values: np.ndarray, threshold: float = 0.5) -> np.ndarray:
		return (self.predict_attack_proba(values) >= threshold).astype(int)

	def save(self, path: str | Path) -> None:
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		if self.family == "keras":
			self.model.save(path)
		else:
			joblib.dump(self.model, path)

	@classmethod
	def load(cls, path: str | Path, name: str | None = None, family: str | None = None) -> "ModelAdapter":
		path = Path(path)
		family = family or ("keras" if path.suffix == ".keras" else "sklearn")
		model = keras.models.load_model(path) if family == "keras" else joblib.load(path)
		return cls(name or path.stem, family, model)


def build_adapter(name: str, seed: int = 42) -> ModelAdapter:
	if name == "rf":
		return ModelAdapter(name, "sklearn", build_rf(seed))
	if name == "dt":
		return ModelAdapter(name, "sklearn", build_dt(seed))
	if name in {"mlp_a", "mlp_b", "mlp_c"}:
		return ModelAdapter(name, "keras", globals()[f"build_{name}"]())
	raise ValueError(f"Unknown model: {name}")


MODEL_ZOO = {name: build_adapter for name in ("mlp_a", "mlp_b", "mlp_c", "rf", "dt")}


def print_parameter_counts() -> dict[str, int]:
	counts = {}
	for name in MODEL_ZOO:
		adapter = build_adapter(name)
		counts[name] = int(adapter.model.count_params()) if adapter.family == "keras" else int(sum(np.prod(v.shape) for v in adapter.model.__dict__.get("estimators_", [])))
	return counts
