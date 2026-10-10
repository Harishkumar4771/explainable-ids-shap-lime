"""CLI and public entry point for cached SHAP/LIME explanations."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from . import config
from .data_loader import load_dataset
from .model import ModelAdapter
from .xai.lime_explainer import explain as explain_lime
from .xai.sampling import sample_instances
from .xai.shap_explainer import explain as explain_shap

EXPERIMENTS = {"A": ("cicids", "cicids"), "B": ("unsw", "unsw"), "C": ("cicids", "unsw"), "D": ("unsw", "cicids")}
LOGGER = logging.getLogger(__name__)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=["A"])
	parser.add_argument("--models", nargs="+", default=["dt"])
	parser.add_argument("--methods", nargs="+", choices=["shap", "lime"], default=["shap", "lime"])
	parser.add_argument("--seeds", nargs="+", type=int, default=[42])
	parser.add_argument("--smoke", action="store_true")
	parser.add_argument("--force", action="store_true")
	args = parser.parse_args()
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	for experiment in args.experiments:
		source, target = EXPERIMENTS[experiment]
		instances = sample_instances(experiment, target, force=args.force)
		target_values, _ = load_dataset(target, "test")
		source_values, _ = load_dataset(source, "train")
		selected = np.asarray(instances["indices"][:10] if args.smoke else instances["indices"])
		values = target_values.iloc[selected].to_numpy(float)
		for model_name in args.models:
			for seed in args.seeds[:1] if args.smoke else args.seeds:
				root = config.XAI_CACHE_DIR / experiment / model_name / f"seed{seed}"
				path = config.MODELS_DIR / source / model_name / f"seed{seed}.joblib"
				if not path.exists():
					path = config.MODELS_DIR / source / model_name / f"seed{seed}.keras"
				if not path.exists():
					LOGGER.warning("missing model %s/%s seed=%d", source, model_name, seed)
					continue
				adapter = ModelAdapter.load(path, name=model_name)
				for method in args.methods:
					output = root / f"{method}.npz"
					if output.exists() and not args.force:
						continue
					output.parent.mkdir(parents=True, exist_ok=True)
					if method == "shap":
						attribution, explainer = explain_shap(adapter, source_values.to_numpy(float)[: min(20 if args.smoke else 100, len(source_values))], values)
						np.savez_compressed(output, attributions=attribution, indices=selected, labels=np.asarray(instances["labels"][: len(selected)]), explainer=explainer)
					else:
						attribution, scores, intercepts = explain_lime(adapter, source_values.to_numpy(float), values, seed=seed, num_samples=200 if args.smoke else 1000)
						np.savez_compressed(output, attributions=attribution, scores=scores, intercepts=intercepts, indices=selected, labels=np.asarray(instances["labels"][: len(selected)]))
					LOGGER.info("cached %s %s %s seed=%d", experiment, model_name, method, seed)


if __name__ == "__main__":
	main()
