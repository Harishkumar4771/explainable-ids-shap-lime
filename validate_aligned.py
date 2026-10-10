"""Validate finalized aligned datasets."""

import argparse
import logging

from src import config
from src.data_loader import load_dataset, print_split_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print_split_summary()
    for dataset in config.DATASETS:
        for split in ("train", "test"):
            features, labels = load_dataset(dataset, split)
            assert list(features.columns) == config.FEATURES
            assert set(labels) <= {0, 1}
            assert features.notna().all().all()
            print(f"{dataset},{split},rows={len(labels)},features={features.shape[1]},attacks={int(labels.sum())}")


if __name__ == "__main__":
    main()