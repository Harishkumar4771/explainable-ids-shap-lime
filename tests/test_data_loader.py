import numpy as np

from src import config
from src.data_loader import get_train_val, load_dataset
from src.preprocessing import inverse_transform_to_raw


def test_feature_order_and_binary_labels():
    for dataset in config.DATASETS:
        _, labels = load_dataset(dataset, "train")
        assert set(labels) <= {0, 1}
        assert list(load_dataset(dataset, "test")[0].columns) == config.FEATURES


def test_no_train_test_overlap_after_finalization():
    for dataset in config.DATASETS:
        train, _ = load_dataset(dataset, "train")
        test, _ = load_dataset(dataset, "test")
        train_keys = set(map(tuple, train.to_numpy()))
        assert not train_keys.intersection(map(tuple, test.to_numpy()))


def test_round_trip_and_validation_disjoint_from_test():
    for dataset in config.DATASETS:
        train, _ = load_dataset(dataset, "train")
        recovered = inverse_transform_to_raw(train.to_numpy()[:10], dataset)
        np.testing.assert_allclose(recovered, train.to_numpy()[:10], atol=1e-6)
        x_train, x_val, y_train, y_val = get_train_val(dataset, seed=42)
        assert len(x_train) + len(x_val) == len(train)
        assert len(y_train) + len(y_val) == len(train)