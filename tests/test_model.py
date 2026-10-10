import numpy as np

from src.model import ModelAdapter, build_adapter, build_mlp_c


def test_models_predict_shapes_and_probability_rows():
    values = np.zeros((4, 9), dtype=float)
    for name in ("mlp_a", "mlp_b", "mlp_c"):
        adapter = build_adapter(name)
        probabilities = adapter.predict_attack_proba(values)
        assert probabilities.shape == (4,)
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        np.testing.assert_allclose(adapter.predict_proba(values).sum(axis=1), 1.0)


def test_mlp_c_has_two_residual_add_layers():
    model = build_mlp_c()
    assert sum(layer.__class__.__name__ == "Add" for layer in model.layers) == 2


def test_sklearn_adapter_save_load(tmp_path):
    values = np.zeros((4, 9), dtype=float)
    adapter = build_adapter("dt", seed=42)
    adapter.model.fit(values, np.array([0, 1, 0, 1]))
    path = tmp_path / "dt.joblib"
    adapter.save(path)
    loaded = ModelAdapter.load(path, name="dt")
    np.testing.assert_allclose(adapter.predict_attack_proba(values), loaded.predict_attack_proba(values))