import numpy as np
import pytest

from anomalias.data import FEATURES, MEANS, STDS, generate_dataset


def test_shapes_and_labels():
    X, y = generate_dataset(n_samples=1000, contamination=0.05, seed=1)
    assert X.shape == (1000, len(FEATURES))
    assert y.shape == (1000,)
    assert set(np.unique(y)) == {0, 1}
    assert y.sum() == 50


def test_reproducible_with_same_seed():
    X1, y1 = generate_dataset(500, 0.1, seed=7)
    X2, y2 = generate_dataset(500, 0.1, seed=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_differ():
    X1, _ = generate_dataset(500, 0.1, seed=1)
    X2, _ = generate_dataset(500, 0.1, seed=2)
    assert not np.allclose(X1, X2)


def test_normal_points_follow_reference_distribution():
    X, y = generate_dataset(5000, 0.05, seed=3)
    normal = X[y == 0]
    np.testing.assert_allclose(normal.mean(axis=0), MEANS, atol=0.1 * STDS.max())
    np.testing.assert_allclose(normal.std(axis=0), STDS, rtol=0.1)


def test_anomalies_are_far_from_normal_points():
    X, y = generate_dataset(5000, 0.05, seed=4)
    z = np.abs((X - MEANS) / STDS).max(axis=1)
    assert np.median(z[y == 1]) > 3 * np.median(z[y == 0])


def test_zero_contamination():
    _, y = generate_dataset(100, 0.0, seed=0)
    assert y.sum() == 0


@pytest.mark.parametrize("n, c", [(0, 0.05), (100, -0.1), (100, 0.5)])
def test_invalid_args(n, c):
    with pytest.raises(ValueError):
        generate_dataset(n, c, seed=0)
