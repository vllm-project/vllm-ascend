import numpy as np

from vllm_ascend.distributed.eplb.stair_policy import compress_samples, placement_score, weighted_moments


def test_compression_preserves_every_step_as_weighted_bins():
    samples = np.arange(20).reshape(5, 2, 2)

    compressed, weights = compress_samples(samples, 2)

    np.testing.assert_array_equal(weights, [2, 3])
    np.testing.assert_allclose(compressed[0], samples[:2].mean(axis=0))
    np.testing.assert_allclose(compressed[1], samples[2:].mean(axis=0))


def test_weighted_moments_match_uncompressed_samples():
    samples = np.array([[1.0, 4.0], [3.0, 2.0]])
    weights = np.array([2, 1])
    expanded = np.repeat(samples, weights, axis=0)

    mean, covariance = weighted_moments(samples, weights, covariance=True)

    np.testing.assert_allclose(mean, expanded.mean(axis=0))
    np.testing.assert_allclose(covariance, np.cov(expanded, rowvar=False))


def test_score_uses_mean_and_weighted_nearest_rank_p95():
    samples = np.array([[8.0, 0.0], [4.0, 4.0]])
    weights = np.array([1, 19])
    placement = np.array([[0], [1]])

    score = placement_score(samples, weights, placement)

    assert score.mean == 1.05
    assert score.p95 == 1.0
