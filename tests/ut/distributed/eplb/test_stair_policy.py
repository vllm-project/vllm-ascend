import numpy as np

from vllm_ascend.distributed.eplb.stair_policy import (
    align_slots,
    assign_sources,
    capped_min_max,
    compress_samples,
    constrained_lpt,
    placement_score,
    passes_hysteresis,
    plan_rebalance,
    replica_candidates,
    weighted_moments,
)


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


def test_capped_min_max_respects_one_copy_per_rank():
    replicas = capped_min_max(np.array([8.0, 3.0]), np.ones(2, dtype=np.int64), 4, 3)

    np.testing.assert_array_equal(replicas, [3, 3])


def test_flash_tree_candidates_are_bounded_and_deterministic():
    kwargs = dict(depth=3, width=2, limit=4, score=lambda value: float(np.square(value - 2).sum()))

    first = replica_candidates(np.array([8.0, 4.0, 2.0]), 6, 2, **kwargs)
    second = replica_candidates(np.array([8.0, 4.0, 2.0]), 6, 2, **kwargs)

    assert len(first) <= 4
    assert [item.tolist() for item in first] == [item.tolist() for item in second]
    assert all(item.sum() == 6 and np.all(item <= 2) for item in first)


def test_source_assignment_enforces_pair_cap():
    old = np.array([[0, 1], [2, 3]])

    assert assign_sources(old, [{2, 3}, {0, 1}], (0, 0), 1) is None


def test_source_assignment_prefers_same_node_and_aligns_slots():
    old = np.array([[0, 1], [0, 2], [3, 4]])
    desired = [{0, 1}, {0, 2}, {0, 4}]

    sources = assign_sources(old, desired, (0, 1, 1), 1)
    assert sources == {(2, 0): (1, 0)}
    placement, source_rank, source_slot = align_slots(old, desired, sources)

    np.testing.assert_array_equal(placement, [[0, 1], [0, 2], [0, 4]])
    assert (source_rank[2, 0], source_slot[2, 0]) == (1, 0)


def test_constrained_lpt_obeys_placement_and_pair_invariants():
    old = np.array([[0, 1], [2, 3], [0, 1]])
    mean = np.array([12.0, 8.0, 3.0, 1.0])
    variance = np.zeros(4)

    result = constrained_lpt(
        mean,
        variance,
        np.array([2, 2, 1, 1]),
        old,
        (0, 0, 1),
        z_score=0.0,
        pair_cap=1,
        max_backtracks=20,
    )

    assert result is not None
    placement, source_rank, _, _, _ = result
    np.testing.assert_array_equal(np.bincount(placement.ravel()), [2, 2, 1, 1])
    assert all(len(set(row)) == len(row) for row in placement.tolist())
    pairs = [(int(src), dst) for dst, row in enumerate(source_rank) for src in row if src != dst]
    assert len(pairs) == len(set(pairs))


def test_hysteresis_uses_last_committed_score():
    from vllm_ascend.ascend_config import StairConfig

    config = StairConfig(hysteresis_relative=0.9, hysteresis_absolute=0.8)

    assert not passes_hysteresis(1.05, 1.0, config)
    assert passes_hysteresis(1.2, 1.0, config)


def test_plan_rebalance_filters_zero_and_balanced_layers():
    from vllm_ascend.ascend_config import StairConfig

    load = np.array([[[0, 0]], [[0, 0]]])
    old = np.array([[[0], [1]]])

    plan = plan_rebalance(load, old, np.array([np.nan]), (0, 0), StairConfig())

    np.testing.assert_array_equal(plan.placement, old)
    assert np.isnan(plan.accepted_scores[0])
