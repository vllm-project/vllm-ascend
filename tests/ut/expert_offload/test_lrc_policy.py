from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy


def test_choose_victim_keeps_recently_hot_expert():
    policy = LRCExpertCachePolicy(
        num_layers=1,
        num_experts=8,
        cache_size=4,
        topk=2,
        recent_window=4,
        ema_beta=0.5,
        age_weight=0.0,
    )
    layer_idx = 0

    for _ in range(4):
        policy.observe(layer_idx, [[1, 2]])

    slot_owner = {0: 1, 1: 2, 2: 3, 3: 4}

    assert policy.choose_victim(layer_idx, slot_owner, protected={5, 6}) == 3


def test_choose_victim_never_evicts_current_topk():
    policy = LRCExpertCachePolicy(
        num_layers=1,
        num_experts=8,
        cache_size=4,
        topk=2,
        recent_window=4,
        ema_beta=0.5,
        age_weight=0.0,
    )
    layer_idx = 0

    policy.observe(layer_idx, [[3, 4]])
    slot_owner = {0: 1, 1: 2, 2: 3, 3: 4}

    assert policy.choose_victim(layer_idx, slot_owner, protected={3, 4}) == 1


def test_observe_maintains_recent_window_frequency():
    policy = LRCExpertCachePolicy(
        num_layers=1,
        num_experts=8,
        cache_size=4,
        topk=2,
        recent_window=2,
        ema_beta=0.5,
        age_weight=0.0,
    )
    layer_idx = 0

    policy.observe(layer_idx, [[1, 2]])
    policy.observe(layer_idx, [[2, 3]])
    policy.observe(layer_idx, [[3, 4]])

    state = policy.layer_states[layer_idx]
    assert state.freq[1] == 0
    assert state.freq[2] == 1
    assert state.freq[3] == 2
    assert state.freq[4] == 1


def test_router_score_contributes_to_hotness():
    policy = LRCExpertCachePolicy(
        num_layers=1,
        num_experts=8,
        cache_size=4,
        topk=2,
        recent_window=4,
        ema_beta=0.5,
        recent_weight=0.0,
        ema_weight=0.0,
        router_weight=1.0,
        age_weight=0.0,
    )
    layer_idx = 0

    policy.observe(layer_idx, [[1, 2]], router_scores=[[0.1, 0.9]])
    slot_owner = {0: 1, 1: 2, 2: 3, 3: 4}

    assert policy.choose_victim(layer_idx, slot_owner, protected={5, 6}) == 3


def test_layer_steps_are_tracked_independently():
    policy = LRCExpertCachePolicy(
        num_layers=2,
        num_experts=8,
        cache_size=4,
        topk=2,
    )

    policy.observe(0, [[1, 2], [2, 3]])
    policy.observe(1, [[4, 5]])

    assert policy.layer_step(0) == 2
    assert policy.layer_step(1) == 1


def test_ema_matches_reference_formula():
    """Vectorized EMA must match the per-expert float64 reference within f32.

    Rows ``[2, 3, 5, 5]`` and ``[1, 1, 1, 1]`` carry duplicate ids, so this also
    locks in that a hit never contributes more than (1-beta) per step — numpy
    indexed ``+=`` does not accumulate duplicate indices, matching the original
    ``set()``-based hit semantics.
    """
    num_experts = 16
    beta = 0.9
    policy = LRCExpertCachePolicy(
        num_layers=1,
        num_experts=num_experts,
        cache_size=4,
        topk=4,
        recent_window=8,
        ema_beta=beta,
    )
    rows = [[1, 2, 3, 4], [2, 3, 5, 5], [0, 1, 2, 9], [1, 1, 1, 1]]
    policy.observe(0, rows)

    ref = [0.0] * num_experts
    for row in rows:
        hit = set(row)
        for eid in range(num_experts):
            ref[eid] = beta * ref[eid] + (1.0 - beta) * (1.0 if eid in hit else 0.0)

    ema = policy.layer_states[0].ema
    for eid in range(num_experts):
        assert abs(float(ema[eid]) - ref[eid]) < 1e-6, (eid, ema[eid], ref[eid])


def test_hotness_returns_python_float():
    """np.float64 subclasses ``float``; guard the numpy-scalar return type."""
    policy = LRCExpertCachePolicy(
        num_layers=1, num_experts=8, cache_size=4, topk=2)
    policy.observe(0, [[1, 2]])
    assert isinstance(policy.hotness(0, 1), float)


def test_choose_victim_loading_excludes_and_falls_back():
    policy = LRCExpertCachePolicy(
        num_layers=1,
        num_experts=8,
        cache_size=4,
        topk=2,
        recent_window=4,
        ema_beta=0.5,
        age_weight=0.0,
    )
    for _ in range(4):
        policy.observe(0, [[1, 2]])
    slot_owner = {0: 1, 1: 2, 2: 3, 3: 4}

    # Baseline: coldest resident is expert 3.
    assert policy.choose_victim(0, slot_owner, protected={5, 6}) == 3
    # Loading expert 3 forces eviction of the next-coldest resident (4).
    assert policy.choose_victim(0, slot_owner, protected={5, 6}, loading={3}) == 4
    # Every unprotected resident loading -> fall back to the protected-only
    # filter (loading ignored), so the baseline victim 3 is chosen again.
    assert policy.choose_victim(
        0, slot_owner, protected={5, 6}, loading={1, 2, 3, 4}) == 3
