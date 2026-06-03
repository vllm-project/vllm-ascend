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
