"""Tests for the pure transfer port/topology helpers of the pull connector.

Expected values are hardcoded / rewritten by hand from the legacy formulas; the
helpers under test are never used to build the expectations.
"""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    context_parallel_parameters_check,
    cp_group_meta,
    group_kernel_params,
    kernel_block_scale,
    kv_head_groups,
    local_remote_block_port_mappings,
    local_remote_cp_params,
    remote_host_info_by_port,
    remote_port_send_num,
    set_hma_shared_port,
)


@pytest.mark.parametrize(
    "num_key_value_heads,use_mla,use_sparse,tp_size,expected",
    [
        (8, True, False, 4, [(0,)]),
        (8, False, True, 4, [(0,)]),
        (8, False, False, 2, [(0, 1, 2, 3), (4, 5, 6, 7)]),
        (2, False, False, 4, [(0,), (1,)]),
        (4, False, False, 4, [(0,), (1,), (2,), (3,)]),
        (1, False, False, 8, [(0,)]),
        # Integer division drops remainder heads in the legacy implementation.
        (5, False, False, 2, [(0, 1), (2, 3)]),
    ],
)
def test_kv_head_groups(num_key_value_heads, use_mla, use_sparse, tp_size, expected):
    groups = kv_head_groups(
        num_key_value_heads=num_key_value_heads,
        use_mla=use_mla,
        use_sparse=use_sparse,
        tp_size=tp_size,
    )
    assert groups == expected


def test_kv_head_groups_returns_none_for_uncovered_topology():
    # 2 heads over 3 TP ranks matches neither legacy branch, so the legacy
    # implementation falls through and implicitly returns None. Preserved.
    assert kv_head_groups(num_key_value_heads=2, use_mla=False, use_sparse=False, tp_size=3) is None


def test_kv_head_groups_is_keyword_only():
    with pytest.raises(TypeError):
        kv_head_groups(8, False, False, 2)  # type: ignore[misc]


@pytest.mark.parametrize(
    "remote_cp_size,local_cp_size",
    [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2), (2, 4)],
)
def test_context_parallel_parameters_check_accepts_divisible_cp(remote_cp_size, local_cp_size):
    context_parallel_parameters_check(
        remote_cp_size=remote_cp_size,
        local_cp_size=local_cp_size,
        use_mla=True,
        use_sparse=False,
        num_key_value_heads=8,
        prefill_tp_size=2,
        decode_tp_size=4,
    )


def test_context_parallel_parameters_check_rejects_indivisible_cp():
    with pytest.raises(AssertionError) as excinfo:
        context_parallel_parameters_check(
            remote_cp_size=3,
            local_cp_size=2,
            use_mla=False,
            use_sparse=False,
            num_key_value_heads=8,
            prefill_tp_size=2,
            decode_tp_size=2,
        )
    assert str(excinfo.value) == "P/D CP sizes must be divisible in either direction, got P=3, D=2."


@pytest.mark.parametrize(
    "use_mla,use_sparse",
    [(False, False)],
)
def test_context_parallel_parameters_check_rejects_incompatible_head_partition(use_mla, use_sparse):
    # p heads/rank = ceil(8/2) = 4, d heads/rank = ceil(8/4) = 2, 2 % 4 != 0.
    with pytest.raises(AssertionError):
        context_parallel_parameters_check(
            remote_cp_size=2,
            local_cp_size=2,
            use_mla=use_mla,
            use_sparse=use_sparse,
            num_key_value_heads=8,
            prefill_tp_size=2,
            decode_tp_size=4,
        )


@pytest.mark.parametrize("use_mla,use_sparse", [(True, False), (False, True)])
def test_context_parallel_parameters_check_skips_head_check_for_mla_or_sparse(use_mla, use_sparse):
    context_parallel_parameters_check(
        remote_cp_size=2,
        local_cp_size=2,
        use_mla=use_mla,
        use_sparse=use_sparse,
        num_key_value_heads=8,
        prefill_tp_size=2,
        decode_tp_size=4,
    )


def test_cp_group_meta_mla_single_pcp_single_dcp():
    assert cp_group_meta(
        num_key_value_heads=8,
        use_mla=True,
        use_sparse=False,
        tp_size=2,
        pcp_size=1,
        dcp_size=1,
        port_base=30000,
    ) == {(0,): {"cp_groups": [[30000], [30001]], "select_cp_groups_id": 0}}


def test_cp_group_meta_mla_dcp_two():
    assert cp_group_meta(
        num_key_value_heads=8,
        use_mla=True,
        use_sparse=False,
        tp_size=4,
        pcp_size=1,
        dcp_size=2,
        port_base=30000,
    ) == {(0,): {"cp_groups": [[30000, 30001], [30002, 30003]], "select_cp_groups_id": 0}}


def test_cp_group_meta_non_mla_one_head_per_rank():
    assert cp_group_meta(
        num_key_value_heads=4,
        use_mla=False,
        use_sparse=False,
        tp_size=4,
        pcp_size=1,
        dcp_size=1,
        port_base=30000,
    ) == {
        (0,): {"cp_groups": [[30000]], "select_cp_groups_id": 0},
        (1,): {"cp_groups": [[30001]], "select_cp_groups_id": 0},
        (2,): {"cp_groups": [[30002]], "select_cp_groups_id": 0},
        (3,): {"cp_groups": [[30003]], "select_cp_groups_id": 0},
    }


def test_cp_group_meta_non_mla_two_pcp_ranks():
    assert cp_group_meta(
        num_key_value_heads=4,
        use_mla=False,
        use_sparse=False,
        tp_size=4,
        pcp_size=2,
        dcp_size=1,
        port_base=30000,
    ) == {
        (0,): {"cp_groups": [[30000], [30004]], "select_cp_groups_id": 0},
        (1,): {"cp_groups": [[30001], [30005]], "select_cp_groups_id": 0},
        (2,): {"cp_groups": [[30002], [30006]], "select_cp_groups_id": 0},
        (3,): {"cp_groups": [[30003], [30007]], "select_cp_groups_id": 0},
    }


def test_cp_group_meta_non_mla_two_head_groups():
    assert cp_group_meta(
        num_key_value_heads=2,
        use_mla=False,
        use_sparse=False,
        tp_size=4,
        pcp_size=1,
        dcp_size=1,
        port_base=31000,
    ) == {
        (0,): {"cp_groups": [[31000], [31001]], "select_cp_groups_id": 0},
        (1,): {"cp_groups": [[31002], [31003]], "select_cp_groups_id": 0},
    }


@pytest.mark.parametrize(
    "num_key_value_heads,use_mla,use_sparse,tp_size,pcp_size,dcp_size,port_base",
    [
        (8, True, False, 2, 1, 1, 30000),
        (8, True, False, 4, 1, 2, 30000),
        (8, True, False, 4, 2, 2, 32000),
        (4, False, False, 4, 1, 1, 30000),
        (4, False, False, 4, 2, 1, 30000),
        (2, False, False, 4, 1, 1, 31000),
    ],
)
def test_cp_group_meta_port_and_group_properties(
    num_key_value_heads, use_mla, use_sparse, tp_size, pcp_size, dcp_size, port_base
):
    result = cp_group_meta(
        num_key_value_heads=num_key_value_heads,
        use_mla=use_mla,
        use_sparse=use_sparse,
        tp_size=tp_size,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
        port_base=port_base,
    )
    all_ports = []
    for head_group, meta in result.items():
        assert meta["select_cp_groups_id"] == 0
        assert len(head_group) >= 1
        for cp_group in meta["cp_groups"]:
            assert len(cp_group) == dcp_size
            all_ports.extend(cp_group)
    assert all(port_base <= port < port_base + tp_size * pcp_size for port in all_ports)


# --------------------------------------------------- local_remote_block_port_mappings


def _port_mappings(**overrides):
    params = {
        "num_key_value_heads": 8,
        "use_mla": True,
        "use_sparse": False,
        "prefill_tp_size": 2,
        "remote_cp_size": 1,
        "local_cp_size": 1,
        "remote_pcp_size": 1,
        "remote_dcp_size": 1,
        "remote_port": 30000,
        "decode_tp_size": 2,
        "decode_pcp_size": 1,
        "decode_dcp_size": 1,
        "decode_side_channel_port": 31000,
        "r_blk": 1,
    }
    params.update(overrides)
    return local_remote_block_port_mappings(**params)


def test_local_remote_block_port_mappings_mla_single_cp():
    p_meta, d_meta, mappings = _port_mappings()
    assert p_meta == {(0,): {"cp_groups": [[30000], [30001]], "select_cp_groups_id": 0}}
    assert d_meta == {(0,): {"cp_groups": [[31000], [31001]], "select_cp_groups_id": 0}}
    assert mappings == {31000: [[30000]], 31001: [[30001]]}


def test_local_remote_block_port_mappings_mla_dcp_two():
    p_meta, d_meta, mappings = _port_mappings(
        prefill_tp_size=4,
        remote_dcp_size=2,
        decode_tp_size=4,
        decode_dcp_size=2,
    )
    assert p_meta == {(0,): {"cp_groups": [[30000, 30001], [30002, 30003]], "select_cp_groups_id": 0}}
    assert d_meta == {(0,): {"cp_groups": [[31000, 31001], [31002, 31003]], "select_cp_groups_id": 0}}
    assert mappings == {31000: [[30000]], 31001: [[30001]], 31002: [[30002]], 31003: [[30003]]}


def test_local_remote_block_port_mappings_non_mla_one_head_per_rank():
    p_meta, d_meta, mappings = _port_mappings(
        num_key_value_heads=4,
        use_mla=False,
        prefill_tp_size=4,
        decode_tp_size=4,
    )
    assert p_meta == {
        (0,): {"cp_groups": [[30000]], "select_cp_groups_id": 0},
        (1,): {"cp_groups": [[30001]], "select_cp_groups_id": 0},
        (2,): {"cp_groups": [[30002]], "select_cp_groups_id": 0},
        (3,): {"cp_groups": [[30003]], "select_cp_groups_id": 0},
    }
    assert d_meta == {
        (0,): {"cp_groups": [[31000]], "select_cp_groups_id": 0},
        (1,): {"cp_groups": [[31001]], "select_cp_groups_id": 0},
        (2,): {"cp_groups": [[31002]], "select_cp_groups_id": 0},
        (3,): {"cp_groups": [[31003]], "select_cp_groups_id": 0},
    }
    assert mappings == {31000: [[30000]], 31001: [[30001]], 31002: [[30002]], 31003: [[30003]]}


def test_local_remote_block_port_mappings_block_size_ratio_two():
    # r_blk = 2: P CP ranks q and q+1 of a group map to D rank (q // 2), so only
    # even D CP ranks get P ports and odd ones get an empty P list.
    _, _, mappings = _port_mappings(prefill_tp_size=4, remote_dcp_size=2, decode_tp_size=4, decode_dcp_size=2, r_blk=2)
    assert mappings == {31000: [[30000, 30001]], 31001: [[]], 31002: [[30002, 30003]], 31003: [[]]}


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"prefill_tp_size": 4, "remote_dcp_size": 2, "decode_tp_size": 4, "decode_dcp_size": 2},
        {"prefill_tp_size": 4, "remote_dcp_size": 2, "decode_tp_size": 4, "decode_dcp_size": 2, "r_blk": 2},
        {"num_key_value_heads": 4, "use_mla": False, "prefill_tp_size": 4, "decode_tp_size": 4},
        {"num_key_value_heads": 4, "use_mla": False, "prefill_tp_size": 4, "decode_tp_size": 2},
    ],
)
def test_local_remote_block_port_mappings_structure(overrides):
    p_meta, d_meta, mappings = _port_mappings(**overrides)
    # P and D head groups may be partitioned differently (one head per rank on
    # one side, merged head groups on the other); the implementation matches
    # each P group against the D group that contains it.
    assert all(any(set(pk).issubset(set(dk)) for dk in d_meta) for pk in p_meta)
    for d_port, p_port_lists in mappings.items():
        assert isinstance(d_port, int)
        assert isinstance(p_port_lists, list)
        for p_ports in p_port_lists:
            assert isinstance(p_ports, list)
            for port in p_ports:
                assert 30000 <= port < 30000 + overrides.get("prefill_tp_size", 2) * 1
    for d_port in mappings:
        assert 31000 <= d_port < 31000 + overrides.get("decode_tp_size", 2) * 1


def test_local_remote_block_port_mappings_cross_partition_head_groups():
    # P side keeps one head per TP rank ((0,), (1,), (2,), (3,)) while D side
    # merges two heads per rank ((0, 1), (2, 3)). Subset matching pairs each P
    # head group with its D group, so each D port receives one P port list per
    # matched head group.
    p_meta, d_meta, mappings = _port_mappings(num_key_value_heads=4, use_mla=False, prefill_tp_size=4, decode_tp_size=2)
    assert set(p_meta) == {(0,), (1,), (2,), (3,)}
    assert set(d_meta) == {(0, 1), (2, 3)}
    assert mappings == {31000: [[30000], [30001]], 31001: [[30002], [30003]]}


# ---------------------------------------------------------------- set_hma_shared_port


def _hma(**overrides):
    params = {
        "is_hma_required": True,
        "use_mla": False,
        "use_sparse": False,
        "num_key_value_heads": 1,
        "prefill_tp_size": 4,
        "remote_port": 30000,
        "remote_dcp_size": 2,
        "decode_tp_size": 2,
        "decode_tp_rank": 0,
        "chosen_tp_list": [2, 3],
        "remote_handshake_port_list": [[30000, 30001]],
    }
    params.update(overrides)
    return params


def test_set_hma_shared_port_noop_when_not_hma():
    port_list = [[30000, 30001]]
    result = set_hma_shared_port(**_hma(is_hma_required=False, remote_handshake_port_list=port_list))
    assert result is port_list
    assert result == [[30000, 30001]]


@pytest.mark.parametrize("use_mla,use_sparse", [(True, False), (False, True)])
def test_set_hma_shared_port_skips_mla_or_sparse(use_mla, use_sparse):
    port_list = [[30000, 30001]]
    result = set_hma_shared_port(**_hma(use_mla=use_mla, use_sparse=use_sparse, remote_handshake_port_list=port_list))
    assert result is port_list
    assert result == [[30000, 30001]]


def test_set_hma_shared_port_replaces_replicas_and_appends_mamba_ports():
    port_list = [[30000, 30001]]
    result = set_hma_shared_port(**_hma(remote_handshake_port_list=port_list))
    # Replica swap: chosen TP ranks 2/3 pick replica 1 of the dcp repeat, so both
    # attention ports move by +2. Mamba ports for decode TP rank 0 (k = 2) are
    # appended to the final shard.
    assert result is port_list
    assert result == [[30002, 30003, 30000, 30001]]


def test_set_hma_shared_port_single_replica_only_appends_mamba_ports():
    # num_heads == prefill_tp => one head per prefill rank => group_span = 1 and
    # n_replica = 1, so attention ports are untouched; Mamba ports are appended.
    port_list = [[30000, 30001]]
    result = set_hma_shared_port(
        **_hma(
            num_key_value_heads=4,
            remote_dcp_size=1,
            decode_tp_size=2,
            decode_tp_rank=1,
            chosen_tp_list=[0],
            remote_handshake_port_list=port_list,
        )
    )
    assert result is port_list
    assert result == [[30000, 30001, 30002, 30003]]


def test_set_hma_shared_port_skips_duplicate_mamba_ports():
    # With one KV head over 4 prefill ranks group_span = 4, n_replica = 4 and the
    # chosen ranks keep the original replica, so attention ports stay; the Mamba
    # ports already exist in the final shard and must not be appended twice.
    port_list = [[30000, 30001]]
    result = set_hma_shared_port(**_hma(remote_dcp_size=1, chosen_tp_list=[0, 1], remote_handshake_port_list=port_list))
    assert result is port_list
    assert result == [[30000, 30001]]


# ------------------------------------------------------------ remote host/port


@pytest.mark.parametrize(
    "mapping,expected",
    [
        (None, ("host-0", "engine-0")),
        ({"20000": {"host": "host-kv", "engine_id": "engine-kv"}}, ("host-kv", "engine-kv")),
        ({"0": {"host": "host-base"}}, ("host-base", "engine-0")),
        ({"9": {"host": "host-other"}}, ("host-0", "engine-0")),
        ({"20000": {"engine_id": "engine-kv"}}, ("host-0", "engine-kv")),
    ],
    ids=["mapping-none", "kv-port-hit", "base-port-fallback", "no-hit", "engine-only-override"],
)
def test_remote_host_info_by_port(mapping, expected):
    assert (
        remote_host_info_by_port(
            base_port=30000,
            remote_handshake_port=30000,
            remote_host="host-0",
            remote_engine_id="engine-0",
            remote_multi_nodes_meta_mapping=mapping,
            kv_port=10000,
        )
        == expected
    )


def _port_mappings_empty():
    return {}


def test_remote_port_send_num_base_range():
    result = remote_port_send_num(
        remote_port=30000,
        remote_pcp_size=1,
        prefill_tp_size=2,
        kv_port=10000,
        remote_multi_nodes_meta_mapping={},
        remote_host="host-0",
        remote_engine_id="engine-0",
        local_remote_block_port_mappings=_port_mappings_empty(),
    )
    assert result == {
        30000: {"num": 0, "host": "host-0"},
        30001: {"num": 0, "host": "host-0"},
    }


def test_remote_port_send_num_counts_duplicates():
    result = remote_port_send_num(
        remote_port=30000,
        remote_pcp_size=1,
        prefill_tp_size=2,
        kv_port=10000,
        remote_multi_nodes_meta_mapping={},
        remote_host="host-0",
        remote_engine_id="engine-0",
        local_remote_block_port_mappings={30000: [[30000], [30001]], 30001: [[30001]]},
    )
    assert result == {
        30000: {"num": 1, "host": "host-0"},
        30001: {"num": 2, "host": "host-0"},
    }


def test_remote_port_send_num_adds_multi_node_ports():
    result = remote_port_send_num(
        remote_port=30000,
        remote_pcp_size=1,
        prefill_tp_size=1,
        kv_port=10000,
        remote_multi_nodes_meta_mapping={"1": {"handshake_port": 30002}, "2": {}},
        remote_host="host-0",
        remote_engine_id="engine-0",
        local_remote_block_port_mappings={},
    )
    assert result == {
        30000: {"num": 0, "host": "host-0"},
        30002: {"num": 0, "host": "host-0"},
        10002: {"num": 0, "host": "host-0"},
    }


def test_remote_port_send_num_resolves_multi_node_host():
    result = remote_port_send_num(
        remote_port=30000,
        remote_pcp_size=1,
        prefill_tp_size=2,
        kv_port=10000,
        remote_multi_nodes_meta_mapping={"20000": {"handshake_port": 30000, "host": "host-9"}},
        remote_host="host-0",
        remote_engine_id="engine-0",
        local_remote_block_port_mappings={},
    )
    assert result[30000] == {"num": 0, "host": "host-9"}
    assert result[30001] == {"num": 0, "host": "host-0"}


# ---------------------------------------------------------------- cp geometry


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            dict(remote_block_size=16, block_size=16, dcp_rank=1, dcp_size=2, remote_dcp_size=2),
            (16, 1, 2, 2, 1),
        ),
        (
            dict(remote_block_size=None, block_size=16, dcp_rank=0, dcp_size=1, remote_dcp_size=1),
            (16, 0, 1, 1, 1),
        ),
        (
            dict(remote_block_size=32, block_size=16, dcp_rank=0, dcp_size=1, remote_dcp_size=1),
            (32, 0, 1, 1, 1),
        ),
        (
            dict(remote_block_size=8, block_size=16, dcp_rank=0, dcp_size=1, remote_dcp_size=1),
            (8, 0, 1, 1, 2),
        ),
        (
            dict(remote_block_size=8, block_size=16, dcp_rank=1, dcp_size=2, remote_dcp_size=4),
            (8, 1, 2, 4, 2),
        ),
    ],
    ids=["equal", "none-uses-block-size", "p-block-larger", "d-block-larger", "d-block-larger-dcp"],
)
def test_local_remote_cp_params(kwargs, expected):
    assert local_remote_cp_params(**kwargs) == expected


def test_local_remote_cp_params_rejects_indivisible_block_sizes():
    with pytest.raises(AssertionError) as excinfo:
        local_remote_cp_params(remote_block_size=24, block_size=16, dcp_rank=0, dcp_size=1, remote_dcp_size=1)
    assert str(excinfo.value) == "Block sizes of P (24) and D (16) must be divisible by each other."


def test_local_remote_cp_params_rejects_p_block_larger_under_dcp():
    with pytest.raises(AssertionError) as excinfo:
        local_remote_cp_params(remote_block_size=32, block_size=16, dcp_rank=0, dcp_size=2, remote_dcp_size=2)
    assert str(excinfo.value) == "D node DCP not support P node block_size(32) > D block_size(16)"


def test_local_remote_cp_params_rejects_dcp_shard_misalignment():
    with pytest.raises(AssertionError) as excinfo:
        local_remote_cp_params(remote_block_size=8, block_size=16, dcp_rank=0, dcp_size=2, remote_dcp_size=2)
    assert str(excinfo.value) == "remote_cp_size(2) must be an integer multiple ofr(2) * local_cp_size(2)"


# ------------------------------------------------------------- kernel params


@pytest.mark.parametrize(
    "layer_indices,block_size_scale,expected",
    [
        ([], [[4]], 1),
        ([0], [[4]], 4),
        ([2], [[1], [2], [3]], 3),
        ([5], [[1]], 1),
        ([0], [[]], 1),
        ([1], [[4], []], 1),
    ],
)
def test_kernel_block_scale(layer_indices, block_size_scale, expected):
    assert kernel_block_scale(layer_indices=layer_indices, block_size_scale=block_size_scale) == expected


def test_group_kernel_params_skips_mamba_and_computes_scales():
    kv_group2layeridx = {
        0: ({"kv_cache_spec_type": "FullAttentionSpec"}, [0]),
        1: ({"kv_cache_spec_type": "MambaSpec"}, [1]),
        2: ({"kv_cache_spec_type": "MLAAttentionSpec"}, [2]),
    }
    result = group_kernel_params(
        kv_group2layeridx=kv_group2layeridx,
        block_size=16,
        block_size_scale=[[2], [1], []],
        remote_block_size=32,
    )
    assert result == {0: (2, 4, 8), 2: (1, 2, 16)}


def test_group_kernel_params_rejects_indivisible_remote_block_size():
    kv_group2layeridx = {0: ({"kv_cache_spec_type": "FullAttentionSpec"}, [0])}
    with pytest.raises(AssertionError) as excinfo:
        group_kernel_params(
            kv_group2layeridx=kv_group2layeridx,
            block_size=16,
            block_size_scale=[[3]],
            remote_block_size=16,
        )
    assert str(excinfo.value) == "remote_block_size(16) not divisible by kernel_size(5)"
