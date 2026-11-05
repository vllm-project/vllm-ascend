"""Tests for the pure transfer port/topology helpers of the pull connector.

Expected values are hardcoded / rewritten by hand from the legacy formulas; the
helpers under test are never used to build the expectations.
"""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.pull.transfer_plan import (
    context_parallel_parameters_check,
    cp_group_meta,
    kv_head_groups,
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
