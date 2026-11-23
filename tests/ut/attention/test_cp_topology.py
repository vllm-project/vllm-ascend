# SPDX-License-Identifier: Apache-2.0

"""Tests for the backend-agnostic CPTopology derivation and common validation."""

from types import SimpleNamespace

import pytest

from vllm_ascend.attention.context_parallel.topology import (
    CPTopology,
    DCPGroupAxis,
    resolve_cp_topology,
    validate_common_topology,
    validate_dcp_block_interleave,
)


def test_default_topology_is_disabled() -> None:
    topology = CPTopology(tp_size=2, pcp_size=1, dcp_size=1)
    assert topology.is_pcp_enabled is False
    assert topology.is_dcp_enabled is False
    assert topology.dcp_axis is DCPGroupAxis.DISABLED
    assert topology.dcp_reuses_tp_ranks is False


def test_standalone_dcp_uses_tp_axis() -> None:
    topology = CPTopology(tp_size=2, pcp_size=1, dcp_size=2)
    assert topology.is_dcp_enabled is True
    assert topology.dcp_axis is DCPGroupAxis.TP
    assert topology.dcp_reuses_tp_ranks is True


def test_stacked_dcp_equal_pcp_uses_pcp_axis() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=2)
    assert topology.dcp_axis is DCPGroupAxis.PCP
    assert topology.dcp_reuses_tp_ranks is False


def test_full_grid_dcp_equals_tp_times_pcp() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=4)
    assert topology.dcp_axis is DCPGroupAxis.TP_PCP
    assert topology.dcp_reuses_tp_ranks is True


def test_logical_dcp_rank_order_pcp_major() -> None:
    full_grid = CPTopology(tp_size=2, pcp_size=2, dcp_size=4)
    assert full_grid.pcp_face_ranks() == (0, 2)
    # The PCP coordinate varies before the TP coordinate: (pcp=0,tp=0),
    # (pcp=1,tp=0), (pcp=0,tp=1), (pcp=1,tp=1).
    assert full_grid.dcp_logical_rank_order() == (0, 2, 1, 3)
    assert CPTopology(tp_size=1, pcp_size=4, dcp_size=4).dcp_logical_rank_order() == (0, 1, 2, 3)


def test_invalid_dcp_size_rejected() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=3)
    with pytest.raises(ValueError, match="DCP=3"):
        _ = topology.dcp_axis


def test_interleave_must_divide_block_size() -> None:
    # The check runs after the backend validators, which may rewrite the
    # interleave (SFA normalises it to the cache block size), so it is a separate
    # entry point rather than part of validate_common_topology.
    cache_config = SimpleNamespace(block_size=128)
    invalid = CPTopology(tp_size=2, pcp_size=1, dcp_size=2, interleave_size=96)
    with pytest.raises(
        ValueError,
        match=r"cp_kv_cache_interleave_size\(96\) must divide the cache block_size\(128\)",
    ):
        validate_dcp_block_interleave(invalid, cache_config)

    validate_dcp_block_interleave(
        CPTopology(tp_size=2, pcp_size=1, dcp_size=2, interleave_size=64), cache_config=cache_config
    )
    # The interleave requirement only applies when decode context parallel is on.
    validate_dcp_block_interleave(
        CPTopology(tp_size=2, pcp_size=1, dcp_size=1, interleave_size=96), cache_config=cache_config
    )


def test_resolve_topology_reads_parallel_config() -> None:
    parallel_config = SimpleNamespace(
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=4,
        cp_kv_cache_interleave_size=128,
        pipeline_parallel_size=1,
    )
    topology = resolve_cp_topology(SimpleNamespace(parallel_config=parallel_config))
    assert topology == CPTopology(
        tp_size=2,
        pcp_size=2,
        dcp_size=4,
        interleave_size=128,
        pipeline_parallel_size=1,
    )

    # Partial configs (tests and draft helpers) fall back to 1 for the optional
    # fields; the PCP/DCP sizes are always read from the config.
    partial_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2, decode_context_parallel_size=2)
    )
    partial_topology = resolve_cp_topology(partial_config)
    assert (partial_topology.tp_size, partial_topology.interleave_size) == (1, 1)
    assert partial_topology.pipeline_parallel_size == 1


def test_pcp_only_mrv2_message_preserved() -> None:
    topology = CPTopology(tp_size=1, pcp_size=2, dcp_size=1)
    with pytest.raises(ValueError, match="Prefill Context Parallel"):
        validate_common_topology(topology, vllm_config=SimpleNamespace(use_v2_model_runner=False))
    validate_common_topology(topology, vllm_config=SimpleNamespace(use_v2_model_runner=True))


def test_parallel_config_fields_roundtrip() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=4, interleave_size=128, pipeline_parallel_size=2)
    fields = topology.to_parallel_config_fields()
    assert fields == {
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "decode_context_parallel_size": 4,
        "cp_kv_cache_interleave_size": 128,
        "pipeline_parallel_size": 2,
    }
    recreated = CPTopology(
        tp_size=fields["tensor_parallel_size"],
        pcp_size=fields["prefill_context_parallel_size"],
        dcp_size=fields["decode_context_parallel_size"],
        interleave_size=fields["cp_kv_cache_interleave_size"],
        pipeline_parallel_size=fields["pipeline_parallel_size"],
    )
    assert recreated == topology
