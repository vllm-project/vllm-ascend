# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

import pytest

from vllm_ascend.distributed.gva import (
    GvaCapabilities,
    GvaObjectRef,
    GvaObjectSpec,
    GvaPoolMember,
    GvaPoolRef,
    GvaPoolSpec,
    GvaRowRange,
)


class _IntegralValue(IntEnum):
    ZERO = 0
    ONE = 1
    THREE = 3
    FOUR = 4


def test_gva_capabilities_require_an_access_mode():
    with pytest.raises(ValueError, match="direct_map or entry_gather"):
        GvaCapabilities()

    assert GvaCapabilities(entry_gather=True, graph_safe=True, max_entry_bytes=64).entry_gather


def test_integral_descriptor_values_are_normalized():
    row_range = GvaRowRange(_IntegralValue.ONE, _IntegralValue.ZERO, _IntegralValue.THREE)
    capabilities = GvaCapabilities(entry_gather=True, max_entry_bytes=_IntegralValue.THREE)
    pool = GvaPoolSpec(
        "qwen-ple",
        "generation-1",
        "offload",
        (_IntegralValue.ZERO, _IntegralValue.ONE),
        _IntegralValue.FOUR,
        "tcp://127.0.0.1:8500",
        "schema",
    )
    member = GvaPoolMember(_IntegralValue.ONE, _IntegralValue.ZERO)
    object_spec = GvaObjectSpec("layer-0", "generation-1", _IntegralValue.FOUR, "schema", "ckpt", "layout")
    object_ref = GvaObjectRef(
        GvaPoolRef("pool", "generation-1", "offload", "schema"),
        "layer-0",
        _IntegralValue.FOUR,
        "schema",
        "ckpt",
        "layout",
    )

    assert all(type(value) is int for value in (row_range.owner_rank, row_range.start, row_range.end))
    assert type(capabilities.max_entry_bytes) is int
    assert all(type(value) is int for value in (*pool.members, pool.capacity_bytes, member.rank, member.device_id))
    assert type(object_spec.size_bytes) is int
    assert type(object_ref.size_bytes) is int


def test_gva_descriptors_keep_identity_and_membership():
    members = [0, 2]
    pool = GvaPoolSpec(
        pool_id="qwen-ple",
        generation="generation-1",
        api_family="offload",
        members=members,
        capacity_bytes=4096,
        store_url="tcp://127.0.0.1:8500",
        schema="qwen-ple-bf16-v1",
    )
    members.append(3)
    pool_member = GvaPoolMember(rank=2, device_id=3)
    object_spec = GvaObjectSpec(
        "ple-layer-2", "generation-1", 1024, "qwen-ple-bf16-v1", "sha256:weights", "sha256:rows"
    )
    pool_ref = GvaPoolRef("qwen-ple", "generation-1", "offload", "qwen-ple-bf16-v1")
    object_ref = GvaObjectRef(
        pool_ref,
        "ple-layer-2",
        1024,
        "qwen-ple-bf16-v1",
        "sha256:weights",
        "sha256:rows",
    )

    assert pool.members == (0, 2)
    assert pool_member.device_id == 3
    assert object_spec.checkpoint_identity == object_ref.checkpoint_identity
    assert object_ref.pool == pool_ref


def test_gva_pool_spec_rejects_invalid_members():
    kwargs = {
        "pool_id": "qwen-ple",
        "generation": "generation-1",
        "api_family": "offload",
        "capacity_bytes": 4096,
        "store_url": "tcp://127.0.0.1:8500",
        "schema": "qwen-ple-bf16-v1",
    }
    with pytest.raises(ValueError, match="integer rank IDs"):
        GvaPoolSpec(members=(0, True), **kwargs)
