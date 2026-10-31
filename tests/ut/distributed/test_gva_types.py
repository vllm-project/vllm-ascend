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
)


class _IntegralValue(IntEnum):
    ZERO = 0
    ONE = 1
    SIXTY_FOUR = 64
    CAPACITY = 4096


def test_gva_capabilities_require_an_access_mode():
    with pytest.raises(ValueError, match="direct_map or entry_gather"):
        GvaCapabilities()

    assert GvaCapabilities(entry_gather=True, graph_safe=True, max_entry_bytes=64).entry_gather


def test_gva_descriptors_normalize_integral_fields():
    capabilities = GvaCapabilities(entry_gather=True, max_entry_bytes=_IntegralValue.SIXTY_FOUR)
    pool = GvaPoolSpec(
        pool_id="qwen-ple",
        generation="generation-1",
        api_family="offload",
        members=(_IntegralValue.ZERO, _IntegralValue.ONE),
        capacity_bytes=_IntegralValue.CAPACITY,
        store_url="tcp://127.0.0.1:8500",
        schema="qwen-ple-bf16-v1",
    )
    member = GvaPoolMember(rank=_IntegralValue.ONE, device_id=_IntegralValue.ZERO)
    object_spec = GvaObjectSpec("ple-layer-2", "generation-1", _IntegralValue.CAPACITY, "schema", "ckpt", "layout")
    object_ref = GvaObjectRef(
        GvaPoolRef("qwen-ple", "generation-1", "offload", "schema"),
        "ple-layer-2",
        _IntegralValue.CAPACITY,
        "schema",
        "ckpt",
        "layout",
    )

    assert type(capabilities.max_entry_bytes) is int
    assert all(type(rank) is int for rank in pool.members)
    assert type(pool.capacity_bytes) is int
    assert type(member.rank) is int
    assert type(member.device_id) is int
    assert type(object_spec.size_bytes) is int
    assert type(object_ref.size_bytes) is int


def test_gva_pool_spec_requires_static_membership_and_valid_rank():
    members = [0, 2]
    spec = GvaPoolSpec(
        pool_id="qwen-ple",
        generation="generation-1",
        api_family="offload",
        members=members,
        capacity_bytes=4096,
        store_url="tcp://127.0.0.1:8500",
        schema="qwen-ple-bf16-v1",
    )
    members.append(3)
    assert spec.members == (0, 2)
    assert hash(spec)

    with pytest.raises(ValueError, match="integer rank IDs"):
        GvaPoolSpec(
            pool_id="qwen-ple",
            generation="generation-1",
            api_family="offload",
            members=(0, True),
            capacity_bytes=4096,
            store_url="tcp://127.0.0.1:8500",
            schema="qwen-ple-bf16-v1",
        )

    with pytest.raises(ValueError, match="non-negative rank IDs"):
        GvaPoolSpec(
            pool_id="qwen-ple",
            generation="generation-1",
            api_family="offload",
            members=(0, -1),
            capacity_bytes=4096,
            store_url="tcp://127.0.0.1:8500",
            schema="qwen-ple-bf16-v1",
        )


def test_gva_object_spec_keeps_checkpoint_identity_and_schema():
    spec = GvaObjectSpec(
        object_id="ple-layer-2",
        generation="generation-1",
        size_bytes=1024,
        schema="qwen-ple-bf16-v1",
        checkpoint_identity="sha256:weights-index",
        layout_identity="sha256:row-layout",
    )
    assert spec.checkpoint_identity == "sha256:weights-index"

    with pytest.raises(ValueError, match="size_bytes"):
        GvaObjectSpec("ple-layer-2", "generation-1", 0, "schema", "checkpoint", "layout")


def test_pool_member_device_identity_is_separate_from_shared_pool_spec():
    assert GvaPoolMember(rank=2, device_id=3) != GvaPoolMember(rank=0, device_id=1)

    with pytest.raises(ValueError, match="device_id"):
        GvaPoolMember(rank=0, device_id=-1)
