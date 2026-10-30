# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.distributed.gva import (
    GvaCapabilities,
    GvaObjectSpec,
    GvaPoolMember,
    GvaPoolSpec,
)


def test_gva_capabilities_require_an_access_mode():
    with pytest.raises(ValueError, match="direct_map or entry_gather"):
        GvaCapabilities()

    assert GvaCapabilities(entry_gather=True, graph_safe=True, max_entry_bytes=64).entry_gather


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
