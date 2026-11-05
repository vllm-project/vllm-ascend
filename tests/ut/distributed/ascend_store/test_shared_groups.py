# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker ownership survives the scheduler's representative-only cache config."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import lookup_key
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_transfer import SharedPoolTransfer


def description():
    # Group 1 is draft-only; group 2 is a request-lifetime ring.
    return {"version": 1, "group_count": 4, "target_groups": [0, 2, 3], "prefix_groups": [0, 3], "stateful": True}


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"group_count": 5},
        {"target_groups": [0, 0]},
        {"target_groups": [True]},
        {"target_groups": [4]},
        {"prefix_groups": [1]},
        {"prefix_groups": []},
        {"stateful": None},
    ],
)
def test_malformed_group_manifest_rejected(change):
    value = description() | change
    with pytest.raises(ValueError):
        lookup_key.validate_shared_groups(value, 4)


def test_registered_layouts_are_authoritative():
    transfer = object.__new__(SharedPoolTransfer)
    transfer._registered = False
    transfer.worker = SimpleNamespace(kv_cache_config=SimpleNamespace(kv_cache_groups=[None] * 4))
    transfer.layouts = {0: SimpleNamespace(specs={}), 3: SimpleNamespace(specs={})}
    transfer.request_states = {2: object()}
    with pytest.raises(RuntimeError, match="not been registered"):
        transfer.describe_groups()
    transfer._registered = True
    assert transfer.describe_groups() == description()


@pytest.mark.parametrize("bad_reply", [False, True])
def test_existing_lookup_rpc_returns_groups_and_fails_closed(tmp_path, monkeypatch, bad_reply):
    monkeypatch.setattr(lookup_key, "get_zmq_rpc_path_lookup", lambda _: f"ipc://{tmp_path}/groups")
    manifest = deepcopy(description())
    if bad_reply:
        manifest["prefix_groups"] = [1]
    worker = SimpleNamespace(shared_pool=SimpleNamespace(describe_groups=lambda: manifest))
    server = lookup_key.LookupKeyServer(worker, None)
    client = lookup_key.LookupKeyClient(None)
    try:
        if bad_reply:
            with pytest.raises(ValueError):
                client.describe_shared_groups(4)
            assert client.closed
        else:
            assert client.describe_shared_groups(4) == description()
            assert not client.closed
    finally:
        client.close()
        server.close()


def test_unavailable_worker_is_not_a_cache_miss(tmp_path, monkeypatch):
    import zmq

    monkeypatch.setattr(lookup_key, "get_zmq_rpc_path_lookup", lambda _: f"ipc://{tmp_path}/absent")
    client = lookup_key.LookupKeyClient(None)
    client.socket.setsockopt(zmq.RCVTIMEO, 20)
    with pytest.raises(zmq.ZMQError):
        client.describe_shared_groups(4)
    assert client.closed


@pytest.mark.parametrize("draft_first", [False, True])
def test_native_scheduler_compression_loses_ownership(draft_first):
    import torch
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
    from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.shared_transfer import target_group_specs

    target = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float16)
    draft = replace(target, is_draft=True)
    specs = {"draft": draft, "target": target} if draft_first else {"target": target, "draft": draft}
    group = SimpleNamespace(
        layer_names=list(specs), kv_cache_spec=UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs=specs)
    )
    worker_config = SimpleNamespace(num_blocks=3, hisparse_host_num_blocks=0, kv_cache_groups=[group])
    scheduler_config = generate_scheduler_kv_cache_config([worker_config])
    assert list(target_group_specs(group)) == ["target"]
    # Neither representative ordering preserves the actual owner set.
    assert list(target_group_specs(scheduler_config.kv_cache_groups[0])) != ["target"]
    assert list(target_group_specs(group)) == ["target"]  # original remains intact
