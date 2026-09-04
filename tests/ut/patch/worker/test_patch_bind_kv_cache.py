# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import vllm.v1.worker.utils as upstream_utils

from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache


@pytest.mark.parametrize("legacy", [False, True])
def test_bind_kv_cache_forwards_replayssm_group_metadata(monkeypatch, legacy) -> None:
    layer_name = "model.layers.0.self_attn"
    kv_cache = torch.empty(1)
    forward_context = {layer_name: SimpleNamespace(kv_cache=None)}
    runner_kv_caches: list[torch.Tensor] = []
    kv_cache_groups = [SimpleNamespace(layer_names=[layer_name])]
    calls = []
    monkeypatch.setattr("vllm_ascend.patch.worker.patch_bind_kv_cache.vllm_version_is", lambda _: legacy)

    monkeypatch.setattr(
        upstream_utils,
        "share_replayssm_ring_trackers",
        lambda ordered_names, context, groups: calls.append((ordered_names, context, groups)),
        raising=False,
    )

    bind_kv_cache(
        {layer_name: kv_cache},
        forward_context,
        runner_kv_caches,
        kv_cache_groups=kv_cache_groups,
    )

    assert runner_kv_caches == [kv_cache]
    assert forward_context[layer_name].kv_cache is kv_cache
    assert calls == ([] if legacy else [([layer_name], forward_context, kv_cache_groups)])
