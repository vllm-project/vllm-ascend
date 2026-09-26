# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Unit tests for the Model Runner V2 KV cache map passed to KV connectors.

Model Runner V2 filters ``kv_caches_dict.values()`` by ``.device`` and
therefore needs one tensor per layer, while Ascend keeps K and V as separate
views. These tests pin the boundary: the runner gets a single tensor per
layer, the KV connector gets every view.
"""

import pytest
import torch
from vllm.v1.worker.gpu import kv_connector as kv_connector_module

from vllm_ascend.patch.worker.patch_v2 import patch_attn_utils


def _fake_init_kv_cache(kv_caches: dict) -> object:
    def fake_init_kv_cache(*args, **kwargs) -> dict:
        return kv_caches

    return fake_init_kv_cache


def test_init_kv_cache_keeps_full_views(monkeypatch: pytest.MonkeyPatch) -> None:
    key_cache = torch.empty(2, 4, dtype=torch.uint8)
    value_cache = torch.empty(2, 4, dtype=torch.uint8)
    monkeypatch.setattr(
        patch_attn_utils,
        "_orig_init_kv_cache",
        _fake_init_kv_cache({"layer": (key_cache, value_cache)}),
    )

    kv_caches = patch_attn_utils._ascend_init_kv_cache()

    # The runner filters the values by `.device`, so each layer exposes a tensor.
    assert kv_caches["layer"] is key_cache
    # The complete views stay reachable for the KV connector.
    assert kv_caches.full_view == {"layer": (key_cache, value_cache)}


def test_get_kv_connector_receives_full_views(monkeypatch: pytest.MonkeyPatch) -> None:
    key_cache = torch.empty(2, 4, dtype=torch.uint8)
    value_cache = torch.empty(2, 4, dtype=torch.uint8)
    seen: dict[str, object] = {}

    def fake_get_kv_connector(vllm_config, kv_caches_dict):
        seen["kv_caches"] = kv_caches_dict
        return "connector"

    monkeypatch.setattr(
        patch_attn_utils,
        "_orig_init_kv_cache",
        _fake_init_kv_cache({"layer": (key_cache, value_cache)}),
    )
    monkeypatch.setattr(kv_connector_module, "get_kv_connector", fake_get_kv_connector)

    runner_view = patch_attn_utils._ascend_init_kv_cache()
    assert patch_attn_utils._ascend_get_kv_connector("vllm-config", runner_view) == "connector"
    assert seen["kv_caches"] == {"layer": (key_cache, value_cache)}

    # Anything that is not a runner view is forwarded unchanged.
    patch_attn_utils._ascend_get_kv_connector("vllm-config", {"plain": key_cache})
    assert seen["kv_caches"] == {"plain": key_cache}
