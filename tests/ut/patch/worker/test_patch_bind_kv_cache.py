# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.patch.worker import patch_bind_kv_cache as binding


def test_mrv2_binding_preserves_ascend_tuple_and_tracker_order():
    assert binding.gpu_attn_utils.bind_kv_cache_to_layers is binding.bind_kv_cache_to_layers
    caches = {name: (torch.zeros(2, 3), torch.ones(2, 1)) for name in ("model.layers.10.attn", "model.layers.2.attn")}
    layers = {name: SimpleNamespace(bind_kv_cache=Mock(side_effect=AssertionError("CUDA layout"))) for name in caches}
    groups = []
    with (
        patch.object(binding, "vllm_version_is", return_value=False),
        patch.object(binding.utils, "share_replayssm_ring_trackers") as trackers,
    ):
        binding.bind_kv_cache_to_layers(caches, layers, kv_cache_groups=groups)
        trackers.assert_called_once_with(["model.layers.2.attn", "model.layers.10.attn"], layers, groups)
    for name, cache in caches.items():
        assert layers[name].kv_cache is cache


def test_legacy_binding_keeps_runner_layer_order():
    caches = {name: (torch.zeros(1), torch.ones(1)) for name in ("model.layers.10.attn", "model.layers.2.attn")}
    layers = {name: SimpleNamespace() for name in caches}
    runner_caches = []
    with patch.object(binding, "vllm_version_is", return_value=True):
        binding.bind_kv_cache(caches, layers, runner_caches)
    assert runner_caches[0] is caches["model.layers.2.attn"]
    assert runner_caches[1] is caches["model.layers.10.attn"]
