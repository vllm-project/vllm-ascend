"""Regression tests for bind_kv_cache per-layer dispatch.

The patched bind_kv_cache used to assign the raw allocation directly to
``forward_context[layer].kv_cache``, skipping each layer's own
``bind_kv_cache`` override. Mamba-family layers rely on that override to
unpack the raw buffer into conv/ssm state views; without the dispatch every
GLM-5.3-Flash KDA layer failed at the first forward with
"too many values to unpack (expected 2)".
"""

import torch

from tests.ut.base import TestBase
from vllm_ascend.patch.worker import patch_bind_kv_cache as patch_mod


class _RecordingMambaLayer:
    """Minimal Mamba-shaped layer: splits the raw buffer into two states."""

    def __init__(self):
        self.kv_cache: tuple = ()
        self.bind_calls = 0

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.bind_calls += 1
        pages = kv_cache.squeeze(dim=(1, 2))
        self.kv_cache = (pages[:, :8], pages[:, 8:])


class _PlainAttentionLayer:
    """Mirrors AttentionLayerBase: store the view as-is."""

    def __init__(self):
        self.kv_cache = None

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache


class TestBindKVCacheDispatch(TestBase):
    def test_dispatches_to_each_layers_bind(self):
        mamba = _RecordingMambaLayer()
        plain = _PlainAttentionLayer()
        raw = torch.zeros(4, 1, 1, 16, dtype=torch.int8)
        kv_caches = {
            "model.layers.0.linear_attn": raw,
            "model.layers.1.self_attn": raw,
        }
        forward_context = {
            "model.layers.0.linear_attn": mamba,
            "model.layers.1.self_attn": plain,
        }
        runner_kv_caches: list = []

        patch_mod.bind_kv_cache(kv_caches, forward_context, runner_kv_caches)

        # The Mamba layer's own override ran and unpacked the buffer into
        # conv/ssm views instead of storing the raw 4-D tensor.
        self.assertEqual(mamba.bind_calls, 1)
        self.assertEqual(len(mamba.kv_cache), 2)
        # Plain attention layers keep the as-is view.
        self.assertIs(plain.kv_cache, raw)
        # Runner cache list is filled in layer-index order.
        self.assertEqual(runner_kv_caches, [raw, raw])

    def test_module_patching_wired(self):
        """The patched module replaces vllm's bind_kv_cache."""
        import vllm.v1.worker.utils as utils

        self.assertIs(utils.bind_kv_cache, patch_mod.bind_kv_cache)
