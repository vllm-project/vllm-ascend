# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Late layout resolution: the worker rebinds attention impls at adoption.

The engine core constructs the executor (model load, attention impl
__init__) BEFORE resolve_kv_cache_layout runs, so an impl constructed in
that window caches the PRE-resolution fallback binding. The
set_kv_cache_layout RPC -- after resolution, before every
layout-dependent operation (memory profiling, allocation, KV writes,
graph capture) -- must give every already-constructed impl the
effective layout, or attention dispatch (use_bnsd_kv_cache) diverges
from the allocation (get_kv_cache_shape) that the same resolved
CacheConfig drives.

The control case (a pre-populated CacheConfig) constructs the impl with
the final layout already present, so adoption is a no-op rebind.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    needs_layer_aware_fia_graph_replay,
)
from vllm_ascend.worker.worker import NPUWorker

BS = 128


def _mock_config(cache_config):
    config = MagicMock()
    config.speculative_config = None
    config.parallel_config.prefill_context_parallel_size = 1
    config.quant_config = None
    config.model_config.dtype = torch.bfloat16
    config.cache_config = cache_config
    return config


def _make_impl(kv_cache_layout):
    """A REAL impl constructed in the executor-construction window: the
    engine core has not resolved the layout yet, so the config carries
    only what the caller pre-populated and the env decides the rest."""
    cache_config = SimpleNamespace(kv_cache_layout=kv_cache_layout)
    mock_config = _mock_config(cache_config)
    with (
        patch(
            "vllm_ascend.attention.attention_v1.get_current_vllm_config",
            return_value=mock_config,
        ),
        patch(
            "vllm_ascend.attention.utils.get_current_vllm_config",
            return_value=mock_config,
        ),
    ):
        needs_layer_aware_fia_graph_replay.cache_clear()
        impl = AscendAttentionBackendImpl(
            num_heads=8,
            head_size=64,
            scale=1.0,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="float16",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
        )
    needs_layer_aware_fia_graph_replay.cache_clear()
    return impl, cache_config, mock_config


def _make_worker(impl, cache_config, cudagraph_mode=CUDAGraphMode.NONE):
    """A bare NPUWorker: no device init, only the RPC path runs."""
    worker = NPUWorker.__new__(NPUWorker)
    worker.vllm_config = SimpleNamespace(
        cache_config=cache_config,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=0),
        compilation_config=SimpleNamespace(
            static_forward_context={"model.layers.0": SimpleNamespace(impl=impl)},
            cudagraph_mode=cudagraph_mode,
        ),
    )
    return worker


class TestLayoutAdoption:
    def test_adoption_rebinds_dispatch_to_effective_layout(self, monkeypatch):
        """RED on the pre-rebind patch: the ctor-cached provisional
        binding survives adoption, so dispatch diverges from the
        allocation that follows the same resolved CacheConfig."""
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        impl, cache_config, mock_config = _make_impl(kv_cache_layout=None)
        # Constructed pre-resolution with no env: the provisional
        # binding is the token-major default.
        assert impl.kv_cache_layout is None
        assert impl.use_bnsd_kv_cache is False

        worker = _make_worker(impl, cache_config)
        worker.set_kv_cache_layout("LBHNC", 0)

        # The RPC adopted the layout on the shared CacheConfig...
        assert cache_config.kv_cache_layout == "LBHNC"
        # ...attention dispatch follows it...
        assert impl.kv_cache_layout == "LBHNC"
        assert impl.use_bnsd_kv_cache is True
        # ...and so does the first layout-dependent allocation shape
        # (head-major: heads on dim 2, tokens on dim 3).
        with patch(
            "vllm_ascend.attention.attention_v1.get_current_vllm_config",
            return_value=mock_config,
        ):
            assert AscendAttentionBackend.get_kv_cache_shape(4, BS, 2, 64) == (
                2,
                4,
                2,
                BS,
                64,
            )

    def test_control_prepopulated_config_binds_at_construction(self, monkeypatch):
        """Control: an explicit pre-populated CacheConfig constructs the
        impl with the final layout; adoption is a no-op rebind."""
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        impl, cache_config, _ = _make_impl(kv_cache_layout="LBHNC")
        assert impl.kv_cache_layout == "LBHNC"
        assert impl.use_bnsd_kv_cache is True

        worker = _make_worker(impl, cache_config)
        worker.set_kv_cache_layout("LBHNC", 0)
        assert impl.use_bnsd_kv_cache is True
        assert impl.kv_cache_layout == "LBHNC"

    def test_lbnhc_adoption_rebinds_to_token_major(self, monkeypatch):
        """A resolution to the token-major layout rebinds the impl the
        same way (the engine core's vote may land on either)."""
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        impl, cache_config, mock_config = _make_impl(kv_cache_layout=None)
        assert impl.use_bnsd_kv_cache is False
        worker = _make_worker(impl, cache_config)
        worker.set_kv_cache_layout("LBNHC", 0)
        assert impl.use_bnsd_kv_cache is False
        assert impl.kv_cache_layout == "LBNHC"
        with patch(
            "vllm_ascend.attention.attention_v1.get_current_vllm_config",
            return_value=mock_config,
        ):
            assert AscendAttentionBackend.get_kv_cache_shape(4, BS, 2, 64) == (
                2,
                4,
                BS,
                2,
                64,
            )

    def test_adoption_cannot_change_a_resolved_layout(self, monkeypatch):
        """The layout is immutable once resolved: adopting a different
        one over an explicit config fails loudly (the allocation would
        have followed the recorded one)."""
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        impl, cache_config, _ = _make_impl(kv_cache_layout="LBHNC")
        worker = _make_worker(impl, cache_config)
        with pytest.raises(ValueError, match="already resolved"):
            worker.set_kv_cache_layout("LBNHC", 0)

    def test_adoption_refuses_full_graph_with_head_major_layout(self, monkeypatch):
        """The effective-layout graph gate: FULL graphs + the head-major
        layout crash the first replayed prefill graph (507011), and the
        pre-resolution platform gate cannot have seen this layout."""
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        impl, cache_config, _ = _make_impl(kv_cache_layout=None)
        worker = _make_worker(impl, cache_config, cudagraph_mode=CUDAGraphMode.FULL)
        with pytest.raises(RuntimeError, match="507011"):
            worker.set_kv_cache_layout("LBHNC", 0)

    def test_admission_allows_full_decode_only_with_head_major(self, monkeypatch):
        """FULL_DECODE_ONLY is the validated graph mode for the
        head-major layout: adoption proceeds and rebinds."""
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        impl, cache_config, _ = _make_impl(kv_cache_layout=None)
        worker = _make_worker(impl, cache_config, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
        worker.set_kv_cache_layout("LBHNC", 0)
        assert impl.use_bnsd_kv_cache is True


class TestRebindMethod:
    def test_rebind_returns_whether_binding_changed(self, monkeypatch):
        monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
        impl, _, _ = _make_impl(kv_cache_layout=None)
        assert impl.rebind_kv_cache_layout("LBHNC") is True
        assert impl.rebind_kv_cache_layout("LBHNC") is False
        assert impl.rebind_kv_cache_layout(None) is False
        assert impl.use_bnsd_kv_cache is True
        assert impl.rebind_kv_cache_layout("LBNHC") is True
        assert impl.use_bnsd_kv_cache is False
