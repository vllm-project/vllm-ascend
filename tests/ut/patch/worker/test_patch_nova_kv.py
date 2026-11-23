# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.attention.nova_kv import NovaAttentionBackend, NovaAttentionImpl, NovaMetadataBuilder
from vllm_ascend.patch.worker import patch_nova_kv as patch


def make_config(options=None):
    return SimpleNamespace(
        additional_config={} if options is None else {"nova_kv": options},
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text"), max_model_len=91024),
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=4),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE")),
        cache_config=SimpleNamespace(enable_prefix_caching=False, cache_dtype="auto"),
        speculative_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1, pipeline_parallel_size=1
        ),
    )


def test_disabled_constructor_keeps_original_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: make_config())
    monkeypatch.setattr(patch, "_ORIGINAL_INIT", lambda *args, **kwargs: calls.append((args, kwargs)))
    layer = SimpleNamespace()
    patch.nova_init(layer, 4, 256, 0.0625, prefix="model.layers.0.self_attn.attn")
    assert calls == [((layer, 4, 256, 0.0625), {"prefix": "model.layers.0.self_attn.attn"})]


@pytest.mark.parametrize("v2,batch,length", [(False, 4, 91024), (True, 5, 91024), (True, 4, 100000)])
def test_nova_rejects_unsupported_runner_or_capacity(monkeypatch, v2, batch, length):
    config = make_config({"artifact_path": "/artifacts"})
    config.use_v2_model_runner = v2
    config.scheduler_config.max_num_seqs = batch
    config.model_config.max_model_len = length
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    with pytest.raises(ValueError, match="NOVA-KV"):
        patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)


@pytest.mark.parametrize("backend", ["tilelang", "direct", "cube"])
def test_removed_backend_is_rejected_before_layer_initialization(monkeypatch, backend):
    config = make_config({"artifact_path": "/artifacts", "backend": backend})
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    with pytest.raises(ValueError, match="backend"):
        patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)


def test_nova_backend_selected_before_initialization(monkeypatch):
    config = make_config({"artifact_path": "/artifacts"})
    config.use_v2_model_runner = True
    config.scheduler_config.max_num_seqs = 4
    config.model_config.max_model_len = 91024
    backend = type("NovaBackend", (NovaAttentionBackend,), {})
    monkeypatch.setattr("vllm_ascend.attention.nova_kv.NovaAttentionBackend", backend)
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    observed = []
    monkeypatch.setattr(patch, "_ORIGINAL_INIT", lambda *args, **kwargs: observed.append(kwargs["attn_backend"]))
    layer = SimpleNamespace()
    patch.nova_init(layer, 4, 256, 0.0625)
    assert observed == [backend]
    assert layer.align_kv_cache_with_mamba is False
    assert "vllm_ascend.ops.triton.nova_direct_decode" not in sys.modules


def test_packed_spec_preserves_allocator_padding(monkeypatch):
    dense = FullAttentionSpec(
        block_size=128, num_kv_heads=2, head_size=256, dtype=torch.bfloat16, page_size_padded=262144
    )
    monkeypatch.setattr(patch, "_ORIGINAL_SPEC", lambda self, config: dense)
    layer = SimpleNamespace(attn_backend=NovaAttentionBackend, align_kv_cache_with_mamba=False)
    result = patch.nova_spec(layer, make_config())
    assert result.dtype == torch.uint8
    assert result.state_content_bytes == 134
    assert result.page_size_padded == 262144
    assert result.block_size == dense.block_size
    assert NovaAttentionBackend.customize_spec(result) == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("speculative_config", object()),
        ("kv_transfer_config", object()),
    ],
)
def test_unsupported_runtime_rejected(monkeypatch, field, value):
    config = make_config()
    setattr(config, field, value)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    with pytest.raises(ValueError):
        patch._validate(config)


@pytest.mark.parametrize("mode", ["NONE", "FULL_DECODE_ONLY"])
def test_graph_modes_allowed_for_compressed_backend(monkeypatch, mode):
    config = make_config({"artifact_path": "/artifacts"})
    config.compilation_config.cudagraph_mode.name = mode
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "_ORIGINAL_INIT", lambda *args, **kwargs: None)
    patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)
    assert NovaMetadataBuilder.get_cudagraph_support(config, None) == AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


@pytest.mark.parametrize("mode", ["PIECEWISE", "FULL", "FULL_AND_PIECEWISE"])
def test_unsupported_graph_modes_rejected(monkeypatch, mode):
    config = make_config({"artifact_path": "/artifacts"})
    config.compilation_config.cudagraph_mode.name = mode
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    with pytest.raises(ValueError, match="NONE or FULL_DECODE_ONLY"):
        patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)


@pytest.mark.parametrize("field,value", [("async_scheduling", True), ("pipeline_parallel_size", 2)])
def test_recovery_rejects_overlapping_batches(monkeypatch, field, value):
    config = make_config({"artifact_path": "/artifacts"})
    setattr(config.scheduler_config if field == "async_scheduling" else config.parallel_config, field, value)
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    with pytest.raises(ValueError, match="synchronous scheduling"):
        patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)


def test_calibration_capture_still_requires_eager(monkeypatch):
    config = make_config({"capture_path": "/capture"})
    config.compilation_config.cudagraph_mode.name = "FULL_DECODE_ONLY"
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    with pytest.raises(ValueError, match="calibration capture requires --enforce-eager"):
        patch.nova_init(SimpleNamespace(), 4, 256, 0.0625)


def test_capture_preserves_original_attention_and_collects_only_actual_rows(monkeypatch):
    config = make_config({"capture_path": "/capture"})
    layer = SimpleNamespace(layer_name="model.layers.0.self_attn.attn")
    original_forward = Mock(return_value="original-output")
    capture = SimpleNamespace(add=Mock())
    factory = Mock(return_value=capture)
    monkeypatch.setattr(patch, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(patch, "enable_sp", lambda config: False)
    monkeypatch.setattr(
        patch,
        "_ORIGINAL_INIT",
        lambda self, *args, **kwargs: setattr(self, "impl", SimpleNamespace(forward=original_forward)),
    )
    monkeypatch.setattr("vllm_ascend.quantization.methods.kv_cache.nova_kv.capture.NovaCapture", factory)

    patch.nova_init(layer, 4, 256, 0.0625)
    query = torch.ones(4, 2, 256)
    key, value = torch.ones(4, 1, 256), torch.zeros(4, 1, 256)
    cache, output = object(), object()
    metadata = SimpleNamespace(num_actual_tokens=3)
    result = layer.impl.forward(layer, query, key, value, cache, metadata, output=output)

    assert result == "original-output"
    factory.assert_called_once()
    actual = capture.add.call_args.args
    for observed, expected in zip(actual, (query[:3], key[:3], value[:3])):
        torch.testing.assert_close(observed, expected)
    original_forward.assert_called_once_with(layer, query, key, value, cache, metadata, output=output)


def test_memory_profile_exercises_codec_without_lookup_temporaries():
    impl_cls = NovaAttentionImpl
    calls = []
    codec = SimpleNamespace(
        key_inverse=torch.eye(256)[None],
        key_mean=torch.zeros(1, 256),
        codebooks=torch.zeros(1, 64, 256, 4),
        _books_float=torch.zeros(1, 64, 256, 4),
        encode=lambda key, value, **kwargs: calls.append(("encode", len(key), kwargs)),
    )
    impl = object.__new__(impl_cls)
    impl.head_size = 256
    impl.num_kv_heads = 1
    impl._limit = 2
    impl._options = SimpleNamespace(codec_batch_size=3)
    impl._codec = lambda layer, device: codec
    impl._ensure_runtime = lambda device, dtype: None
    query = torch.ones(5, 2, 256)
    key = torch.ones(5, 1, 256)
    output = torch.empty_like(query)
    impl.forward(SimpleNamespace(head_size_v=256), query, key, key, None, None, output=output)
    assert calls == [("encode", 5, {"chunk_size": 3})]
    assert not output.any()
