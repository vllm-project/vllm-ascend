# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.models.qwen4_exp.common.ngram_embedding import (
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

from vllm_ascend.models.qwen4_exp.ple_quant import (
    AscendPLEQuantizedEmbeddingMethod,
    load_ple_quantized_weights,
    modelslim_ple_method,
    patch_modelslim_ple,
)
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig


class Embedding(nn.Module):
    supports_prefetch = False

    def __init__(self, mxfp8=True, start=2, end=7, vocab=10, width=160):
        super().__init__()
        self.org_vocab_size = vocab
        self.embedding_dim = width
        self.shard_indices = SimpleNamespace(org_vocab_start_index=start, org_vocab_end_index=end)
        self.embedding_method = AscendPLEQuantizedEmbeddingMethod(mxfp8=mxfp8)
        self.embedding_method.create_weights(
            self, width, [end - start], width, vocab, torch.bfloat16, weight_loader=self.weight_loader
        )

    def allocate_embedding_weight(self, rows, width, dtype):
        return torch.zeros(rows, width, dtype=dtype)

    def weight_loader(self, param, value, checkpoint_start):
        start = max(self.shard_indices.org_vocab_start_index, checkpoint_start)
        end = min(self.shard_indices.org_vocab_end_index, checkpoint_start + value.shape[0])
        if start < end:
            param.data[
                start - self.shard_indices.org_vocab_start_index : end - self.shard_indices.org_vocab_start_index
            ].copy_(value[start - checkpoint_start : end - checkpoint_start])


@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("shape", [(0,), (1,), (2, 3)])
def test_selected_rows_match_reference(mxfp8, shape):
    layer = Embedding(mxfp8)
    codes = torch.arange(800).reshape(5, 160).remainder(15).sub(7)
    layer.weight.data.copy_(codes.to(layer.weight.dtype))
    if mxfp8:
        layer.weight_scale.data.copy_(torch.tensor([125, 126, 127, 128, 129], dtype=torch.uint8))
    else:
        layer.weight_scale.data.copy_(torch.tensor([0.25, 0.5, 1, 2, 4]))
    ids = torch.arange(torch.Size(shape).numel()).reshape(shape).remainder(5)
    reference = (codes.float().unflatten(-1, (5, 32)) * torch.tensor([0.25, 0.5, 1, 2, 4])[:, None]).flatten(-2)
    expected = reference[ids].bfloat16()
    result = layer.embedding_method.embedding(layer, ids)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert layer.weight.element_size() == 1


def test_e8m0_extremes_and_no_double_scaling():
    layer = Embedding(width=160)
    layer.weight.data.fill_(1)
    exponents = torch.tensor([0, 1, 127, 254, 255], dtype=torch.uint8)
    layer.weight_scale.data.copy_(exponents)
    expected = torch.exp2(exponents.double() - 127).float()
    expected[-1] = float("nan")
    expected = expected.repeat_interleave(32).expand(1, -1).bfloat16()
    result = layer.embedding_method.embedding(layer, torch.tensor([0]))
    torch.testing.assert_close(result, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        layer.embedding_method.dequantize(layer, result, torch.bfloat16), expected, equal_nan=True
    )


@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("start,end", [(2, 7), (5, 10), (0, 1), (10, 10)])
def test_streaming_scales_follow_tp_overlap(mxfp8, start, end):
    layer = Embedding(mxfp8, start, end)
    owner = SimpleNamespace(ngram_embedding=layer, split_ngram_parts=3)
    method = layer.embedding_method
    values = torch.arange(50).reshape(10, 5).to(method.scale_dtype)
    for index, value in enumerate(values.split(4)):
        name = f"ngram_embedding.shard_{index}{method.scale_suffix}"
        loaded = load_ple_quantized_weights(owner, [(name, value)], lambda _, rest: set(rest))
        assert loaded == {"ngram_embedding.weight_scale"}
    method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight_scale, values[start:end])
    with pytest.raises(ValueError, match="Duplicate PLE scale"):
        load_ple_quantized_weights(
            owner, [(f"ngram_embedding.shard_0{method.scale_suffix}", values[:4])], lambda _, rest: set(rest)
        )


@pytest.mark.parametrize("mxfp8", [False, True])
def test_no_checkpoint_loading_allows_dummy_weights(mxfp8):
    layer = Embedding(mxfp8)
    # DummyModelLoader initializes parameters without invoking the model's
    # checkpoint iterator. Postprocessing must not reject that lifecycle.
    layer.embedding_method.process_weights_after_loading(layer)


@pytest.mark.parametrize("mxfp8", [False, True])
def test_missing_and_invalid_scales(mxfp8):
    layer = Embedding(mxfp8)
    method = layer.embedding_method
    owner = SimpleNamespace(ngram_embedding=layer, split_ngram_parts=2)
    suffix = method.scale_suffix
    value = torch.ones(5, 5, dtype=method.scale_dtype)

    def load(name, tensor):
        return load_ple_quantized_weights(owner, [(name, tensor)], lambda _, rest: set(rest))

    # An actual checkpoint load, including an empty one, establishes the
    # expected shard set and must reject missing scales after loading.
    load_ple_quantized_weights(owner, iter(()), lambda _, rest: set(rest))
    with pytest.raises(ValueError, match="Missing PLE scale"):
        method.process_weights_after_loading(layer)
    for shard in ("-1", "2", "bad"):
        with pytest.raises(ValueError, match="Invalid PLE shard"):
            load(f"ngram_embedding.shard_{shard}{suffix}", value)
    for bad in (value[:, :4], value.double()):
        with pytest.raises(ValueError, match="PLE scale .* expected"):
            load(f"ngram_embedding.shard_0{suffix}", bad)
    wrong_suffix = ".scale" if mxfp8 else ".weight_scale"
    with pytest.raises(ValueError, match="Unexpected PLE scale format"):
        load(f"ngram_embedding.shard_0{wrong_suffix}", value)
    with pytest.raises(ValueError, match="PLE weight .* expected"):
        load("ngram_embedding.shard_0.weight", torch.ones(5, 160))
    load(f"ngram_embedding.shard_0{suffix}", value)
    with pytest.raises(ValueError, match="Missing PLE scale.*1"):
        method.process_weights_after_loading(layer)


def test_invalid_width_and_unsupported_prefetch():
    with pytest.raises(ValueError, match="divisible"):
        Embedding(width=159)
    layer = SimpleNamespace(supports_prefetch=True)
    with pytest.raises(NotImplementedError, match="prefetch"):
        AscendPLEQuantizedEmbeddingMethod(mxfp8=False).create_weights(layer, 160, [5], 160, 5, torch.bfloat16)


def test_modelslim_selection_rejects_conflicts():
    prefix = "model.layers.1.ple_embedding.ngram_embedding"
    key = f"{prefix}.shard_0"
    assert modelslim_ple_method({key + ".weight": "W8A8_MXFP8"}, prefix, "float8_e4m3fn").mxfp8
    assert not modelslim_ple_method({key + ".quant_type": "INT8_SYM"}, prefix, None).mxfp8
    assert isinstance(
        modelslim_ple_method({key + ".weight": "FLOAT"}, prefix, None), Qwen4ExpPLEUnquantizedEmbeddingMethod
    )
    for descriptions, dtype in (
        ({key + ".weight": "W8A8_MXFP8", key + ".quant_type": "INT8_SYM"}, None),
        ({key + ".weight": "W8A8_MXFP8", prefix + ".shard_1.weight": "FLOAT"}, None),
        ({key + ".quant_type": "INT8_SYM"}, "float8_e4m3fn"),
        ({key + ".weight": "W8A8_MXFP8"}, "nvfp4"),
        ({key + ".quant_type": "INT4_SYM"}, None),
        ({}, "float8_e4m3fn"),
    ):
        with pytest.raises(ValueError):
            modelslim_ple_method(descriptions, prefix, dtype)


def test_patch_is_idempotent_and_preserves_other_quantization():
    from vllm.models.qwen4_exp.amd import ple_layer
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEPinnedHostEmbedding

    patch_modelslim_ple()
    selector = Qwen4ExpPLEEmbeddingMethod.from_quant_config
    loader = ple_layer.Qwen4ExpNGramEmbedding.load_weights
    host = ple_layer.Qwen4ExpPLEPinnedHostEmbedding
    patch_modelslim_ple()
    assert Qwen4ExpPLEEmbeddingMethod.from_quant_config is selector
    assert ple_layer.Qwen4ExpNGramEmbedding.load_weights is loader
    assert ple_layer.Qwen4ExpPLEPinnedHostEmbedding is host
    assert host is Qwen4ExpPLEPinnedHostEmbedding
    assert isinstance(selector(None, "other"), Qwen4ExpPLEUnquantizedEmbeddingMethod)
    config = AscendModelSlimConfig({"ple.shard_0.quant_type": "INT8_SYM"})
    assert not selector(config, "ple").mxfp8
    layer = SimpleNamespace(ngram_embedding=SimpleNamespace(embedding_method=Qwen4ExpPLEUnquantizedEmbeddingMethod()))
    original = Mock(return_value={"untouched"})
    weights: Iterator[tuple[str, torch.Tensor]] = iter([])
    assert load_ple_quantized_weights(layer, weights, original) == {"untouched"}
    original.assert_called_once_with(layer, weights)


@pytest.mark.parametrize("mxfp8", [False, True])
def test_real_upstream_ngram_loader(mxfp8):
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpNGramEmbedding

    patch_modelslim_ple()
    layer = Embedding(mxfp8, start=2, end=7)
    owner = SimpleNamespace(
        ngram_embedding=layer,
        split_ngram_parts=2,
        layer_multipliers=torch.zeros(1),
        ngram_heads_offsets=torch.zeros(1),
        ngram_heads_vocab_sizes=torch.zeros(1),
    )
    method = layer.embedding_method
    codes = torch.arange(1600).reshape(10, 160).remainder(15).sub(7).to(method.weight_dtype)
    scales = torch.arange(50).reshape(10, 5).to(method.scale_dtype)
    weights = []
    for index in range(2):
        weights.extend(
            [
                (f"ngram_embedding.shard_{index}.weight", codes[index * 5 : (index + 1) * 5]),
                (f"ngram_embedding.shard_{index}{method.scale_suffix}", scales[index * 5 : (index + 1) * 5]),
            ]
        )
    loaded = Qwen4ExpNGramEmbedding.load_weights(owner, iter(weights))
    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    torch.testing.assert_close(layer.weight.view(torch.uint8), codes[2:7].view(torch.uint8))
    torch.testing.assert_close(layer.weight_scale, scales[2:7])
    method.process_weights_after_loading(layer)


def test_host_close_retries_failed_unregister():
    from vllm_ascend.models.qwen4_exp.ple_embedding import AscendPLEHostEmbedding

    layer = AscendPLEHostEmbedding.__new__(AscendPLEHostEmbedding)
    nn.Module.__init__(layer)
    layer.register_parameter("weight", nn.Parameter(torch.zeros(1), requires_grad=False))
    layer.register_parameter("weight_scale", nn.Parameter(torch.zeros(1), requires_grad=False))
    table, scale = Mock(), Mock()
    layer._host_table, layer._host_scale = table, scale
    scale.close.side_effect = [RuntimeError("unregister failed"), None]
    with pytest.raises(RuntimeError, match="unregister failed"):
        layer.close_host_offload()
    assert layer._host_scale is scale
    table.close.assert_not_called()
    layer.close_host_offload()
    assert layer.weight is None and layer.weight_scale is None
    assert layer._host_table is None and layer._host_scale is None
    layer.close_host_offload()
    table.close.assert_called_once()


@pytest.mark.parametrize("rank", [0, 1])
def test_shared_host_scale_loader_uses_owner_only(monkeypatch, rank):
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

    from vllm_ascend.models.qwen4_exp import ple_embedding

    layer = ple_embedding.AscendPLEHostEmbedding.__new__(ple_embedding.AscendPLEHostEmbedding)
    nn.Module.__init__(layer)
    layer._shared_group = SimpleNamespace(rank_in_group=rank, world_size=2, cpu_group=object())
    loader = Mock()
    monkeypatch.setattr(Qwen4ExpPLEEmbedding, "weight_loader", loader)
    collective = Mock()
    monkeypatch.setattr(ple_embedding.dist, "all_gather_object", collective)
    value = torch.ones(2, 5)
    layer.weight_loader(value, value, checkpoint_start=3)
    assert loader.call_count == (1 if rank == 0 else 0)
    collective.assert_called_once()


@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("fail_scale", [False, True])
def test_host_constructor_preserves_partial_quantized_buffers(monkeypatch, mxfp8, fail_scale):
    from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding

    from vllm_ascend.models.host_offload import track_host_offload_owners
    from vllm_ascend.models.qwen4_exp import ple_embedding

    method = AscendPLEQuantizedEmbeddingMethod(mxfp8=mxfp8)
    failed_dtype = method.scale_dtype if fail_scale else method.weight_dtype
    owners = {}

    def allocate(shape, dtype, device, buffer_name, module):
        assert buffer_name == ("_host_scale" if dtype == method.scale_dtype else "_host_table")
        assert isinstance(module, nn.Module)
        owner = Mock(tensor=torch.zeros(shape, dtype=dtype))
        owners[dtype] = owner
        if dtype == failed_dtype:
            raise ple_embedding.HostUvaBufferCleanupError(
                owner, ValueError("allocation failed"), RuntimeError("unregister failed")
            )
        return owner

    def initialize(self, *args, **kwargs):
        nn.Module.__init__(self)
        method.create_weights(self, 160, [2], 160, 2, torch.bfloat16)

    monkeypatch.setattr(ple_embedding, "HostUvaBuffer", allocate)
    monkeypatch.setattr(ple_embedding.torch.npu, "current_device", lambda: 0)
    monkeypatch.setattr(ple_embedding, "get_current_vllm_config", lambda: SimpleNamespace(engram_config=None))
    monkeypatch.setattr(ple_embedding, "get_etp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(ple_embedding, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(Qwen4ExpPLEEmbedding, "__init__", initialize)

    with track_host_offload_owners() as tracked:
        with pytest.raises(ple_embedding.HostUvaBufferCleanupError):
            ple_embedding.AscendPLEHostEmbedding(embedding_method=method, params_dtype=torch.bfloat16)
        assert len(tracked) == 1
        layer = tracked[0]
        assert layer._host_table is owners[method.weight_dtype]
        if fail_scale:
            assert layer._host_scale is owners[method.scale_dtype]
        layer.close_host_offload()
        assert layer._host_table is None and layer._host_scale is None
    for owner in owners.values():
        owner.close.assert_called_once()


@pytest.mark.parametrize("mxfp8", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_quantized_host_registers_distinct_pointer_buffers(monkeypatch, mxfp8, shared):
    """Sleep-mode restoration must preserve both the codes and scale addresses."""
    from vllm_ascend.models.qwen4_exp import ple_embedding

    method = AscendPLEQuantizedEmbeddingMethod(mxfp8=mxfp8)
    layer = object.__new__(ple_embedding.AscendPLEHostEmbedding)
    nn.Module.__init__(layer)
    layer._shared = None
    layer._quantized = True
    layer._weight_dtype = method.weight_dtype
    layer._scale_dtype = method.scale_dtype
    layer._shared_group = SimpleNamespace(rank_in_group=0, cpu_group=object()) if shared else None
    allocations: list[SimpleNamespace] = []

    def allocate(shape, dtype, device, *args):
        if shared:
            group, buffer_name, owner = args
            assert group is layer._shared_group
        else:
            buffer_name, owner = args
        pointer = torch.tensor([len(allocations) + 1], dtype=torch.int64)
        owner.register_buffer(f"{buffer_name}_ptrs", pointer, persistent=False)
        buffer = SimpleNamespace(tensor=torch.empty(shape, dtype=dtype), ptrs=pointer)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(ple_embedding, "HostUvaBuffer", allocate)
    monkeypatch.setattr(ple_embedding, "SharedUvaBuffer", allocate)
    monkeypatch.setattr(ple_embedding.torch.npu, "current_device", lambda: 0)
    barrier = Mock()
    monkeypatch.setattr(ple_embedding.dist, "barrier", barrier)
    layer.allocate_embedding_weight(2, 160, method.weight_dtype)
    layer.allocate_embedding_weight(2, 5, method.scale_dtype)
    pointers = dict(layer.named_buffers())
    assert pointers["_host_table_ptrs"] is allocations[0].ptrs
    assert pointers["_host_scale_ptrs"] is allocations[1].ptrs
    assert layer._host_table is allocations[0]
    assert layer._host_scale is allocations[1]
    assert not layer.state_dict()
    assert barrier.call_count == (2 if shared else 0)
