# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.linear import ReplicatedLinear, RowParallelLinear, UnquantizedLinearMethod

import vllm_ascend.attention.indexer_kpool as kpool
import vllm_ascend.attention.sfa_v1 as sfa
from vllm_ascend.attention.prolog_query import (
    mxfp8_query_scale_width,
    normalize_prolog_query,
    prolog_query_unsupported_reasons,
)
from vllm_ascend.attention.utils import PreprocessType, mark_fused_preprocess_weights
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import HardwareCapability, get_hardware_profile
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic, w8a8_mxfp8
from vllm_ascend.quantization.prepared_linear_input import PreparedLinearInput


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 4])
def test_prolog_slots_preserve_prefix_values_and_shared_metadata(dtype, num_tokens):
    slots = torch.tensor([42, 43, 128, -1], dtype=dtype)
    metadata = SimpleNamespace(slot_mapping=slots)
    before = slots.clone()
    first = sfa._prolog_kv_slots(slots, metadata, num_tokens)
    second = sfa._prolog_kv_slots(slots, metadata, 4)
    torch.testing.assert_close(first, before[:num_tokens].long())
    assert first.dtype == torch.int64
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert metadata.slot_mapping is slots
    torch.testing.assert_close(slots, before)
    if dtype == torch.int32:
        assert metadata.kv_slots_i64[0] is slots
        assert second is metadata.kv_slots_i64[1]
    else:
        assert second is slots


@pytest.mark.parametrize("slots,num_tokens", [(torch.ones(2), 3), (torch.ones(2, 1), 2), (torch.ones(2), -1)])
def test_prolog_slots_reject_incomplete_or_non_vector_mapping(slots, num_tokens):
    with pytest.raises(ValueError, match="one-dimensional slot mapping"):
        sfa._prolog_kv_slots(slots, SimpleNamespace(), num_tokens)


def test_prolog_slots_new_step_does_not_reuse_previous_int32_conversion():
    metadata = SimpleNamespace()
    old = sfa._prolog_kv_slots(torch.tensor([42, 43], dtype=torch.int32), metadata, 2)
    new = sfa._prolog_kv_slots(torch.tensor([128, -1], dtype=torch.int32), metadata, 2)
    torch.testing.assert_close(old, torch.tensor([42, 43]))
    torch.testing.assert_close(new, torch.tensor([128, -1]))


@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("actual_tokens", [1, 3])
@pytest.mark.parametrize("skip_topk", [False, True])
def test_mock_nope_prolog_forward_aligns_slots_and_preserves_raw_keys(
    monkeypatch, graph_mode, actual_tokens, skip_topk
):
    # This forces only the mock branch; production NoPE routing stays native.
    hidden = torch.arange(20, dtype=torch.float32).view(4, 5)
    slots = torch.tensor([42, 43, 128, -1], dtype=torch.int32)
    slots[actual_tokens:] = -1
    rows = 4 if graph_mode == CUDAGraphMode.FULL else actual_tokens
    metadata = SimpleNamespace(slot_mapping=slots, num_actual_tokens=actual_tokens, cos=None, sin=None)
    indexer_metadata = SimpleNamespace(slot_mapping=torch.tensor([7, 8, 9, -1]))
    cache = (torch.empty(1),)

    def preprocess(**kwargs):
        assert kwargs["hidden_states"].shape[0] == rows
        torch.testing.assert_close(kwargs["slot_mapping"], slots[:rows].long())
        return kwargs["hidden_states"], torch.zeros(rows, 1, 2), None, torch.zeros(rows, 2)

    impl = SimpleNamespace(
        qk_rope_head_dim=0,
        q_lora_rank=64,
        enable_sparse_sfa_turboquant=False,
        g_proj=None,
        layerwise_kv_cache_hook=None,
        layer_name="test.attn",
        preprocess_type=PreprocessType.PROLOG_V3,
        runtime_has_indexer=True,
        skip_topk=skip_topk,
        use_index_cache=False,
        _compose_sfa_kv_cache=lambda value: value,
        _get_sfa_kv_slot_mapping=lambda value: value.slot_mapping,
        _get_indexer_attn_metadata=lambda: indexer_metadata,
        _get_parallel_forward_context=lambda *args: SimpleNamespace(
            actual_seq_lengths_query=None,
            actual_seq_lengths_key=None,
            gather_full_o_proj=False,
            topk_num_tokens=rows,
        ),
        _sfa_preprocess_prolog_v3=MagicMock(side_effect=preprocess),
        _prepare_indexer_metadata=MagicMock(),
        _get_indexer_attn_q_gather_handle=lambda _: None,
        indexer=MagicMock(return_value=torch.zeros(rows, 1, 1, dtype=torch.int32)),
        _get_indexcache_topk_indices=MagicMock(return_value=torch.zeros(rows, 1, 1, dtype=torch.int32)),
        _execute_sparse_flash_attention_process=MagicMock(return_value=torch.zeros(rows, 2)),
        _v_up_proj=lambda value: value,
        _finalize_o_proj=MagicMock(),
    )
    monkeypatch.setattr(sfa, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=graph_mode))
    for name in ("wait_for_kv_layer_from_connector", "notify_kv_cache_written", "maybe_save_kv_layer_to_connector"):
        monkeypatch.setattr(sfa, name, MagicMock())
    monkeypatch.setattr(sfa, "attention_transfer_window", nullcontext)
    output = torch.empty(4, 2)
    assert sfa.AscendSFAImpl.forward(impl, impl.layer_name, hidden, cache, metadata, output) is output
    impl.indexer.assert_called_once()
    raw_keys = impl.indexer.call_args.args[2]
    assert raw_keys.data_ptr() == hidden.data_ptr()
    torch.testing.assert_close(raw_keys, hidden[:rows])
    assert impl.indexer.call_args.args[3] is indexer_metadata
    assert impl.indexer.call_args.kwargs["compute_topk"] is not skip_topk
    assert impl._get_indexcache_topk_indices.call_count == int(skip_topk)
    assert metadata.slot_mapping is slots


@pytest.mark.parametrize(
    "scheme_class", [w8a8_dynamic.AscendW8A8DynamicLinearMethod, w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod]
)
@pytest.mark.parametrize("consumer", [False, True])
def test_nope_still_uses_native_before_weight_preparation(scheme_class, consumer):
    scheme = scheme_class.__new__(scheme_class)
    if scheme_class is w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod:
        scheme.group_size = 32
    projection = SimpleNamespace(quant_method=SimpleNamespace(quant_method=scheme))
    impl = SimpleNamespace(
        qk_rope_head_dim=0,
        q_lora_rank=64,
        fused_qkv_a_proj=projection,
        q_proj=SimpleNamespace(),
        kv_a_layernorm=object(),
        q_a_layernorm=object(),
        has_indexer=True,
        indexer=SimpleNamespace(
            wq_b=SimpleNamespace(
                weight_scale=object(),
                input_size=64,
                quant_method=projection.quant_method,
                mxfp8_tp_padding=(0, 0),
            )
        ),
        is_kv_consumer=consumer,
        enable_mlapo=True,
        enable_sparse_sfa_turboquant=False,
        enable_sparse_sfa_c8=False,
        _get_layer_quant_method=sfa.AscendSFAImpl._get_layer_quant_method,
        _try_enable_type=MagicMock(),
    )
    impl._get_fused_type_unsupported_reasons = lambda kind: sfa.AscendSFAImpl._get_fused_type_unsupported_reasons(
        impl, kind
    )
    impl._fused_preprocess_type = lambda: sfa.AscendSFAImpl._fused_preprocess_type(impl)
    reasons = impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
    assert len(reasons) == 1 and "NoPE" in reasons[0]
    mark_fused_preprocess_weights(impl)
    assert projection._fused_preprocess_managed is False
    assert impl.q_proj._fused_preprocess_managed is False
    assert sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16) is PreprocessType.NATIVE
    impl._try_enable_type.assert_not_called()


class _QuantizedProjection(nn.Module):
    def __init__(self, scheme, weight, weight_scale):
        super().__init__()
        self.quant_method = SimpleNamespace(quant_method=scheme)
        self.weight = weight
        self.weight_scale = weight_scale
        self.input_size = weight.shape[0]
        self.mxfp8_tp_padding = (0, 0)
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value)
        return self.quant_method.quant_method.apply(self, value), None


class _FloatProjection(nn.Module):
    def __init__(self, width, dtype=torch.bfloat16):
        super().__init__()
        self.quant_method = UnquantizedLinearMethod()
        self.input_size = width
        self.weight = (torch.arange(4 * width).view(4, width) % 13 - 6).to(dtype) / 32
        self.inputs = []

    def forward(self, value):
        self.inputs.append(value)
        return torch.nn.functional.linear(value, self.weight), None


def _query_backend(kind):
    backend = kpool.Glm5NextKPoolIndexerBackend.__new__(kpool.Glm5NextKPoolIndexerBackend)
    nn.Module.__init__(backend)
    backend.q_lora_rank = 64
    backend.n_head, backend.head_dim = 2, 2
    values = (torch.arange(256).view(4, 64) % 7 - 3).float()
    weights = (torch.arange(256).view(64, 4) % 5 - 2).float()
    if kind == "int8":
        scheme = w8a8_dynamic.AscendW8A8DynamicLinearMethod()
        query = values.to(torch.int8)
        scales = torch.tensor([0.125, 0.25, 0.5, 1.0])
        weight = weights.to(torch.int8)
        weight_scale = torch.tensor([0.25, 0.5, 1.0, 2.0])
        expected = (values * scales[:, None]) @ (weights * weight_scale)
    else:
        scheme = w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod.__new__(w8a8_mxfp8.AscendW8A8MXFP8DynamicLinearMethod)
        scheme.group_size = 32
        query = values.to(torch.float8_e4m3fn)
        scales = torch.tensor([[126, 127], [127, 128], [128, 126], [125, 129]], dtype=torch.uint8)
        weight = weights.to(torch.float8_e4m3fn)
        weight_scale = torch.full((1, 4, 2), 127, dtype=torch.uint8)
        factors = torch.pow(2.0, scales.float() - 127).repeat_interleave(32, dim=1)
        expected = (values * factors) @ weights
    backend.wq_b = _QuantizedProjection(scheme, weight, weight_scale)
    return backend, query, scales, expected


@pytest.mark.parametrize("kind", ["int8", "mx"])
@pytest.mark.parametrize("scale_layout", ["flat", "expanded"])
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 4])
def test_kpool_prepared_query_reuses_scheme_without_requantization(
    monkeypatch, kind, scale_layout, output_dtype, num_tokens
):
    backend, query, scales, expected = _query_backend(kind)
    if scale_layout == "expanded":
        scales = scales[:, None] if kind == "int8" else scales.view(4, 1, 2)
    original_query, original_scales = query.clone(), scales.clone()

    def matmul(data, weight, weight_scale, *, pertoken_scale, output_dtype, **kwargs):
        if kind == "int8":
            assert pertoken_scale.shape == (num_tokens,)
            output = (data.float() * pertoken_scale[:, None]) @ (weight.float() * weight_scale)
        else:
            assert pertoken_scale.shape == (num_tokens, 1, 2)
            assert kwargs["group_sizes"] == [1, 1, 32]
            factors = torch.pow(2.0, pertoken_scale.view(num_tokens, 2).float() - 127).repeat_interleave(32, dim=1)
            output = (data.float() * factors) @ weight.float()
        return output.to(output_dtype)

    npu = w8a8_dynamic.torch_npu
    mm = MagicMock(side_effect=matmul)
    quant = MagicMock(side_effect=AssertionError("prepared q_c must not be quantized twice"))
    monkeypatch.setattr(npu, "npu_quant_matmul", mm)
    monkeypatch.setattr(npu, "npu_dynamic_quant", quant)
    monkeypatch.setattr(npu, "npu_dynamic_mx_quant", quant, raising=False)
    result = backend._project_query((query, scales), num_tokens, output_dtype)
    torch.testing.assert_close(result, expected[:num_tokens].to(output_dtype).view(num_tokens, 2, 2))
    assert result.dtype == output_dtype
    prepared = backend.wq_b.inputs[0]
    assert isinstance(prepared, PreparedLinearInput)
    assert prepared.quantized.shape == (num_tokens, 64)
    assert prepared.quantized.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()
    assert prepared.scale.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr()
    torch.testing.assert_close(query.float(), original_query.float())
    torch.testing.assert_close(scales, original_scales)
    mm.assert_called_once()
    quant.assert_not_called()


def test_kpool_float_query_keeps_existing_projection():
    backend, query, _, _ = _query_backend("int8")
    backend.wq_b.forward = MagicMock(return_value=(torch.arange(12).view(3, 4).float(), None))
    result = backend._project_query(query.float(), 3, torch.float16)
    torch.testing.assert_close(backend.wq_b.forward.call_args.args[0], query[:3].float())
    torch.testing.assert_close(result, torch.arange(12).view(3, 2, 2).float())


@pytest.mark.parametrize(
    "case,error",
    [
        ("row_mismatch", ValueError),
        ("short_input", ValueError),
        ("wrong_rank", ValueError),
        ("int8_scale_shape", ValueError),
        ("int8_scale_dtype", TypeError),
        ("wrong_query_dtype", TypeError),
        ("unsupported_scheme", ValueError),
        ("mx_scale_shape", ValueError),
        ("mx_group_size", ValueError),
        ("mx_tp_padding", ValueError),
        ("mx_scale_dtype", TypeError),
        ("pair_length", TypeError),
    ],
)
def test_kpool_rejects_incompatible_prequantized_inputs(case, error):
    backend, query, scales, _ = _query_backend("mx" if case.startswith("mx_") else "int8")
    rows = 3
    if case == "row_mismatch":
        scales = scales[:3]
    elif case == "short_input":
        rows = 5
    elif case == "wrong_rank":
        query = query[:, :32]
    elif case == "int8_scale_shape":
        scales = scales[:, None].repeat(1, 2)
    elif case == "int8_scale_dtype":
        scales = scales.half()
    elif case == "wrong_query_dtype":
        query = query.float()
    elif case == "unsupported_scheme":
        backend.wq_b.quant_method.quant_method = object()
    elif case == "mx_scale_shape":
        scales = scales[:, :1]
    elif case == "mx_group_size":
        backend.wq_b.quant_method.quant_method.group_size = 64
    elif case == "mx_tp_padding":
        backend.wq_b.mxfp8_tp_padding = (0, 32)
    elif case == "mx_scale_dtype":
        scales = scales.float()
    inputs = (query,) if case == "pair_length" else (query, scales)
    with pytest.raises(error, match="KPool"):
        backend._project_query(inputs, rows, torch.bfloat16)
    assert backend.wq_b.inputs == []


@pytest.mark.parametrize("graph_mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("compute_topk", [False, True])
def test_kpool_forward_keeps_cache_update_when_topk_is_skipped(monkeypatch, graph_mode, compute_topk):
    backend, query, scales, _ = _query_backend("int8")
    rows = 4 if graph_mode == CUDAGraphMode.FULL else 3
    hidden = torch.ones(4, 3, dtype=torch.bfloat16)
    keys = hidden + 5
    backend._key_weight = torch.ones(2, 3)
    backend._head_weight_f32 = torch.ones(2, 3)
    backend.softmax_scale = 0.5
    backend.k_cache = SimpleNamespace(prefix="k", kv_cache=torch.zeros(1))
    backend.tail_cache = SimpleNamespace(prefix="tail", kv_cache=torch.zeros(1))
    backend._project_query = MagicMock(return_value=torch.ones(rows, 2, 2, dtype=hidden.dtype))
    metadata = kpool.AscendIndexerKPoolMetadata(
        block_table=torch.zeros(1, 1),
        slot_mapping=torch.zeros(4),
        seq_lens=torch.ones(1),
        positions=torch.arange(4),
        block_size=128,
        compress_ratio=4,
        num_actual_tokens=3,
    )
    tail = kpool.AscendIndexerKPoolTailMetadata(
        block_table=torch.zeros(1, 1),
        slot_mapping=torch.zeros(4),
        block_size=4,
    )
    monkeypatch.setattr(
        kpool,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"tail": tail},
            cudagraph_runtime_mode=graph_mode,
        ),
    )

    def update(k_hidden, q_values, weights, cache, tail_cache, *args, **kwargs):
        torch.testing.assert_close(k_hidden, keys[:rows])
        assert k_hidden.dtype == keys.dtype
        assert kwargs["compute_topk"] is compute_topk
        assert (q_values is None) is not compute_topk
        assert (weights is None) is not compute_topk
        cache.fill_(7)
        tail_cache.fill_(11)

    backend._forward_cann = MagicMock(side_effect=update)
    backend.forward(hidden, (query, scales), keys, metadata, compute_topk=compute_topk)
    backend._forward_cann.assert_called_once()
    assert backend._project_query.call_count == int(compute_topk)
    if compute_topk:
        assert backend._project_query.call_args.args[1:] == (rows, hidden.dtype)
    assert backend.k_cache.kv_cache.item() == 7
    assert backend.tail_cache.kv_cache.item() == 11


def _routing_impl(kind="mx", rank=64):
    backend, _, _, _ = _query_backend(kind)
    backend.wq_b.input_size = rank
    main = SimpleNamespace(quant_method=backend.wq_b.quant_method)
    impl = SimpleNamespace(
        qk_rope_head_dim=64,  # Exercise the live Prolog gate without enabling NoPE.
        q_lora_rank=rank,
        fused_qkv_a_proj=main,
        q_proj=SimpleNamespace(),
        kv_a_layernorm=object(),
        q_a_layernorm=object(),
        has_indexer=True,
        indexer=backend,
        is_kv_consumer=False,
        enable_mlapo=False,
        enable_sparse_sfa_turboquant=False,
        enable_sparse_sfa_c8=False,
        _get_layer_quant_method=sfa.AscendSFAImpl._get_layer_quant_method,
        _process_weights_for_fused_prolog_v3=MagicMock(),
        _process_weights_for_fused_mlapo=MagicMock(),
    )
    impl._get_fused_type_unsupported_reasons = lambda kind: sfa.AscendSFAImpl._get_fused_type_unsupported_reasons(
        impl, kind
    )
    impl._fused_preprocess_type = lambda: sfa.AscendSFAImpl._fused_preprocess_type(impl)
    impl._try_enable_type = lambda kind, dtype: sfa.AscendSFAImpl._try_enable_type(impl, kind, dtype)
    return impl


@pytest.mark.parametrize("kind", ["int8", "mx"])
@pytest.mark.parametrize("rank", [64, 96, 1536])
def test_supported_query_keeps_preload_and_postload_plan_consistent(kind, rank):
    impl = _routing_impl(kind, rank)
    mark_fused_preprocess_weights(impl)
    assert impl._fused_preprocess_planned_type is PreprocessType.PROLOG_V3
    assert impl.fused_qkv_a_proj._fused_preprocess_managed is True
    assert sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16) is PreprocessType.PROLOG_V3
    impl._process_weights_for_fused_prolog_v3.assert_called_once()


@pytest.mark.parametrize(
    "case",
    [
        "scheme",
        "mixed_encoding",
        "group_size",
        "padding",
        "unknown_padding",
        "invalid_rank",
        "input_width",
    ],
)
def test_invalid_query_configuration_selects_native_before_weight_conversion(case):
    impl = _routing_impl()
    projection = impl.indexer.wq_b
    if case == "scheme":
        # Presence of weight_scale alone must not admit this projection.
        projection.quant_method = SimpleNamespace(quant_method=object())
    elif case == "mixed_encoding":
        projection.quant_method = SimpleNamespace(quant_method=w8a8_dynamic.AscendW8A8DynamicLinearMethod())
    elif case == "group_size":
        projection.quant_method.quant_method.group_size = 64
    elif case == "padding":
        projection.mxfp8_tp_padding = (0, 32)
    elif case == "unknown_padding":
        del projection.mxfp8_tp_padding
    elif case == "invalid_rank":
        impl.q_lora_rank = 0
    elif case == "input_width":
        projection.input_size = 96
    reasons = impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
    assert reasons
    mark_fused_preprocess_weights(impl)
    assert impl._fused_preprocess_planned_type is None
    assert impl.fused_qkv_a_proj._fused_preprocess_managed is False
    assert impl.q_proj._fused_preprocess_managed is False
    assert sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16) is PreprocessType.NATIVE
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_native_plan_cannot_silently_switch_to_prolog_after_loading():
    impl = _routing_impl()
    projection = impl.indexer.wq_b
    del projection.mxfp8_tp_padding
    mark_fused_preprocess_weights(impl)
    projection.mxfp8_tp_padding = (0, 0)
    assert impl._fused_preprocess_type() is PreprocessType.PROLOG_V3
    assert sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16) is PreprocessType.NATIVE
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_changed_fused_plan_fails_before_weight_conversion_instead_of_falling_back():
    impl = _routing_impl()
    mark_fused_preprocess_weights(impl)
    impl.indexer.wq_b.quant_method.quant_method.group_size = 64
    with pytest.raises(RuntimeError, match="eligibility changed after weight management"):
        sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16)
    impl._process_weights_for_fused_prolog_v3.assert_not_called()
    with pytest.raises(RuntimeError, match="fused weight plan is no longer supported"):
        impl._try_enable_type(PreprocessType.PROLOG_V3, torch.bfloat16)


def test_weight_conversion_failure_is_not_swallowed_as_native_fallback():
    impl = _routing_impl()
    mark_fused_preprocess_weights(impl)
    impl._process_weights_for_fused_prolog_v3.side_effect = RuntimeError("weight conversion failed")
    with pytest.raises(RuntimeError, match="weight conversion failed"):
        sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16)
    impl._process_weights_for_fused_prolog_v3.assert_called_once()
    impl._process_weights_for_fused_mlapo.assert_not_called()


@pytest.mark.parametrize(
    "layer_class,width,tp_rank,expected_supported",
    [
        (ReplicatedLinear, 96, 0, True),
        (RowParallelLinear, 64, 1, True),
        (RowParallelLinear, 48, 1, False),
    ],
)
def test_mxfp8_padding_is_inferred_before_loading(layer_class, width, tp_rank, expected_supported):
    backend, _, _, _ = _query_backend("mx")
    projection = layer_class.__new__(layer_class)
    nn.Module.__init__(projection)
    projection.input_size = width
    projection.input_size_per_partition = width
    projection.tp_rank = tp_rank
    projection.input_is_parallel = True
    projection.quant_method = backend.wq_b.quant_method
    projection.weight_scale = torch.ones(1)
    assert not hasattr(projection, "mxfp8_tp_padding")
    reasons = prolog_query_unsupported_reasons(projection, width)
    assert (not reasons) is expected_supported
    if expected_supported:
        projection.mxfp8_tp_padding = (0, 0)
        assert not prolog_query_unsupported_reasons(projection, width)
    else:
        # A stale post-load attribute must not override the inferred padding.
        projection.mxfp8_tp_padding = (0, 0)
        assert prolog_query_unsupported_reasons(projection, width)


@pytest.mark.parametrize("rank,width", [(32, 2), (64, 2), (96, 4), (128, 4), (1536, 48)])
def test_mxfp8_scale_width_does_not_impose_a_64_multiple_on_rank(rank, width):
    assert mxfp8_query_scale_width(rank) == width
    q = torch.zeros(4, 1, rank, dtype=torch.float8_e4m3fn)
    scales = torch.full((4, width // 2, 2), 127, dtype=torch.uint8)
    actual_q, actual_scale = normalize_prolog_query(q, scales, 4, rank, mxfp8=True)
    assert actual_q.shape == (4, rank)
    assert actual_scale.shape == (4, width)
    assert actual_q.data_ptr() == q.data_ptr()
    assert actual_scale.data_ptr() == scales.data_ptr()


@pytest.mark.parametrize(
    "case",
    [
        "missing_query",
        "short_query",
        "wrong_query_width",
        "wrong_query_rows",
        "short_scale",
        "narrow_scale",
        "missing_scale",
    ],
)
def test_prolog_output_rejects_incomplete_full_bucket_before_indexer(case):
    query = torch.zeros(4, 64, dtype=torch.float8_e4m3fn)
    scale = torch.ones(4, 2, dtype=torch.uint8)
    if case == "missing_query":
        query = None
    elif case == "short_query":
        query = query[:3]
    elif case == "wrong_query_width":
        query = query.view(4, 2, 32)
    elif case == "wrong_query_rows":
        query = query.view(2, 2, 64)
    elif case == "short_scale":
        scale = scale[:3]
    elif case == "narrow_scale":
        scale = scale[:, :1]
    elif case == "missing_scale":
        scale = None
    with pytest.raises(RuntimeError, match="Prolog"):
        normalize_prolog_query(query, scale, 4, 64, mxfp8=True)


@pytest.mark.parametrize("num_tokens", [0, 1, 4])
def test_prolog_output_accepts_float_and_per_token_scale_contracts(num_tokens):
    query = torch.zeros(num_tokens, 64, dtype=torch.bfloat16)
    result = normalize_prolog_query(query, None, num_tokens, 64, mxfp8=False)
    assert result.shape == (num_tokens, 64)
    int8_query = query.to(torch.int8)
    scales = torch.ones(num_tokens, 1)
    _, result_scales = normalize_prolog_query(int8_query, scales, num_tokens, 64, mxfp8=False)
    assert result_scales.shape == (num_tokens,)


@pytest.mark.parametrize(
    "mxfp8,query_dtype,scale_dtype",
    [
        (False, torch.bfloat16, torch.float32),
        (False, torch.float8_e4m3fn, torch.float32),
        (False, torch.int8, torch.float16),
        (True, torch.int8, torch.uint8),
        (True, torch.bfloat16, torch.uint8),
        (True, torch.float8_e4m3fn, torch.float32),
    ],
)
def test_prolog_output_rejects_mismatched_quantization_encoding(mxfp8, query_dtype, scale_dtype):
    query = torch.zeros(4, 64, dtype=query_dtype)
    scale = torch.zeros(4, 2 if mxfp8 else 1, dtype=scale_dtype)
    with pytest.raises(RuntimeError, match="Prolog.*requires"):
        normalize_prolog_query(query, scale, 4, 64, mxfp8=mxfp8)


@pytest.mark.parametrize("mxfp8", [False, True])
def test_prolog_output_rejects_cross_device_scale(mxfp8):
    query = torch.zeros(4, 64, dtype=torch.float8_e4m3fn if mxfp8 else torch.int8)
    scale = torch.empty(4, 2 if mxfp8 else 1, dtype=torch.uint8 if mxfp8 else torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="Prolog.*same device"):
        normalize_prolog_query(query, scale, 4, 64, mxfp8=mxfp8)


@pytest.mark.parametrize("returned_rows", [3, 4])
def test_real_prolog_adapter_checks_returned_query_rows(monkeypatch, returned_rows):
    # Run the actual wrapper, replacing only the device operator. No NoPE ABI
    # assumptions are introduced: this uses the existing RoPE operator inputs.
    norm = SimpleNamespace(weight=torch.ones(64), variance_epsilon=1e-5)
    impl = SimpleNamespace(
        _quant_type=None,
        enable_sparse_sfa_c8=False,
        q_a_layernorm=norm,
        kv_a_layernorm=norm,
        weight_dq=torch.ones(1),
        weight_uq_qr=torch.ones(1),
        W_UK_T=torch.ones(1),
        weight_dkv_kr=torch.ones(1),
        has_indexer=True,
        local_num_heads=1,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=2,
    )
    op = MagicMock(
        return_value=(
            torch.zeros(4, 1, 64),
            torch.zeros(4, 1, 2),
            None,
            torch.zeros(returned_rows, 64, dtype=torch.bfloat16),
            None,
        )
    )
    monkeypatch.setattr(sfa.torch_npu, "npu_mla_prolog_v3", op, raising=False)
    args = (torch.zeros(4, 64), (torch.zeros(1), torch.zeros(1)), torch.zeros(4, 2), torch.zeros(4, 2), torch.arange(4))
    if returned_rows == 3:
        with pytest.raises(RuntimeError, match="including graph padding"):
            sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(impl, *args)
    else:
        assert sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(impl, *args)[-1].shape == (4, 64)
    op.assert_called_once()


def _nope_routing_impl():
    impl = _routing_impl("int8", rank=1536)
    impl.qk_rope_head_dim = 0
    impl.qk_nope_head_dim = 256
    impl.kv_lora_rank = 512
    impl.enable_nope_prolog = True
    impl.enable_sparse_sfa_c8 = True
    impl.c8_cache_dtype = torch.int8
    impl.sfa_qsfa_tile_size = 128
    impl.q_proj.quant_method = impl.fused_qkv_a_proj.quant_method
    impl.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1)
    )
    return impl


@pytest.mark.parametrize("family", list(AscendDeviceType))
def test_nope_hardware_contract_matches_operator_tiling(family):
    profile = get_hardware_profile(family)
    assert profile.supports(HardwareCapability.NOPE_PROLOG_W8A8_C8) == (
        family in (AscendDeviceType.A2, AscendDeviceType.A3)
    )


@pytest.mark.parametrize("family", [AscendDeviceType.A2, AscendDeviceType.A3])
@pytest.mark.parametrize("rank", [1536, 2048])
def test_nope_opt_in_selects_prolog_before_weight_loading(monkeypatch, family, rank):
    monkeypatch.setattr(sfa, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    impl = _nope_routing_impl()
    impl.q_lora_rank = impl.indexer.wq_b.input_size = rank
    mark_fused_preprocess_weights(impl)
    assert impl._fused_preprocess_planned_type is PreprocessType.PROLOG_V3
    assert impl.fused_qkv_a_proj._fused_preprocess_managed
    assert impl.q_proj._fused_preprocess_managed
    assert sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16) is PreprocessType.PROLOG_V3
    impl._process_weights_for_fused_prolog_v3.assert_called_once()
    assert impl._get_fused_type_unsupported_reasons(PreprocessType.MLAPO)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rank", [1536, 2048])
def test_nope_float_indexer_selects_prolog_without_weight_scale(monkeypatch, dtype, rank):
    monkeypatch.setattr(sfa, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3))
    impl = _nope_routing_impl()
    impl.q_lora_rank = rank
    impl.indexer.wq_b = _FloatProjection(rank, dtype)
    weight = impl.indexer.wq_b.weight
    assert not hasattr(impl.indexer.wq_b, "weight_scale")
    assert not impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
    mark_fused_preprocess_weights(impl)
    assert impl._fused_preprocess_planned_type is PreprocessType.PROLOG_V3
    assert impl.fused_qkv_a_proj._fused_preprocess_managed
    assert impl.q_proj._fused_preprocess_managed
    assert sfa.AscendSFAImpl._resolve_preprocess_type(impl, dtype) is PreprocessType.PROLOG_V3
    impl._process_weights_for_fused_prolog_v3.assert_called_once()
    assert impl.indexer.wq_b.weight is weight


@pytest.mark.parametrize("invalid", ["method", "missing_method", "dtype", "width", "partition", "split_input"])
def test_nope_float_indexer_invalid_contract_still_falls_back(monkeypatch, invalid):
    monkeypatch.setattr(sfa, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3))
    impl = _nope_routing_impl()
    projection = _FloatProjection(impl.q_lora_rank)
    impl.indexer.wq_b = projection
    if invalid == "method":
        projection.quant_method = SimpleNamespace(quant_method=object())
    elif invalid == "missing_method":
        projection.quant_method = None
    elif invalid == "dtype":
        projection.weight = projection.weight.to(torch.int8)
    elif invalid == "width":
        projection.input_size = 64
    elif invalid == "partition":
        projection.input_size_per_partition = 768
    else:
        projection.input_is_parallel = False
    assert impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
    mark_fused_preprocess_weights(impl)
    assert not impl.fused_qkv_a_proj._fused_preprocess_managed
    assert sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16) is PreprocessType.NATIVE
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


def test_float_indexer_support_does_not_relax_rope_or_prepared_query_contracts():
    impl = _routing_impl("int8", rank=64)
    impl.indexer.wq_b = _FloatProjection(64)
    assert impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
    query = torch.zeros(4, 64, dtype=torch.int8)
    with pytest.raises(ValueError, match="KPool query configuration"):
        impl.indexer._project_query((query, torch.ones(4)), 4, torch.bfloat16)
    assert not impl.indexer.wq_b.inputs


def test_nope_admission_accepts_unquantized_method_subclasses(monkeypatch):
    # Ascend's FLOAT method subclasses vLLM's unquantized linear method.
    class FloatMethod(UnquantizedLinearMethod):
        pass

    monkeypatch.setattr(sfa, "get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A3))
    impl = _nope_routing_impl()
    impl.indexer.wq_b = _FloatProjection(impl.q_lora_rank)
    impl.indexer.wq_b.quant_method = FloatMethod()
    assert not impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)


@pytest.mark.parametrize(
    "invalid",
    [
        "disabled",
        "hardware",
        "mx_weights",
        "q_proj",
        "float_cache",
        "fp8_cache",
        "D",
        "kv_rank",
        "q_rank",
        "tile",
        "pcp",
        "dcp",
    ],
)
def test_nope_unsupported_configuration_stays_native_before_conversion(monkeypatch, invalid):
    impl = _nope_routing_impl()
    family = AscendDeviceType.A5 if invalid == "hardware" else AscendDeviceType.A2
    monkeypatch.setattr(sfa, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    if invalid == "disabled":
        impl.enable_nope_prolog = False
    elif invalid == "mx_weights":
        impl.fused_qkv_a_proj.quant_method = _routing_impl("mx").fused_qkv_a_proj.quant_method
    elif invalid == "q_proj":
        impl.q_proj.quant_method = None
    elif invalid == "float_cache":
        impl.enable_sparse_sfa_c8 = False
    elif invalid == "fp8_cache":
        impl.c8_cache_dtype = torch.float8_e4m3fn
    elif invalid == "D":
        impl.qk_nope_head_dim = 128
    elif invalid == "kv_rank":
        impl.kv_lora_rank = 256
    elif invalid == "q_rank":
        impl.q_lora_rank = impl.indexer.wq_b.input_size = 64
    elif invalid == "tile":
        impl.sfa_qsfa_tile_size = 64
    elif invalid in ("pcp", "dcp"):
        field = "prefill_context_parallel_size" if invalid == "pcp" else "decode_context_parallel_size"
        setattr(impl.vllm_config.parallel_config, field, 2)
    assert impl._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3)
    mark_fused_preprocess_weights(impl)
    assert impl._fused_preprocess_planned_type is None
    assert sfa.AscendSFAImpl._resolve_preprocess_type(impl, torch.bfloat16) is PreprocessType.NATIVE
    impl._process_weights_for_fused_prolog_v3.assert_not_called()


def _nope_operator_case(monkeypatch, rows=4):
    impl = _nope_routing_impl()
    impl._quant_type = w8a8_dynamic.AscendW8A8DynamicLinearMethod
    impl.local_num_heads = 2
    for name in ("q_a_layernorm", "kv_a_layernorm"):
        setattr(impl, name, SimpleNamespace(weight=torch.ones(1, dtype=torch.bfloat16), variance_epsilon=1e-5))
    for name in (
        "weight_dq",
        "weight_uq_qr",
        "W_UK_T",
        "weight_dkv_kr",
        "dequant_scale_w_dq",
        "dequant_scale_w_uq_qr",
        "dequant_scale_w_dkv_kr",
    ):
        setattr(impl, name, torch.ones(1))
    impl.sfa_qsfa_k_nope_clip_alpha = torch.ones(1)
    impl.sfa_qsfa_kr_cache_dummy = torch.empty(0, dtype=torch.bfloat16)
    hidden = torch.randn(rows, 8, dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 1, 528, dtype=torch.int8)
    slots = torch.arange(rows, dtype=torch.int64)
    if rows:
        slots[-1] = -1
    query = torch.zeros(rows, 2, 512, dtype=torch.bfloat16)
    rope = torch.empty(rows, 2, 0, dtype=torch.bfloat16)
    qc = torch.zeros(rows, 1536, dtype=torch.int8)
    scales = torch.ones(rows, 1)
    op = MagicMock(return_value=(query, rope, None, qc, scales))
    quant = MagicMock(return_value=(torch.zeros_like(hidden, dtype=torch.int8), torch.ones(rows)))
    monkeypatch.setattr(sfa.torch_npu, "npu_mla_prolog_v3", op, raising=False)
    monkeypatch.setattr(sfa.torch_npu, "npu_dynamic_quant", quant)
    return impl, hidden, cache, slots, op


@pytest.mark.parametrize("rows", [0, 1, 4])
@pytest.mark.parametrize("input_rope", [None, "empty", "legacy"])
def test_nope_adapter_calls_operator_with_zero_width_rope_and_c8(monkeypatch, rows, input_rope):
    impl, hidden, cache, slots, op = _nope_operator_case(monkeypatch, rows)
    rope = None if input_rope is None else torch.ones(rows, 0 if input_rope == "empty" else 64)
    raw, query, qrope, (qc, scales) = sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(
        impl, hidden, (cache,), rope, rope, slots
    )
    assert raw is hidden
    assert query.shape == (rows, 2, 512)
    assert qrope.shape == (rows, 2, 0)
    assert qc.shape == (rows, 1536) and scales.shape == (rows,)
    args = op.call_args.kwargs
    assert args["rope_sin"].shape == args["rope_cos"].shape == (rows, 0)
    assert args["rope_sin"].dtype == hidden.dtype
    assert args["kr_cache"] is impl.sfa_qsfa_kr_cache_dummy
    assert args["kv_cache"] is cache
    torch.testing.assert_close(args["cache_index"], slots)
    for key, value in {
        "weight_quant_mode": 2,
        "kv_cache_quant_mode": 3,
        "query_quant_mode": 0,
        "ckvkr_repo_mode": 1,
        "quant_scale_repo_mode": 1,
        "tile_size": 128,
    }.items():
        assert args[key] == value
    op.assert_called_once()


@pytest.mark.parametrize("rows,valid_rows", [(0, 0), (1, 1), (4, 1), (4, 3), (4, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scale_ndim", [1, 2])
def test_nope_prolog_dequantized_query_reaches_float_kpool_projection(monkeypatch, rows, valid_rows, dtype, scale_ndim):
    impl, hidden, cache, slots, op = _nope_operator_case(monkeypatch, rows)
    hidden = hidden.to(dtype)
    backend = impl.indexer
    backend.q_lora_rank = impl.q_lora_rank
    backend.wq_b = _FloatProjection(impl.q_lora_rank, dtype)
    weight_before = backend.wq_b.weight.clone()
    values = list(op.return_value)
    quantized = (torch.arange(rows * impl.q_lora_rank).view(rows, impl.q_lora_rank) % 255 - 127).to(torch.int8)
    scales = torch.tensor([0.00311, 0.417, 1.1, 2.7])[:rows]
    values[3], values[4] = quantized, scales if scale_ndim == 1 else scales[:, None]
    op.return_value = tuple(values)
    before_q, before_scale = quantized.clone(), scales.clone()

    raw, query, rope, qc = sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(impl, hidden, (cache,), None, None, slots)
    assert raw is hidden
    assert query.shape == (rows, 2, 512) and rope.shape == (rows, 2, 0)
    assert isinstance(qc, torch.Tensor) and qc.dtype == dtype and qc.shape == (rows, impl.q_lora_rank)
    # A double-precision reference detects incorrect row broadcasting, missing
    # scales, and casts performed before rather than after FP32 multiplication.
    expected_q = (quantized.double() * scales.double()[:, None]).to(dtype)
    torch.testing.assert_close(qc, expected_q)
    result = backend._project_query(qc, valid_rows, dtype)
    expected = torch.nn.functional.linear(expected_q[:valid_rows], weight_before).view(valid_rows, 2, 2)
    torch.testing.assert_close(result, expected)
    assert backend.wq_b.inputs[0].shape == (valid_rows, impl.q_lora_rank)
    torch.testing.assert_close(quantized, before_q)
    torch.testing.assert_close(scales, before_scale)
    torch.testing.assert_close(backend.wq_b.weight, weight_before)
    op.assert_called_once()


@pytest.mark.parametrize("invalid", ["missing_scale", "scale_dtype", "query_dtype", "scale_device"])
def test_float_indexer_does_not_bypass_prolog_query_encoding_checks(monkeypatch, invalid):
    impl, hidden, cache, slots, op = _nope_operator_case(monkeypatch)
    impl.indexer.wq_b = _FloatProjection(impl.q_lora_rank)
    values = list(op.return_value)
    if invalid == "missing_scale":
        values[4] = None
    elif invalid == "scale_dtype":
        values[4] = values[4].bfloat16()
    elif invalid == "query_dtype":
        values[3] = values[3].float()
    else:
        values[4] = values[4].to("meta")
    op.return_value = tuple(values)
    with pytest.raises(RuntimeError, match="missing its scale|INT8 query|same device"):
        sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(impl, hidden, (cache,), None, None, slots)
    assert not impl.indexer.wq_b.inputs


def test_float_indexer_accepts_already_floating_prolog_query(monkeypatch):
    impl, hidden, cache, slots, op = _nope_operator_case(monkeypatch)
    impl.indexer.wq_b = _FloatProjection(impl.q_lora_rank)
    values = list(op.return_value)
    values[3] = torch.randn(hidden.shape[0], impl.q_lora_rank, dtype=hidden.dtype)
    values[4] = None
    op.return_value = tuple(values)
    qc = sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(impl, hidden, (cache,), None, None, slots)[-1]
    torch.testing.assert_close(qc, values[3])
    assert qc.data_ptr() == values[3].data_ptr()


@pytest.mark.parametrize("floating_indexer", [False, True])
@pytest.mark.parametrize(
    "invalid", ["q_rows", "q_width", "rope_none", "rope_rows", "rope_width", "qc_rows", "scale_rows"]
)
def test_nope_output_must_cover_full_bucket(monkeypatch, invalid, floating_indexer):
    impl, hidden, cache, slots, op = _nope_operator_case(monkeypatch)
    if floating_indexer:
        impl.indexer.wq_b = _FloatProjection(impl.q_lora_rank)
    values = list(op.return_value)
    if invalid == "q_rows":
        values[0] = values[0][:3]
    elif invalid == "q_width":
        values[0] = values[0].view(4, 4, 256)
    elif invalid == "rope_none":
        values[1] = None
    elif invalid == "rope_rows":
        values[1] = values[1][:3]
    elif invalid == "rope_width":
        values[1] = torch.zeros(4, 2, 64)
    elif invalid == "qc_rows":
        values[3] = values[3][:3]
    elif invalid == "scale_rows":
        values[4] = values[4][:3]
    op.return_value = tuple(values)
    with pytest.raises(RuntimeError, match="full-bucket|graph padding|scale must cover"):
        sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(impl, hidden, (cache,), None, None, slots)
    op.assert_called_once()


@pytest.mark.parametrize(
    "shape,dtype", [((1, 16, 1, 512), torch.int8), ((1, 16, 1, 528), torch.bfloat16), ((16, 528), torch.int8)]
)
def test_nope_invalid_cache_rejected_before_operator_write(monkeypatch, shape, dtype):
    impl, hidden, _, slots, op = _nope_operator_case(monkeypatch)
    with pytest.raises(RuntimeError, match="INT8"):
        sfa.AscendSFAImpl._sfa_preprocess_prolog_v3(impl, hidden, (torch.zeros(shape, dtype=dtype),), None, None, slots)
    op.assert_not_called()
