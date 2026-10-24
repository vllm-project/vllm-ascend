import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


@pytest.mark.parametrize("query_lengths", [[1], [1, 1], [3, 2]])
@pytest.mark.parametrize("weights_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("cache_indices", [(0, 1), (1, 0)])
@pytest.mark.parametrize("query_dtype", [torch.float8_e4m3fn, torch.int8])
def test_a5_c8_indexer_uses_v2_quantization_contract(query_lengths, weights_dtype, cache_indices, query_dtype):
    tokens, heads, dim = sum(query_lengths), 32, 128
    scale_dtype = torch.float16 if query_dtype == torch.int8 else torch.float32
    quant_mode = 2 if query_dtype == torch.int8 else 1
    query = torch.zeros(tokens * heads, dim, dtype=query_dtype)
    query_scale = torch.ones(tokens * heads, dtype=scale_dtype)
    key = torch.zeros(4, 128, 1, dim, dtype=query_dtype)
    key_scale = torch.ones(4, 128, 1, 1, dtype=scale_dtype)
    weights = torch.ones(tokens, heads, dtype=weights_dtype)
    query_ends = torch.tensor(query_lengths, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    key_lengths = torch.full((len(query_lengths),), 256, dtype=torch.int32)
    block_table = torch.zeros(len(query_lengths), 2, dtype=torch.int32)
    cache = (key, key_scale) if cache_indices == (0, 1) else (key_scale, key)
    metadata = torch.empty(1024, dtype=torch.int32)
    indices = torch.full((tokens, 1, 2048), -1, dtype=torch.int32)
    metadata_op = mock.Mock(return_value=metadata)
    indexer_op = mock.Mock(return_value=(indices, torch.empty(0)))
    with (
        mock.patch.object(
            A5DeviceAdaptor, "_load_cann_quant_lightning_indexer_ops", return_value=(metadata_op, indexer_op)
        ),
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_quant_lightning_indexer", create=True) as legacy_op,
    ):
        result = A5DeviceAdaptor.indexer_select_post_process(
            query,
            query_scale,
            (tokens, heads, dim),
            weights,
            cache,
            *cache_indices,
            SimpleNamespace(block_table=block_table),
            query_ends,
            key_lengths,
            True,
            False,
            True,
        )
    assert result is indices
    legacy_op.assert_not_called()
    metadata_op.assert_called_once()
    indexer_op.assert_called_once()
    metadata_kwargs = metadata_op.call_args.kwargs
    kwargs = indexer_op.call_args.kwargs
    expected_cumulative = torch.tensor([0, *query_ends.tolist()], dtype=torch.int32)
    for call_kwargs in (metadata_kwargs, kwargs):
        torch.testing.assert_close(call_kwargs["cu_seqlens_q"], expected_cumulative)
        assert call_kwargs["seqused_k"] is key_lengths
        assert call_kwargs["quant_mode"] == quant_mode
        assert call_kwargs["topk"] == 2048
        assert call_kwargs["layout_q"] == "TND"
        assert call_kwargs["layout_k"] == "PA_BBND"
        assert call_kwargs["mask_mode"] == 3
        assert call_kwargs["cmp_ratio"] == 1
    assert metadata_kwargs["batch_size"] == len(query_lengths)
    assert metadata_kwargs["num_heads_q"] == heads
    assert metadata_kwargs["head_dim"] == dim
    assert kwargs["metadata"] is metadata
    assert kwargs["block_table"] is block_table
    assert kwargs["key"] is key
    assert kwargs["query"].shape == (tokens, heads, dim)
    assert kwargs["query"].data_ptr() == query.data_ptr()
    assert kwargs["query_dequant_scale"].shape == (tokens, heads)
    assert kwargs["key_dequant_scale"].shape == (4, 128, 1)
    assert kwargs["key_dequant_scale"].data_ptr() == key_scale.data_ptr()
    assert kwargs["weights"].dtype == scale_dtype
    torch.testing.assert_close(kwargs["weights"], weights.to(scale_dtype))
    assert kwargs["return_value"] == 0


def test_a5_c8_indexer_without_scale_preserves_unquantized_fallback():
    query = torch.zeros(1, 32, 128, dtype=torch.bfloat16)
    key = torch.zeros(1, 128, 1, 128, dtype=torch.bfloat16)
    weights = torch.ones(1, 32)
    query_ends = torch.tensor([1], dtype=torch.int32)
    key_lengths = torch.tensor([128], dtype=torch.int32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    indices = torch.full((1, 1, 2048), -1, dtype=torch.int32)
    with (
        mock.patch.object(A5DeviceAdaptor, "_load_cann_quant_lightning_indexer_ops") as loader,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_lightning_indexer",
            create=True,
            return_value=(indices, torch.empty(0)),
        ) as op,
    ):
        result = A5DeviceAdaptor.indexer_select_post_process(
            query,
            None,
            query.shape,
            weights,
            (key, None),
            0,
            1,
            SimpleNamespace(block_table=block_table),
            query_ends,
            key_lengths,
            True,
            False,
            True,
        )
    assert result is indices
    loader.assert_not_called()
    assert op.call_args.kwargs["actual_seq_lengths_query"] is query_ends
    assert op.call_args.kwargs["actual_seq_lengths_key"] is key_lengths


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("shape", [(2, 8), (2, 3, 8), (2, 1, 3, 8)])
@pytest.mark.parametrize("inverse", [False, True])
def test_partial_rotary_preserves_storage_and_adapts_native_signature(device, shape, inverse):
    x = torch.randn(shape)
    cos, sin = torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 4)
    with (
        mock.patch(
            "vllm_ascend.device.device_op.get_current_hardware_profile", return_value=get_hardware_profile(device)
        ),
        mock.patch.object(torch.ops._C_ascend, "inplace_partial_rotary_mul", create=True) as op,
    ):
        result = BaseDeviceAdaptor.apply_partial_rotary_inplace(x, cos, sin, start=4, end=8, inverse=inverse)
    assert result is x
    work, actual_cos, actual_sin = op.call_args.args
    assert work.ndim == 4
    assert work.data_ptr() == x.data_ptr()
    assert actual_cos is cos
    assert op.call_args.kwargs["partial_slice"] == [4, 8]
    if device == AscendDeviceType.A5:
        assert "negate_sin" not in op.call_args.kwargs
        torch.testing.assert_close(actual_sin, -sin if inverse else sin)
    else:
        assert actual_sin is sin
        assert op.call_args.kwargs["negate_sin"] is inverse


@pytest.mark.parametrize("device", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("custom_ops", [False, True])
def test_rms_norm_cast_only_uses_supported_fused_operator(device, custom_ops):
    x, weight = torch.randn(2, 8), torch.ones(8)
    outputs = (x, x.float())
    with (
        mock.patch(
            "vllm_ascend.device.device_op.get_current_hardware_profile", return_value=get_hardware_profile(device)
        ),
        mock.patch("vllm_ascend.utils.enable_custom_op", return_value=custom_ops),
        mock.patch.object(torch.ops._C_ascend, "npu_rms_norm_cast", create=True, return_value=outputs) as op,
    ):
        result = BaseDeviceAdaptor.rms_norm_cast(x, weight, 1e-6)
    if device == AscendDeviceType.A3 and custom_ops:
        assert result is outputs
        op.assert_called_once_with(x, weight, 1e-6)
    else:
        assert result is None
        op.assert_not_called()


def test_host_registration_flags_follow_runtime_abi():
    assert BaseDeviceAdaptor.host_register_flags() == 0x10000002
    assert A5DeviceAdaptor.host_register_flags() == 0x2


def test_deepseek_v41_backend_is_device_routed():
    assert BaseDeviceAdaptor.get_dsv41_packed_cache_ops() is None
    with mock.patch.dict("sys.modules", {"vllm_ascend.vllm_ascend_C": mock.Mock()}):
        assert A5DeviceAdaptor.get_dsv41_packed_cache_ops().__name__ == "MixedQuantPackedCacheOps"


@pytest.mark.parametrize("adaptor", [BaseDeviceAdaptor, A5DeviceAdaptor])
@pytest.mark.parametrize("block_table_mode", ["pa", "none", "omitted"])
@pytest.mark.parametrize("contiguous", [False, True])
def test_fia_only_contiguizes_non_pa_inputs(adaptor, block_table_mode, contiguous):
    use_pa = block_table_mode == "pa"
    omit_block_table = block_table_mode == "omitted"
    key = torch.randn(2, 3, 4).transpose(0, 1)
    value = torch.randn_like(key)
    if contiguous:
        key = key.contiguous()
        value = value.contiguous()
    block_table = torch.tensor([[0]], dtype=torch.int32) if use_pa else None
    kwargs = {} if omit_block_table else {"block_table": block_table}
    expected = (object(), object())

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fused_infer_attention_score", return_value=expected
    ) as mock_fia:
        result = adaptor.npu_fused_infer_attention_score(
            query=torch.randn(3, 2, 4),
            key=key,
            value=value,
            attn_metadata=None,
            key_cache=None,
            value_cache=None,
            current_key=key,
            current_value=value,
            num_heads=2,
            num_key_value_heads=2,
            head_size=4,
            scale=0.5,
            is_prefill_no_cache=not use_pa,
            **kwargs,
        )

    assert result is expected
    mock_fia.assert_called_once()
    call_kwargs = mock_fia.call_args.kwargs
    for name, original in (("key", key), ("value", value)):
        actual = call_kwargs[name]
        torch.testing.assert_close(actual, original)
        if use_pa or contiguous:
            assert actual is original
        else:
            assert actual.is_contiguous()
            assert actual.data_ptr() != original.data_ptr()
        if use_pa:
            assert actual.stride() == original.stride()
    assert call_kwargs.get("block_table") is block_table
    assert ("block_table" in call_kwargs) is (not omit_block_table)


def test_reshape_and_cache_makes_scatter_inputs_contiguous():
    key = torch.randn(2, 3, 4).transpose(0, 1)
    value = torch.randn(2, 3, 4).transpose(0, 1)
    slot_mapping = torch.arange(8, dtype=torch.int32)[::2]
    key_cache = object()
    value_cache = object()

    assert not key.is_contiguous()
    assert not value.is_contiguous()
    assert not slot_mapping.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_scatter_pa_kv_cache") as mock_scatter:
        BaseDeviceAdaptor.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    mock_scatter.assert_called_once()
    call_kwargs = mock_scatter.call_args.kwargs
    assert call_kwargs["key"] is not key
    assert call_kwargs["value"] is not value
    assert call_kwargs["slot_mapping"] is not slot_mapping
    assert call_kwargs["key"].is_contiguous()
    assert call_kwargs["value"].is_contiguous()
    assert call_kwargs["slot_mapping"].is_contiguous()
    torch.testing.assert_close(call_kwargs["key"], key)
    torch.testing.assert_close(call_kwargs["value"], value)
    torch.testing.assert_close(call_kwargs["slot_mapping"], slot_mapping)
    assert call_kwargs["key_cache"] is key_cache
    assert call_kwargs["value_cache"] is value_cache
    assert call_kwargs["cache_mode"] == "Norm"


def test_base_reshape_and_cache_uses_custom_scatter_for_bnsd():
    key = torch.randn(2, 8, 64)
    value = torch.randn_like(key)
    key_cache = torch.empty(4, 8, 128, 64)
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(2, dtype=torch.int32)

    with (
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_scatter_pa_kv_cache",
            create=True,
        ) as mock_custom_scatter,
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_scatter_pa_kv_cache") as mock_public_scatter,
    ):
        BaseDeviceAdaptor.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            use_bnsd=True,
        )

    mock_public_scatter.assert_not_called()
    mock_custom_scatter.assert_called_once()
    assert mock_custom_scatter.call_args.args[2] is key_cache
    assert mock_custom_scatter.call_args.args[3] is value_cache
    assert mock_custom_scatter.call_args.kwargs["cache_mode"] == "Norm"
    assert mock_custom_scatter.call_args.kwargs["scatter_mode"] == "NHSD"


def test_a5_reshape_and_cache_uses_bsnd_view_for_bnsd():
    key = torch.randn(2, 8, 64)
    value = torch.randn_like(key)
    key_cache = torch.empty(4, 8, 128, 64)
    value_cache = torch.empty_like(key_cache)
    slot_mapping = torch.arange(2, dtype=torch.int32)

    with (
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_scatter_pa_kv_cache",
            create=True,
        ) as mock_custom_scatter,
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_scatter_pa_kv_cache") as mock_public_scatter,
    ):
        A5DeviceAdaptor.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            use_bnsd=True,
        )

    mock_custom_scatter.assert_not_called()
    mock_public_scatter.assert_called_once()
    call_kwargs = mock_public_scatter.call_args.kwargs
    assert call_kwargs["key_cache"].shape == (4, 128, 8, 64)
    assert call_kwargs["value_cache"].shape == (4, 128, 8, 64)
    assert not call_kwargs["key_cache"].is_contiguous()
    assert not call_kwargs["value_cache"].is_contiguous()
    assert call_kwargs["key_cache"].data_ptr() == key_cache.data_ptr()
    assert call_kwargs["value_cache"].data_ptr() == value_cache.data_ptr()


@pytest.mark.parametrize("adaptor", [BaseDeviceAdaptor, A5DeviceAdaptor])
def test_kv_cache_load_makes_seq_lens_contiguous(adaptor):
    cache_kv_c = object()
    cache_k_pe = object()
    block_table = object()
    context_seq_len_npu = torch.arange(8, dtype=torch.int32)[::2]
    seq_starts = object()
    key = object()
    value = object()

    assert not context_seq_len_npu.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache") as mock_gather:
        adaptor.kv_cache_load(
            cache_kv_c,
            cache_k_pe,
            block_table,
            context_seq_len_npu,
            seq_starts,
            key,
            value,
        )

    mock_gather.assert_called_once()
    call_args = mock_gather.call_args.args
    assert call_args[0] is cache_kv_c
    assert call_args[1] is cache_k_pe
    assert call_args[2] is block_table
    assert call_args[3] is not context_seq_len_npu
    assert call_args[3].is_contiguous()
    torch.testing.assert_close(call_args[3], context_seq_len_npu)
    assert mock_gather.call_args.kwargs["seq_offset"] is seq_starts
    assert mock_gather.call_args.kwargs["key"] is key
    assert mock_gather.call_args.kwargs["value"] is value
    assert mock_gather.call_args.kwargs["is_seq_lens_cumsum"] is False


@pytest.mark.parametrize(
    "adaptor,token_fused",
    [(BaseDeviceAdaptor, False), (A5DeviceAdaptor, False), (A5DeviceAdaptor, True)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_cache_load_passes_strided_views_directly_to_native(adaptor, token_fused, dtype):
    blocks, block_size, key_dim, rope_dim = 3, 128, 512, 64
    prefix = 256
    page_elements = block_size * (key_dim + rope_dim) + 128
    raw = torch.empty(prefix + blocks * page_elements, dtype=dtype)
    key_stride = key_dim + rope_dim if token_fused else key_dim
    rope_stride = key_dim + rope_dim if token_fused else rope_dim
    rope_offset = prefix + (key_dim if token_fused else block_size * key_dim)
    cache_k = raw.as_strided((blocks, block_size, 1, key_dim), (page_elements, key_stride, key_stride, 1), prefix)
    cache_v = raw.as_strided(
        (blocks, block_size, 1, rope_dim), (page_elements, rope_stride, rope_stride, 1), rope_offset
    )
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    lengths = torch.tensor([127], dtype=torch.int32)
    offsets = torch.tensor([128], dtype=torch.int32)
    key = torch.empty(127, 1, key_dim, dtype=dtype)
    value = torch.empty(127, 1, rope_dim, dtype=dtype)
    assert not cache_k.is_contiguous() and not cache_v.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache") as gather:
        adaptor.kv_cache_load(cache_k, cache_v, table, lengths, offsets, key, value)

    gather.assert_called_once()
    # Identity preserves both the physical page stride and component offset;
    # making a dense copy of either cache would violate this contract.
    for actual, expected in zip(gather.call_args.args, (cache_k, cache_v, table, lengths)):
        assert actual is expected
    assert gather.call_args.kwargs["seq_offset"] is offsets
    assert gather.call_args.kwargs["key"] is key
    assert gather.call_args.kwargs["value"] is value
    assert gather.call_args.kwargs["is_seq_lens_cumsum"] is False


@pytest.mark.parametrize(
    "shape,dim,indices,value,dtype",
    [
        ((8,), 0, [1, 4, 6], True, torch.bool),
        ((4, 3), 0, [1, 3], -1, torch.int32),
        ((4, 3), 1, [0, 2], 7, torch.int64),
        ((2, 3, 4), -2, [-1, 0], -3.5, torch.float32),
        ((2, 3), 1, [], 9, torch.int64),
        ((5,), 0, [1, 1, 3], 6, torch.int64),
    ],
)
def test_a5_index_fill_matches_torch_index_fill(shape, dim, indices, value, dtype):
    source = torch.arange(math.prod(shape)).reshape(shape)
    source = (source % 2).bool() if dtype is torch.bool else source.to(dtype)
    index = torch.tensor(indices, dtype=torch.int64)
    expected = source.clone().index_fill_(dim, index, value)
    actual_input = source.clone()

    actual = A5DeviceAdaptor.index_fill(actual_input, dim, index, value)

    assert actual is actual_input
    torch.testing.assert_close(actual, expected)


def test_a5_index_fill_uses_scatter():
    tensor = mock.Mock()
    tensor.dim.return_value = 1
    tensor.size.return_value = 8
    tensor.shape = (8,)

    result = A5DeviceAdaptor.index_fill(tensor, 0, torch.tensor([1, 3]), 5)

    assert result is tensor
    tensor.scatter_.assert_called_once()
    tensor.index_fill_.assert_not_called()


@pytest.mark.parametrize("packed_cache", [False, True])
@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("architecture", ["DeepseekV41ForCausalLM", "DeepseekV41DSparkModel", "DeepseekV3ForCausalLM"])
def test_v41_runner_support_is_hardware_scoped(packed_cache, use_v2, architecture):
    from types import SimpleNamespace

    from vllm_ascend.platform import _validate_model_runner_config

    config = SimpleNamespace(model_config=SimpleNamespace(architecture=architecture), use_v2_model_runner=use_v2)
    with mock.patch("vllm_ascend.platform.get_current_hardware_profile") as profile:
        profile.return_value.supports.return_value = packed_cache
        if packed_cache and not use_v2 and architecture != "DeepseekV3ForCausalLM":
            with pytest.raises(ValueError, match="requires Model Runner V2"):
                _validate_model_runner_config(config)
        else:
            _validate_model_runner_config(config)
