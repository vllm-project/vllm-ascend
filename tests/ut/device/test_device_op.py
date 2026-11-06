import math
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import (
    A5DeviceAdaptor,
    Ascend310PDeviceAdaptor,
    BaseDeviceAdaptor,
    get_device_adaptor,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_hardware_profile


@pytest.mark.parametrize(
    "device,adaptor",
    [
        (AscendDeviceType.A2, BaseDeviceAdaptor),
        (AscendDeviceType.A3, BaseDeviceAdaptor),
        (AscendDeviceType.A5, A5DeviceAdaptor),
        (AscendDeviceType._310P, Ascend310PDeviceAdaptor),
    ],
)
def test_sparse_adaptor_selection_is_device_scoped(device, adaptor):
    profile = get_hardware_profile(device)
    if device == AscendDeviceType.A3:
        # A3 retains both the release profile and standard Base adaptor.
        profile = replace(profile, device_adaptor_family=DeviceAdaptorFamily.STANDARD)
    with mock.patch("vllm_ascend.device.device_op.get_current_hardware_profile", return_value=profile):
        assert get_device_adaptor() is adaptor


@pytest.mark.parametrize("group_size", [12, 64, 65, 96, 128])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("use_rope", [False, True])
def test_sparse_attention_query_group_padding_preserves_outputs(group_size, num_kv_heads, return_lse, use_rope):
    tokens, kv_tokens, head_dim = 2, 3, 4
    query = torch.randn(tokens, num_kv_heads * group_size, head_dim)
    query_rope = torch.randn(tokens, num_kv_heads * group_size, 2) if use_rope else None
    key = torch.randn(1, kv_tokens, num_kv_heads, head_dim)
    key_rope = torch.randn(1, kv_tokens, num_kv_heads, 2) if use_rope else None
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    topk = torch.arange(kv_tokens, dtype=torch.int32).view(1, 1, -1).expand(tokens, 1, -1)
    lengths = torch.tensor([tokens], dtype=torch.int32)
    scale = 0.5

    def reference(q, q_rope):
        grouped = q.unflatten(1, (num_kv_heads, -1))
        scores = torch.einsum("tkhd,skd->tkhs", grouped, key[0])
        if q_rope is not None:
            scores += torch.einsum("tkhd,skd->tkhs", q_rope.unflatten(1, (num_kv_heads, -1)), key_rope[0])
        scores *= scale
        maximum = scores.amax(dim=-1)
        total = torch.exp(scores - maximum.unsqueeze(-1)).sum(dim=-1)
        output = torch.einsum("tkhs,skd->tkhd", scores.softmax(dim=-1), key[0]).flatten(1, 2)
        return output, maximum.permute(1, 0, 2).contiguous(), total.permute(1, 0, 2).contiguous()

    def native_op(**kwargs):
        groups = kwargs["query"].shape[1] // num_kv_heads
        assert groups <= 64 or groups == 128
        assert kwargs["key"] is key and kwargs["value"] is key
        assert kwargs["key_rope"] is key_rope and kwargs["sparse_indices"] is topk
        output, maximum, total = reference(kwargs["query"], kwargs["query_rope"])
        return (output, maximum, total) if kwargs["return_softmax_lse"] else (output, torch.empty(0), torch.empty(0))

    expected = reference(query, query_rope)
    with mock.patch.object(torch.ops._C_ascend, "npu_sparse_flash_attention", create=True, side_effect=native_op) as op:
        actual = BaseDeviceAdaptor.execute_sparse_flash_attention_process(
            SimpleNamespace(scale=scale, qk_rope_head_dim=2 if use_rope else 0),
            query,
            query_rope,
            (key, key_rope),
            topk,
            SimpleNamespace(block_table=block_table),
            lengths,
            torch.tensor([kv_tokens], dtype=torch.int32),
            return_lse=return_lse,
        )
    op.assert_called_once()
    if return_lse:
        for output, reference_output in zip(actual, expected):
            torch.testing.assert_close(output, reference_output)
            assert output.is_contiguous()
    else:
        torch.testing.assert_close(actual, expected[0])
        assert actual.is_contiguous()


@pytest.mark.parametrize("group_size", [8, 12, 48, 64, 96, 128])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("return_lse,native_tensor", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("rope_dim", [0, 2])
def test_fp8_sparse_attention_native_heads_preserve_attention_and_lse(
    group_size, num_kv_heads, return_lse, native_tensor, rope_dim
):
    generator = torch.Generator().manual_seed(42)
    tokens, kv_tokens, dim = 2, 3, 4
    query = torch.randn(tokens, num_kv_heads * group_size, dim, generator=generator)
    query_rope = torch.randn(tokens, num_kv_heads * group_size, rope_dim, generator=generator)
    key = torch.randn(1, kv_tokens, num_kv_heads, dim, generator=generator).to(torch.float8_e4m3fn)
    key_rope = torch.randn(kv_tokens, num_kv_heads, rope_dim, generator=generator)
    topk = torch.arange(kv_tokens, dtype=torch.int32).view(1, 1, -1).expand(tokens, 1, -1)
    table = torch.zeros(1, 1, dtype=torch.int32)

    def reference(q, q_rope):
        grouped = q.unflatten(1, (num_kv_heads, -1))
        scores = torch.einsum("tkhd,skd->tkhs", grouped, key[0].float())
        scores += torch.einsum("tkhd,skd->tkhs", q_rope.unflatten(1, (num_kv_heads, -1)), key_rope)
        scores *= 0.5
        maximum = scores.amax(-1)
        total = torch.exp(scores - maximum.unsqueeze(-1)).sum(-1)
        output = torch.einsum("tkhs,skd->tkhd", scores.softmax(-1), key[0].float()).flatten(1, 2)
        return output, maximum.permute(1, 0, 2).contiguous(), total.permute(1, 0, 2).contiguous()

    def native_op(**kwargs):
        q = kwargs["query"]
        assert q.is_contiguous()
        assert q.shape[1] // num_kv_heads == group_size
        assert kwargs["return_softmax_lse"]
        assert kwargs["key"] is key and kwargs["block_table"] is table and kwargs["sparse_indices"] is topk
        output, maximum, total = reference(q[..., :dim], q[..., dim:])
        if native_tensor:
            return output
        return (output, maximum, total) if return_lse else (output, torch.empty(0), torch.empty(0))

    expected = reference(query, query_rope)
    with (
        mock.patch.object(
            torch.ops._C_ascend, "npu_kv_quant_sparse_flash_attention_vllm", create=True, side_effect=native_op
        ),
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_kv_quant_sparse_flash_attention",
            create=True,
            side_effect=AssertionError("QSFA must use the custom operator"),
        ),
    ):
        actual = BaseDeviceAdaptor.execute_sparse_flash_attention_process(
            SimpleNamespace(scale=0.5, is_kimi_k31=True),
            query,
            query_rope,
            (key,),
            topk,
            SimpleNamespace(block_table=table),
            torch.tensor([tokens]),
            torch.tensor([kv_tokens]),
            return_lse=return_lse,
        )
    for output, reference_output in zip(actual if return_lse else (actual,), expected):
        torch.testing.assert_close(output, reference_output)
        assert output.is_contiguous()


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
    if device in (AscendDeviceType.A2, AscendDeviceType.A3) and custom_ops:
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


def test_kv_cache_load_makes_seq_lens_contiguous():
    cache_kv_c = object()
    cache_k_pe = object()
    block_table = object()
    context_seq_len_npu = torch.arange(8, dtype=torch.int32)[::2]
    seq_starts = object()
    key = object()
    value = object()

    assert not context_seq_len_npu.is_contiguous()

    with (
        mock.patch("vllm_ascend.device.device_op.try_load_strided_mla_cache", return_value=False),
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache") as mock_gather,
    ):
        BaseDeviceAdaptor.kv_cache_load(
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


@pytest.mark.parametrize(
    ("adaptor", "uses_fused_kernel"),
    [
        (BaseDeviceAdaptor, True),
        (A5DeviceAdaptor, False),
        (Ascend310PDeviceAdaptor, False),
    ],
)
def test_clipped_swiglu_kernel_selection_per_adaptor(adaptor, uses_fused_kernel):
    limit, alpha, beta = 7.0, 1.702, 1.0
    x = torch.randn(2, 8)

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_clipped_swiglu") as mock_kernel:
        out = adaptor.clipped_swiglu(x, swiglu_limit=limit, swiglu_alpha=alpha, swiglu_beta=beta)

    if uses_fused_kernel:
        mock_kernel.assert_called_once_with(x, interleaved=False, alpha=alpha, limit=limit, bias=beta)
        return

    mock_kernel.assert_not_called()
    d = x.shape[-1] // 2
    gate = torch.clamp(x[..., :d], max=limit)
    up = torch.clamp(x[..., d:], min=-limit, max=limit)
    torch.testing.assert_close(out, gate * torch.sigmoid(alpha * gate) * (up + beta))
