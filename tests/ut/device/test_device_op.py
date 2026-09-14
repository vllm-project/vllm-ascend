from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.device.mxfp_compat import (
    FLOAT4_E2M1FN_X2_DTYPE,
    FLOAT8_E8M0FNU_DTYPE,
    QUANT_DTYPES,
)
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("use_mla_rope", [True, False])
def test_a5_mla_preprocess_only_decode_passes_optional_rope(use_mla_rope):
    num_tokens = 2
    num_heads = 4
    kv_lora_rank = 8
    rope_dim = 6
    hidden_size = 16
    hidden_states = torch.randn(num_tokens, hidden_size)
    cos = torch.randn(num_tokens, 1, 1, rope_dim)
    sin = torch.randn_like(cos)
    kv_cache = (
        torch.randn(4, 1, 1, kv_lora_rank),
        torch.randn(4, 1, 1, rope_dim),
    )
    attn_metadata = SimpleNamespace(
        num_decode_tokens=num_tokens,
        decode=SimpleNamespace(cos=cos, sin=sin),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
    )
    atten_obj = SimpleNamespace(
        weight_dq=mock.MagicMock(),
        weight_uq_qr=mock.MagicMock(),
        W_UK_T=mock.MagicMock(),
        weight_dkv_kr=mock.MagicMock(),
        q_a_layernorm=SimpleNamespace(weight=SimpleNamespace(data=mock.MagicMock())),
        kv_a_layernorm=SimpleNamespace(weight=SimpleNamespace(data=mock.MagicMock())),
        weight_dq_scale=mock.MagicMock(),
        weight_uq_qr_scale=mock.MagicMock(),
        weight_dkv_kr_scale=mock.MagicMock(),
        fa_quant_layer=False,
        fak_descale_reciprocal=None,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim,
        reorg_decode_q=mock.MagicMock(side_effect=lambda q_nope, q_pe: (q_nope, q_pe)),
    )
    query_nope = torch.randn(num_tokens, num_heads, kv_lora_rank)
    query_rope = torch.randn(num_tokens, num_heads, rope_dim)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_dynamic_mx_quant",
            return_value=(hidden_states.unsqueeze(1), mock.MagicMock()),
            create=True,
        ),
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_mla_prolog_v3",
            return_value=(query_nope, query_rope, None, None, None),
            create=True,
        ) as mock_rope_prolog,
        mock.patch(
            "vllm_ascend.device.device_op._npu_mla_prolog_v3_no_rope",
            return_value=(query_nope, query_rope, None, None, None),
        ) as mock_no_rope_prolog,
    ):
        A5DeviceAdaptor.mla_preprocess_only_decode(
            atten_obj,
            hidden_states,
            kv_cache,
            attn_metadata,
            use_mla_rope=use_mla_rope,
        )

    mock_prolog = mock_rope_prolog if use_mla_rope else mock_no_rope_prolog
    call_kwargs = mock_prolog.call_args.kwargs
    if use_mla_rope:
        torch.testing.assert_close(call_kwargs["rope_cos"], cos.view(num_tokens, 1, rope_dim))
        torch.testing.assert_close(call_kwargs["rope_sin"], sin.view(num_tokens, 1, rope_dim))
    else:
        assert call_kwargs["rope_cos"] is None
        assert call_kwargs["rope_sin"] is None


@pytest.mark.parametrize("use_mla_rope", [True, False])
def test_a5_mla_preprocess_only_decode_supports_native_bf16_weights(use_mla_rope):
    num_tokens = 2
    num_heads = 4
    kv_lora_rank = 8
    rope_dim = 6
    hidden_states = torch.randn(num_tokens, 16, dtype=torch.bfloat16)
    kv_cache = (
        torch.randn(4, 1, 1, kv_lora_rank, dtype=torch.bfloat16),
        torch.randn(4, 1, 1, rope_dim, dtype=torch.bfloat16),
    )
    cos = torch.randn(num_tokens, 1, 1, rope_dim, dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    attn_metadata = SimpleNamespace(
        num_decode_tokens=num_tokens,
        decode=SimpleNamespace(cos=cos, sin=sin),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
    )
    atten_obj = SimpleNamespace(
        weight_dq=mock.MagicMock(),
        weight_uq_qr=mock.MagicMock(),
        W_UK_T=mock.MagicMock(),
        weight_dkv_kr=mock.MagicMock(),
        q_a_layernorm=SimpleNamespace(weight=SimpleNamespace(data=mock.MagicMock())),
        kv_a_layernorm=SimpleNamespace(weight=SimpleNamespace(data=mock.MagicMock())),
        mlapo_weight_quant_mode=0,
        fa_quant_layer=False,
        fak_descale_reciprocal=None,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim,
        reorg_decode_q=mock.MagicMock(side_effect=lambda q_nope, q_pe: (q_nope, q_pe)),
    )
    query_nope = torch.randn(num_tokens, num_heads, kv_lora_rank, dtype=torch.bfloat16)
    query_rope = torch.randn(num_tokens, num_heads, rope_dim, dtype=torch.bfloat16)

    with (
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_dynamic_mx_quant", create=True) as mock_quant,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_mla_prolog_v3",
            return_value=(query_nope, query_rope, None, None, None),
            create=True,
        ) as mock_rope_prolog,
        mock.patch(
            "vllm_ascend.device.device_op._npu_mla_prolog_v3_no_rope",
            return_value=(query_nope, query_rope, None, None, None),
        ) as mock_no_rope_prolog,
    ):
        A5DeviceAdaptor.mla_preprocess_only_decode(
            atten_obj,
            hidden_states,
            kv_cache,
            attn_metadata,
            use_mla_rope=use_mla_rope,
        )

    mock_quant.assert_not_called()
    mock_prolog = mock_rope_prolog if use_mla_rope else mock_no_rope_prolog
    call_kwargs = mock_prolog.call_args.kwargs
    torch.testing.assert_close(call_kwargs["token_x"], hidden_states)
    assert call_kwargs["token_x"].shape == hidden_states.shape
    assert call_kwargs["weight_quant_mode"] == 0
    assert call_kwargs["dequant_scale_x"] is None
    assert call_kwargs["dequant_scale_w_dq"] is None
    assert call_kwargs["dequant_scale_w_uq_qr"] is None
    assert call_kwargs["dequant_scale_w_dkv_kr"] is None
    if use_mla_rope:
        torch.testing.assert_close(call_kwargs["rope_cos"], cos.view(num_tokens, rope_dim))
        torch.testing.assert_close(call_kwargs["rope_sin"], sin.view(num_tokens, rope_dim))
    else:
        assert call_kwargs["rope_cos"] is None
        assert call_kwargs["rope_sin"] is None
    assert call_kwargs["cache_index"].shape == (num_tokens,)


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


def test_kv_cache_load_makes_seq_lens_contiguous():
    cache_kv_c = object()
    cache_k_pe = object()
    block_table = object()
    context_seq_len_npu = torch.arange(8, dtype=torch.int32)[::2]
    seq_starts = object()
    key = object()
    value = object()

    assert not context_seq_len_npu.is_contiguous()

    with mock.patch("vllm_ascend.device.device_op.torch_npu.npu_gather_pa_kv_cache") as mock_gather:
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


def test_a5_w4a8_mxfp_uses_fused_grouped_matmul_swiglu_quant_v2():
    """W4A8MXFP must dispatch to the fused npu_grouped_matmul_swiglu_quant_v2
    (MXFP4 weight) instead of the two-stage GMM + swiglu_group_quant combo."""
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    weight = torch.randn(4, 8, dtype=torch.uint8)
    weight_scale = torch.randn(4, 8, dtype=torch.uint8)
    group_list = torch.tensor([2])
    x_scale = torch.randn(2, 1, dtype=torch.bfloat16)
    out = torch.randn(2, 4, dtype=torch.bfloat16)
    # ndim != 2 so maybe_normalize_mxfp_scale_layout keeps it as-is.
    out_scale = torch.randn(2, 1, 1, dtype=torch.bfloat16)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_grouped_matmul_swiglu_quant_v2",
            return_value=(out, out_scale),
        ) as mock_v2,
        mock.patch("vllm_ascend.device.device_op.torch_npu.npu_grouped_matmul") as mock_gmm,
        mock.patch(
            "vllm_ascend.device.device_op.torch.ops._C_ascend.npu_swiglu_group_quant",
            create=True,
        ) as mock_swiglu_group_quant,
    ):
        result = A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant(
            x=x,
            weight=weight,
            group_list=group_list,
            weight_scale=weight_scale,
            x_scale=x_scale,
            use_mxfp_quant=True,
            mxfp_quant_dtype=QuantType.W4A8MXFP,
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=FLOAT4_E2M1FN_X2_DTYPE,
            swiglu_limit=0.0,
        )

    assert result[0] is out
    assert result[1] is out_scale
    assert result[2] is None
    mock_gmm.assert_not_called()
    mock_swiglu_group_quant.assert_not_called()
    mock_v2.assert_called_once()
    call_kwargs = mock_v2.call_args.kwargs
    assert len(call_kwargs["weight"]) == 1 and call_kwargs["weight"][0] is weight
    assert len(call_kwargs["weight_scale"]) == 1 and call_kwargs["weight_scale"][0] is weight_scale
    assert call_kwargs["x_scale"] is x_scale
    assert call_kwargs["group_list"] is group_list
    assert call_kwargs["dequant_mode"] == 2
    assert call_kwargs["quant_mode"] == 2
    assert call_kwargs["quant_dtype"] == torch.float8_e4m3fn
    expected_weight_dtype = FLOAT4_E2M1FN_X2_DTYPE if FLOAT4_E2M1FN_X2_DTYPE in QUANT_DTYPES else None
    assert call_kwargs["weight_dtype"] == expected_weight_dtype
    assert "swiglu_limit" not in call_kwargs


def test_a5_w4a8_mxfp_falls_back_to_two_stage_for_swiglu_limit():
    """v2 has no swiglu_limit input, so a non-zero limit falls back to the
    two-stage GMM + swiglu_group_quant path."""
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    weight = torch.randn(4, 8, dtype=torch.uint8)
    weight_scale = torch.randn(4, 8, dtype=torch.uint8)
    group_list = torch.tensor([2])
    x_scale = torch.randn(2, 1, dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
    out = torch.randn(2, 4, dtype=torch.bfloat16)
    out_scale = torch.randn(2, 1, 1, dtype=torch.bfloat16)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_grouped_matmul",
            return_value=[hidden_states],
        ) as mock_gmm,
        mock.patch(
            "vllm_ascend.device.device_op.torch.ops._C_ascend.npu_swiglu_group_quant",
            create=True,
            return_value=(out, out_scale, None),
        ) as mock_swiglu_group_quant,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_grouped_matmul_swiglu_quant_v2",
            return_value=(out, out_scale),
        ) as mock_v2,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.float4_e2m1fn_x2",
            FLOAT4_E2M1FN_X2_DTYPE,
            create=True,
        ),
        mock.patch(
            "vllm_ascend.device.device_op.torch.float8_e8m0fnu",
            FLOAT8_E8M0FNU_DTYPE,
            create=True,
        ),
    ):
        result = A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant(
            x=x,
            weight=weight,
            group_list=group_list,
            weight_scale=weight_scale,
            x_scale=x_scale,
            use_mxfp_quant=True,
            mxfp_quant_dtype=QuantType.W4A8MXFP,
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=FLOAT4_E2M1FN_X2_DTYPE,
            swiglu_limit=7.0,
        )

    assert result[0] is out
    assert result[1] is out_scale
    assert result[2] is None
    mock_v2.assert_not_called()
    gmm_kwargs = mock_gmm.call_args.kwargs
    assert gmm_kwargs["x"] == [x]
    assert gmm_kwargs["weight"] == [weight]
    assert gmm_kwargs["antiquant_scale"] == [weight_scale]
    assert gmm_kwargs["per_token_scale"] == [x_scale]
    assert gmm_kwargs["split_item"] == 2
    assert gmm_kwargs["group_list"] is group_list
    assert gmm_kwargs["output_dtype"] == torch.bfloat16
    swiglu_kwargs = mock_swiglu_group_quant.call_args.kwargs
    assert swiglu_kwargs["dst_type"] == torch.float8_e4m3fn
    assert swiglu_kwargs["quant_mode"] == 2
    assert swiglu_kwargs["clamp_value"] == 7.0


def test_npu_flash_attention_uses_fusion_attention_for_fp32():
    query = torch.randn(5, 4, 64, dtype=torch.float32)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
            return_value=(expected,),
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    mock_flash_attention.assert_not_called()
    mock_fusion_attention.assert_called_once()
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
    assert call_kwargs["head_num"] == 4
    assert call_kwargs["scale"] == 0.125
    assert call_kwargs["input_layout"] == "TND"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_flash_attention_uses_unpad_attention_for_low_precision(dtype):
    query = torch.randn(5, 4, 64, dtype=dtype)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)

    def fake_flash_attention(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        out.copy_(query + 1)

    with (
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        ) as mock_fusion_attention,
        mock.patch(
            "vllm_ascend.device.device_op.torch_npu._npu_flash_attention_unpad",
            side_effect=fake_flash_attention,
            create=True,
        ) as mock_flash_attention,
    ):
        output = BaseDeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    mock_fusion_attention.assert_not_called()
    mock_flash_attention.assert_called_once()
    call_kwargs = mock_flash_attention.call_args.kwargs
    assert call_kwargs["query"] is query
    assert call_kwargs["key"] is key
    assert call_kwargs["value"] is value
    assert call_kwargs["seq_len"] is seq_lens_cpu
    assert call_kwargs["num_heads"] == 4
    assert call_kwargs["num_kv_heads"] == 4
    assert call_kwargs["scale_value"] == 0.125
    torch.testing.assert_close(output, query + 1)


def test_a5_npu_flash_attention_uses_python_sequence_lengths():
    query = torch.randn(5, 4, 64, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    seq_lens_cpu = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.randn_like(query)

    with mock.patch(
        "vllm_ascend.device.device_op.torch_npu.npu_fusion_attention",
        return_value=(expected,),
    ) as mock_fusion_attention:
        output = A5DeviceAdaptor.npu_flash_attention(
            query=query,
            key=key,
            value=value,
            seq_lens_cpu=seq_lens_cpu,
            head_num=4,
            scale_value=0.125,
            num_kv_heads=4,
        )

    assert output is expected
    call_kwargs = mock_fusion_attention.call_args.kwargs
    assert call_kwargs["actual_seq_qlen"] == [2, 5]
    assert all(isinstance(seq_len, int) for seq_len in call_kwargs["actual_seq_qlen"])
    assert call_kwargs["actual_seq_kvlen"] is call_kwargs["actual_seq_qlen"]
