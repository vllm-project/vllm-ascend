from unittest import mock

import pytest
import torch

from vllm_ascend.attention.dsa_attn_kv_plan import (
    DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
    DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
)
from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.ops.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata


def test_a5_dsa_attn_kv_plan_centralizes_sparse_flash_mla_choices():
    bf16_plan = A5DeviceAdaptor.build_dsa_attn_kv_plan(torch.bfloat16)
    assert bf16_plan.uses_sparse_flash_mla is True
    assert bf16_plan.layout_kv == "PA_BBND"
    assert bf16_plan.requires_block_offset_slots is True
    assert bf16_plan.pack_kv_head_dim_extra is False
    assert bf16_plan.compressor_slot_mapping_format == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
    assert bf16_plan.sparse_attn_op is sparse_flash_mla
    assert bf16_plan.sparse_attn_metadata_op is sparse_flash_mla_metadata
    assert bf16_plan.sparse_attn_base_kwargs == {}
    assert bf16_plan.metadata_kwargs("npu:0") == {"device": "npu:0"}

    # Plan fields are the source of truth (no DeviceOperator thin wrappers).
    assert bf16_plan.sparse_attn_kwargs() == {}
    assert bf16_plan.applies_sparse_attn_runtime_kwargs is True
    assert bf16_plan.kv_scatter_kind == "nd_update_"

    with (
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_kv_quant_sparse_attn_sharedkv",
            create=True,
        ) as fp8_attn_op,
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_kv_quant_sparse_attn_sharedkv_metadata",
            create=True,
        ),
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_sparse_attn_sharedkv",
            create=True,
        ),
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_sparse_attn_sharedkv_metadata",
            create=True,
        ),
    ):
        fp8_plan = A5DeviceAdaptor.build_dsa_attn_kv_plan(torch.float8_e4m3fn)
        assert fp8_plan.uses_sparse_flash_mla is False
        assert fp8_plan.layout_kv == "PA_ND"
        assert fp8_plan.requires_block_offset_slots is False
        assert fp8_plan.pack_kv_head_dim_extra is True
        assert fp8_plan.compressor_slot_mapping_format == DSA_COMPRESSOR_SLOT_MAPPING_FLAT
        assert fp8_plan.sparse_attn_base_kwargs["kv_quant_mode"] == 1
        # A5 FP8 quant metadata matches main: kv_quant_mode only, no device=.
        assert fp8_plan.metadata_kwargs("npu:0") == {"kv_quant_mode": 1}
        assert fp8_plan.applies_sparse_attn_runtime_kwargs is False
        assert fp8_plan.sparse_attn_op is fp8_attn_op
        assert fp8_plan.kv_scatter_kind == "kv_compress_epilog"

        base_plan = BaseDeviceAdaptor.build_dsa_attn_kv_plan(torch.bfloat16)
        assert base_plan.uses_sparse_flash_mla is False
        assert base_plan.layout_kv == "PA_ND"
        assert base_plan.requires_block_offset_slots is True
        assert base_plan.pack_kv_head_dim_extra is False
        assert base_plan.compressor_slot_mapping_format == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
        assert base_plan.applies_sparse_attn_runtime_kwargs is True
        assert base_plan.kv_scatter_kind == "nd_update_v2"


def test_add_sparse_attn_extra_kwargs_respects_plan_flag():
    from vllm_ascend.attention.dsa_attn_kv_plan import add_sparse_attn_extra_kwargs

    bf16_plan = A5DeviceAdaptor.build_dsa_attn_kv_plan(torch.bfloat16)
    bf16_kwargs: dict = {}
    add_sparse_attn_extra_kwargs(bf16_plan, bf16_kwargs, cu_seqlens_ori_kv=1)
    assert bf16_kwargs == {"cu_seqlens_ori_kv": 1}

    with (
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_kv_quant_sparse_attn_sharedkv",
            create=True,
        ),
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_kv_quant_sparse_attn_sharedkv_metadata",
            create=True,
        ),
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_sparse_attn_sharedkv",
            create=True,
        ),
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_sparse_attn_sharedkv_metadata",
            create=True,
        ),
    ):
        fp8_plan = A5DeviceAdaptor.build_dsa_attn_kv_plan(torch.float8_e4m3fn)
        fp8_kwargs: dict = {}
        add_sparse_attn_extra_kwargs(fp8_plan, fp8_kwargs, cu_seqlens_ori_kv=1)
        assert fp8_kwargs == {}

        base_plan = BaseDeviceAdaptor.build_dsa_attn_kv_plan(None)
        base_kwargs: dict = {}
        add_sparse_attn_extra_kwargs(base_plan, base_kwargs, cu_seqlens_cmp_kv=2)
        assert base_kwargs == {"cu_seqlens_cmp_kv": 2}


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


def test_a5_format_dsa_slot_mapping_depends_on_kv_dtype():
    flat = torch.tensor([5, 130, 259], dtype=torch.int32)
    block_size = 128

    # BF16 SparseFlashMla needs [block_idx, offset].
    bf16_slots = A5DeviceAdaptor.format_dsa_slot_mapping(flat, block_size, torch.bfloat16)
    assert bf16_slots.shape == (3, 2)
    torch.testing.assert_close(
        bf16_slots,
        torch.tensor([[0, 5], [1, 2], [2, 3]], dtype=torch.int32),
    )

    # Quant path keeps flat ids. Omitting dtype must not accidentally BF16-format.
    with (
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_kv_quant_sparse_attn_sharedkv",
            create=True,
        ),
        mock.patch.object(
            torch.ops._C_ascend,
            "npu_kv_quant_sparse_attn_sharedkv_metadata",
            create=True,
        ),
    ):
        assert A5DeviceAdaptor.format_dsa_slot_mapping(flat, block_size, torch.float8_e4m3fn) is flat
        assert A5DeviceAdaptor.format_dsa_slot_mapping(flat, block_size) is flat


def test_a5_bf16_format_and_scatter_pass_minus1_directly():
    """Experimental path: invalid [-1, -1] rows are passed to the scatter op as-is."""
    flat = torch.tensor([5, -1, -1], dtype=torch.int32)
    formatted = A5DeviceAdaptor.format_dsa_slot_mapping(flat, 128, torch.bfloat16)
    torch.testing.assert_close(
        formatted,
        torch.tensor([[0, 5], [-1, -1], [-1, -1]], dtype=torch.int32),
    )

    cache = torch.zeros((2, 128, 1, 2), dtype=torch.bfloat16)
    x = torch.tensor([[1.0, 2.0], [9.0, 9.0], [8.0, 8.0]], dtype=torch.bfloat16).view(3, 1, 2)

    captured = {}

    def _fake_scatter_nd_update(var, indices, updates_tensor):
        captured["indices"] = indices.detach().clone()
        captured["updates"] = updates_tensor.detach().clone()
        # Apply only non-negative indices (what a -1-tolerant NPU op should do).
        valid = (indices[:, 0] >= 0) & (indices[:, 1] >= 0)
        if valid.any():
            var[indices[valid, 0], indices[valid, 1]] = updates_tensor[valid]

    # Lazy import inside plan.kv_compress_scatter uses torch_npu module attribute.
    with mock.patch(
        "torch_npu.npu_scatter_nd_update_",
        side_effect=_fake_scatter_nd_update,
        create=True,
    ) as scatter:
        A5DeviceAdaptor.dsa_kv_compress_scatter(cache, x, formatted)
    scatter.assert_called_once()
    assert captured["indices"].shape == (3, 2)
    assert captured["updates"].shape == (3, 1, 2)
    # -1 pads must reach the op unchanged (no remap-to-token0).
    torch.testing.assert_close(
        captured["indices"],
        torch.tensor([[0, 5], [-1, -1], [-1, -1]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        captured["updates"],
        torch.tensor([[[1.0, 2.0]], [[9.0, 9.0]], [[8.0, 8.0]]], dtype=torch.bfloat16),
    )
    torch.testing.assert_close(cache[0, 5], x[0])
    assert torch.count_nonzero(cache) == 2


def test_sparse_flash_mla_requires_cann_9_2():
    import sys

    import vllm_ascend.ops.sparse_flash_mla as sparse_flash_mla_mod

    sparse_flash_mla_mod._get_sparse_flash_mla_ops.cache_clear()
    # sys.modules[name] = None makes `import name` raise ImportError.
    with (
        mock.patch.dict(sys.modules, {"cann_ops_transformer": None}),
        pytest.raises(RuntimeError, match="CANN 9.2"),
    ):
        sparse_flash_mla_mod._get_sparse_flash_mla_ops()

    sparse_flash_mla_mod._get_sparse_flash_mla_ops.cache_clear()


def test_sparse_flash_mla_wrappers_adapt_dsa_kwargs():
    attention_op = mock.MagicMock(return_value=("output", "lse"))
    metadata_op = mock.MagicMock(return_value="metadata")
    seqused_kv = torch.tensor([10, 65], dtype=torch.int32)

    with mock.patch(
        "vllm_ascend.ops.sparse_flash_mla._get_sparse_flash_mla_ops",
        return_value=(attention_op, metadata_op),
    ):
        metadata = sparse_flash_mla_metadata(
            num_heads_q=64,
            num_heads_kv=1,
            head_dim=512,
            seqused_kv=seqused_kv,
            max_seqlen_kv=65,
            cmp_ratio=4,
            kv_quant_mode=1,
            device="npu:0",
        )
        output = sparse_flash_mla(
            torch.empty(1, 64, 512, dtype=torch.bfloat16),
            seqused_kv=seqused_kv,
            cmp_ratio=4,
            kv_quant_mode=1,
            tile_size=64,
            rope_head_dim=64,
        )

    assert metadata == "metadata"
    assert output == ("output", "lse")

    metadata_kwargs = metadata_op.call_args.kwargs
    torch.testing.assert_close(metadata_kwargs["seqused_ori_kv"], seqused_kv)
    torch.testing.assert_close(metadata_kwargs["seqused_cmp_kv"], seqused_kv // 4)
    torch.testing.assert_close(metadata_kwargs["cmp_residual_kv"], seqused_kv % 4)
    assert metadata_kwargs["max_seqlen_ori_kv"] == 65
    assert metadata_kwargs["max_seqlen_cmp_kv"] == 16
    assert "kv_quant_mode" not in metadata_kwargs
    assert "device" not in metadata_kwargs

    attention_kwargs = attention_op.call_args.kwargs
    torch.testing.assert_close(attention_kwargs["seqused_ori_kv"], seqused_kv)
    assert "kv_quant_mode" not in attention_kwargs
    assert "tile_size" not in attention_kwargs
    assert "rope_head_dim" not in attention_kwargs


def test_a5_bf16_dsa_scatter_uses_block_offset_mapping():
    cache = torch.zeros(3, 4, 1, 2, dtype=torch.bfloat16)
    updates = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]],
        dtype=torch.bfloat16,
    )
    slot_mapping = torch.tensor([[0, 1], [2, 3], [1, 0]], dtype=torch.int32)

    def _fake_scatter_nd_update(var, indices, updates_tensor):
        # CPU stand-in for torch_npu.npu_scatter_nd_update_.
        var[indices[:, 0], indices[:, 1]] = updates_tensor

    with mock.patch(
        "torch_npu.npu_scatter_nd_update_",
        side_effect=_fake_scatter_nd_update,
        create=True,
    ) as scatter_mock:
        A5DeviceAdaptor.dsa_kv_compress_scatter(cache, updates, slot_mapping)

    scatter_mock.assert_called_once()
    call_args = scatter_mock.call_args.args
    assert call_args[0] is cache
    assert call_args[1].dtype == torch.int64
    assert call_args[1].shape == slot_mapping.shape
    torch.testing.assert_close(cache[0, 1], updates[0])
    torch.testing.assert_close(cache[2, 3], updates[1])
    torch.testing.assert_close(cache[1, 0], updates[2])
