from unittest import mock

import pytest
import torch

from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor
from vllm_ascend.ops.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata


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


def test_a5_dsa_sparse_attention_selects_ops_by_kv_dtype():
    assert A5DeviceAdaptor.get_dsa_sparse_attn_op(torch.bfloat16) is sparse_flash_mla
    assert A5DeviceAdaptor.get_dsa_sparse_attn_metadata_op(torch.bfloat16) is sparse_flash_mla_metadata
    assert A5DeviceAdaptor.get_dsa_kv_layout(torch.bfloat16) == "PA_BBND"

    assert A5DeviceAdaptor.get_dsa_sparse_attn_op(torch.float8_e4m3fn) is not sparse_flash_mla
    assert A5DeviceAdaptor.get_dsa_kv_layout(torch.float8_e4m3fn) == "PA_ND"


def test_sparse_flash_mla_requires_cann92_torch_npu_apis(monkeypatch):
    import vllm_ascend.ops.sparse_flash_mla as sparse_flash_mla_mod

    sparse_flash_mla_mod._get_sparse_flash_mla_ops.cache_clear()
    monkeypatch.delattr(sparse_flash_mla_mod.torch_npu, "npu_sparse_flash_mla", raising=False)
    monkeypatch.delattr(sparse_flash_mla_mod.torch_npu, "npu_sparse_flash_mla_metadata", raising=False)

    with pytest.raises(RuntimeError, match="matching CANN 9.2"):
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

    A5DeviceAdaptor.dsa_kv_compress_scatter(cache, updates, slot_mapping)

    torch.testing.assert_close(cache[0, 1], updates[0])
    torch.testing.assert_close(cache[2, 3], updates[1])
    torch.testing.assert_close(cache[1, 0], updates[2])
