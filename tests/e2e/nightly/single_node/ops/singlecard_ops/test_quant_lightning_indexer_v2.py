import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


@torch.inference_mode()
def test_quant_lightning_indexer_v2_noncontiguous_pa_cache_dim0():
    """QLI v2 must honor the physical block stride of PA_BBND caches."""
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2"):
        pytest.skip("requires the npu_quant_lightning_indexer_v2 custom operator")
    if not hasattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata"):
        pytest.skip("requires the npu_quant_lightning_indexer_v2_metadata custom operator")

    torch.manual_seed(0)
    num_heads_q = 64
    num_heads_k = 1
    head_dim = 128
    block_num = 4
    block_size = 32
    topk = 32

    query = torch.randint(-128, 128, (1, num_heads_q, head_dim), dtype=torch.int8, device="npu")
    weights = torch.rand((1, num_heads_q), dtype=torch.float16, device="npu")
    query_dequant_scale = torch.rand((1, num_heads_q), dtype=torch.float16, device="npu")
    key_contiguous = torch.randint(
        -128,
        128,
        (block_num, block_size, num_heads_k, head_dim),
        dtype=torch.int8,
        device="npu",
    )
    key_scale_contiguous = torch.rand((block_num, block_size, num_heads_k), dtype=torch.float16, device="npu")

    # Keep one physical padding block between logical cache blocks. The QLI v2
    # kernel must use the real axis-0 strides for both key and key scale.
    key_storage = torch.full(
        (block_num * 2, block_size, num_heads_k, head_dim),
        -1,
        dtype=torch.int8,
        device="npu",
    )
    key_scale_storage = torch.full(
        (block_num * 2, block_size, num_heads_k),
        -1.0,
        dtype=torch.float16,
        device="npu",
    )
    key_strided = key_storage[::2]
    key_scale_strided = key_scale_storage[::2]
    key_strided.copy_(key_contiguous)
    key_scale_strided.copy_(key_scale_contiguous)

    assert not key_strided.is_contiguous()
    assert not key_scale_strided.is_contiguous()
    assert key_strided.stride(0) == 2 * block_size * num_heads_k * head_dim
    assert key_scale_strided.stride(0) == 2 * block_size * num_heads_k

    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    seqused_k = torch.tensor([64], dtype=torch.int32, device="npu")
    cmp_residual_k = torch.tensor([1], dtype=torch.int32, device="npu")
    block_table = torch.tensor([[0, 1]], dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        quant_mode=2,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=257,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
    )

    def run(key: torch.Tensor, key_dequant_scale: torch.Tensor) -> torch.Tensor:
        sparse_indices, sparse_values = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            query=query,
            key=key,
            weights=weights,
            query_dequant_scale=query_dequant_scale,
            key_dequant_scale=key_dequant_scale,
            topk=topk,
            quant_mode=2,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            block_table=block_table,
            metadata=metadata,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
            return_value=0,
        )
        assert sparse_values.numel() == 0
        return sparse_indices

    contiguous_indices = run(key_contiguous, key_scale_contiguous)
    strided_indices = run(key_strided, key_scale_strided)

    assert contiguous_indices.shape == (1, num_heads_k, topk)
    assert contiguous_indices.dtype == torch.int32
    torch.testing.assert_close(strided_indices, contiguous_indices, rtol=0, atol=0)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_quant_lightning_indexer_v2_high_physical_page_scale_offset():
    """QLI v2 must not wrap K-scale offsets beyond signed int32."""
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2"):
        pytest.skip("requires the npu_quant_lightning_indexer_v2 custom operator")
    if not hasattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v2_metadata"):
        pytest.skip("requires the npu_quant_lightning_indexer_v2_metadata custom operator")
    soc_version = torch_npu.npu.get_soc_version()
    if not (220 <= soc_version <= 225 or 250 <= soc_version <= 256):
        pytest.skip("the affected arch22 QLI v2 kernel requires an A2/A3 NPU")

    num_heads_q = 64
    num_heads_k = 1
    head_dim = 128
    block_size = 32
    topk = 16
    scale_stride = 73856
    int32_limit = torch.iinfo(torch.int32).max
    first_high_block = int32_limit // scale_stride + 1
    block_num = first_high_block + 2

    # The prefix keeps addresses wrapped by the old int32 multiplication
    # inside this allocation. This nightly regression needs about 8 GiB.
    guard_elements = 1 << 31
    backing_elements = guard_elements + (block_num - 1) * scale_stride + block_size
    required_bytes = backing_elements * torch.tensor([], dtype=torch.float16).element_size()
    free_bytes, _ = torch.npu.mem_get_info()
    if free_bytes < required_bytes + 2 * 1024**3:
        pytest.skip("requires about 10 GiB of free NPU memory")

    query = torch.ones((1, num_heads_q, head_dim), dtype=torch.int8, device="npu")
    weights = torch.ones((1, num_heads_q), dtype=torch.float16, device="npu")
    query_dequant_scale = torch.full(
        (1, num_heads_q),
        1 / 32,
        dtype=torch.float16,
        device="npu",
    )
    key_payload = (
        torch.arange(2 * block_size * num_heads_k * head_dim, device="npu")
        .remainder(17)
        .sub(8)
        .to(torch.int8)
        .reshape(2, block_size, num_heads_k, head_dim)
    )
    scale_payload = torch.linspace(
        0.1,
        1.1,
        2 * block_size,
        dtype=torch.float16,
        device="npu",
    ).reshape(2, block_size, num_heads_k)

    key = torch.empty(
        (block_num, block_size, num_heads_k, head_dim),
        dtype=torch.int8,
        device="npu",
    )
    scale_backing = torch.empty(backing_elements, dtype=torch.float16, device="npu")
    key_dequant_scale = scale_backing.as_strided(
        (block_num, block_size, num_heads_k),
        (scale_stride, 1, 1),
        guard_elements,
    )

    low_blocks = (1, 2)
    high_blocks = (first_high_block, first_high_block + 1)
    for payload_index, (low_block, high_block) in enumerate(zip(low_blocks, high_blocks)):
        key[low_block].copy_(key_payload[payload_index])
        key[high_block].copy_(key_payload[payload_index])
        key_dequant_scale[low_block].copy_(scale_payload[payload_index])
        key_dequant_scale[high_block].copy_(scale_payload[payload_index])

        wrapped_offset = high_block * scale_stride - (1 << 32)
        scale_backing[guard_elements + wrapped_offset : guard_elements + wrapped_offset + block_size].zero_()

    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    seqused_k = torch.tensor([2 * block_size], dtype=torch.int32, device="npu")
    cmp_residual_k = torch.tensor([1], dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        quant_mode=2,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=1,
        max_seqlen_q=1,
        max_seqlen_k=257,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
    )

    def run(
        blocks: tuple[int, int],
        key_cache: torch.Tensor,
        scale_cache: torch.Tensor,
    ) -> torch.Tensor:
        block_table = torch.tensor([blocks], dtype=torch.int32, device="npu")
        sparse_indices, sparse_values = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            query=query,
            key=key_cache,
            weights=weights,
            query_dequant_scale=query_dequant_scale,
            key_dequant_scale=scale_cache,
            topk=topk,
            quant_mode=2,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            block_table=block_table,
            metadata=metadata,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
            return_value=0,
        )
        assert sparse_values.numel() == 0
        return sparse_indices

    low_indices = run(low_blocks, key, key_dequant_scale)
    high_indices = run(high_blocks, key, key_dequant_scale)

    assert first_high_block == 29077
    assert high_blocks[0] * scale_stride > int32_limit
    torch.testing.assert_close(high_indices, low_indices, rtol=0, atol=0)

    del key_dequant_scale, scale_backing, key, key_payload, scale_payload
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
