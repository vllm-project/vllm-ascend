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


@pytest.mark.parametrize(
    ("query_lengths", "key_lengths"),
    [
        ([256], [37]),  # Enough S1 tiles for non-LD; pad n<K candidates.
        ([1], [4096]),  # Split the long key sequence across cores (LD).
        ([1, 1, 1], [0, 0, 0]),  # Entire local rank empty, TND with B>1.
        ([1, 1, 1], [0, 37, 65]),  # Per-request empty and partially filled pages.
    ],
)
@torch.inference_mode()
def test_quant_lightning_indexer_v2_dcp_fp32_candidates(query_lengths, key_lengths):
    """DCP candidates expose unrounded final scores paired with existing indices."""
    soc_version = torch_npu.npu.get_soc_version()
    if not (220 <= soc_version <= 225 or 250 <= soc_version <= 256):
        pytest.skip("FP32 DCP candidates require the arch22 QLI kernel")
    num_heads = 64
    head_dim = 128
    block_size = 32
    topk = 512
    batch_size = len(query_lengths)
    max_key_length = max(1, max(key_lengths))
    blocks_per_request = (max_key_length + block_size - 1) // block_size
    block_num = batch_size * blocks_per_request
    total_queries = sum(query_lengths)

    # An exactly representable dot product/reduction isolates score extraction
    # from accumulation tolerances: score(row) == key_scale(row) / 1024.
    # The factor is the existing A2 QK fixpipe scale, shared by every rank.
    query = torch.zeros((total_queries, num_heads, head_dim), dtype=torch.int8, device="npu")
    query[..., 0] = 1
    weights = torch.ones((total_queries, num_heads), dtype=torch.float16, device="npu")
    query_scale = torch.full_like(weights, 1 / num_heads)
    key = torch.zeros((block_num, block_size, 1, head_dim), dtype=torch.int8, device="npu")
    key[..., 0] = 1
    key_scale_cpu = (
        (1 + torch.arange(block_num * block_size, dtype=torch.float32) / 1024)
        .to(torch.float16)
        .view(block_num, block_size, 1)
    )
    key_scale = key_scale_cpu.npu()
    cu_seqlens_q = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32, device="npu")
    seqused_k = torch.tensor(key_lengths, dtype=torch.int32, device="npu")
    block_table = torch.arange(block_num, dtype=torch.int32, device="npu").view(batch_size, blocks_per_request)
    cmp_residual_k = torch.zeros(batch_size, dtype=torch.int32, device="npu")
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q=num_heads,
        num_heads_k=1,
        head_dim=head_dim,
        topk=topk,
        quant_mode=2,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max_key_length * 4,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=4,
    )
    # Metadata carries LD task flags after the 36 AIC rows (8 integers each).
    # Check the intended score-export branch instead of relying on shape alone.
    metadata_cpu = metadata.cpu().reshape(-1)
    ld_enabled = metadata_cpu[36 * 8 : 36 * 8 + 72 * 8].reshape(72, 8)[:, 0].any()
    if query_lengths == [256]:
        assert not ld_enabled
    elif query_lengths == [1]:
        assert ld_enabled
    kwargs = dict(
        query=query,
        key=key,
        weights=weights,
        query_dequant_scale=query_scale,
        key_dequant_scale=key_scale,
        topk=topk,
        quant_mode=2,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        metadata=metadata,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=0,
        cmp_ratio=4,
    )
    indices, scores = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(**kwargs, return_value=2)
    assert scores.dtype == torch.float32
    assert scores.shape == indices.shape == (total_queries, 1, topk)
    indices_cpu, scores_cpu = indices.cpu().squeeze(1), scores.cpu().squeeze(1)
    offset = 0
    for batch_idx, (num_queries, num_keys) in enumerate(zip(query_lengths, key_lengths)):
        valid_count = min(topk, num_keys)
        row_scores = key_scale_cpu.view(batch_size, -1)[batch_idx, :num_keys].float() / 1024
        for query_idx in range(offset, offset + num_queries):
            row_indices = indices_cpu[query_idx]
            row_values = scores_cpu[query_idx]
            assert torch.all(row_indices[:valid_count] >= 0)
            assert torch.all(row_indices[:valid_count] < num_keys)
            torch.testing.assert_close(row_values[:valid_count], row_scores[row_indices[:valid_count]], rtol=0, atol=0)
            torch.testing.assert_close(row_values[:valid_count], row_scores.topk(valid_count).values, rtol=0, atol=0)
            assert torch.all(row_indices[valid_count:] == -1)
            assert torch.all(torch.isneginf(row_values[valid_count:]))
        offset += num_queries

    if any(key_lengths):
        # Mode 0 is unchanged and returns exactly the same local indices.
        old_indices, old_scores = torch.ops._C_ascend.npu_quant_lightning_indexer_v2(**kwargs, return_value=0)
        torch.testing.assert_close(indices, old_indices, rtol=0, atol=0)
        assert old_scores.numel() == 0
        finite_scores = scores_cpu[torch.isfinite(scores_cpu)]
        assert torch.any(finite_scores != finite_scores.to(torch.bfloat16).float())
