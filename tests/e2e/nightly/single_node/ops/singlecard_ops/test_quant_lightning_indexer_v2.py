# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

enable_custom_op()

from vllm_ascend.device.device_op import A5DeviceAdaptor  # noqa: E402
from vllm_ascend.ops.triton.sharded_indexer import build_local_key_lengths  # noqa: E402


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


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("query_count", [1, 2, 5])
@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="QuantLightningIndexer candidate mode is supported only on Ascend A2/A3",
)
def test_candidate_consumer_preserves_each_query_mask(ratio, query_count):
    """A multi-query tile must reload each row's candidates across KV tiles."""
    torch.manual_seed(71)
    device = "npu"
    heads, width, topk, block_size, kv_len = 64, 128, 1024, 128, 6144
    query = torch.randint(-90, 90, (query_count, heads, width), dtype=torch.int8, device=device)
    key = torch.randint(-90, 90, (kv_len // block_size, block_size, 1, width), dtype=torch.int8, device=device)
    weights = torch.rand(query_count, heads, dtype=torch.float16, device=device)
    query_scale = torch.full((query_count, heads), 0.01, dtype=torch.float16, device=device)
    key_scale = torch.full(key.shape[:-1], 0.01, dtype=torch.float16, device=device)
    qsl = torch.tensor([0, query_count], dtype=torch.int32, device=device)
    lengths = torch.tensor([kv_len], dtype=torch.int32, device=device)
    residual = torch.zeros_like(lengths) if ratio == 2 else None
    table = torch.arange(kv_len // block_size, dtype=torch.int32, device=device).unsqueeze(0)
    # Alternate masks: row 0 excludes the middle KV tile, row 1 excludes the
    # first. A stale mask remains plausible but selects forbidden positions.
    masks = [torch.cat((torch.arange(256), torch.arange(512, 768))), torch.arange(256, 768)]
    candidates_cpu = torch.stack([masks[row % 2] for row in range(query_count)]).int()
    candidates = candidates_cpu.unsqueeze(1).to(device)
    common = dict(
        cu_seqlens_q=qsl,
        seqused_k=lengths,
        cmp_residual_k=residual,
        max_seqlen_q=query_count,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        heads,
        1,
        width,
        topk,
        2,
        batch_size=1,
        max_seqlen_k=kv_len,
        device=str(query.device),
        **common,
    )
    selected, _, _ = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        topk,
        2,
        block_table=table,
        metadata=metadata,
        candidate_topk_index=candidates,
        candidate_mode=2,
        candidate_topk_blocks=512,
        candidate_block_size=8,
        **common,
    )
    for row, indices in enumerate(selected.cpu().reshape(query_count, topk)):
        assert indices.unique().numel() == topk
        visible = (kv_len * ratio - query_count + row + 1) // ratio
        assert ((indices >= 0) & (indices < visible)).all()
        assert torch.isin(indices // 8, candidates_cpu[row]).all(), f"wrong candidate row at query {row}"


@pytest.mark.parametrize("queries", [1, 5])
@pytest.mark.parametrize("context", [1536, 8192, 32768])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("topk", [64, 512])
@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="QLI arch22 vector-repeat and adaptive Top-K tests require Ascend A2/A3",
)
@torch.inference_mode()
def test_qli_candidate_vector_repeats(queries, context, ratio, mode, topk):
    """Check candidate padding, partial causal blocks and strided PA caches."""
    generator = torch.Generator().manual_seed(1234)
    heads, width = 32, 128
    block_size = 128 if ratio == 1 else 64
    query_cpu = torch.randint(-90, 90, (queries, heads, width), dtype=torch.int8, generator=generator)
    key_cpu = torch.randint(
        -90, 90, (context // block_size, block_size, 1, width), dtype=torch.int8, generator=generator
    )
    weights_cpu = torch.rand((queries, heads), generator=generator).half()
    query_scale_cpu = torch.full((queries, heads), 0.01, dtype=torch.float16)
    key_scale_cpu = torch.full(key_cpu.shape[:-1], 0.01, dtype=torch.float16)
    stride = 147712 if ratio == 1 else 131072
    key = torch.empty_strided(key_cpu.shape, (stride, 128, 128, 1), dtype=torch.int8, device="npu")
    key.copy_(key_cpu)
    key_scale = torch.empty_strided(key_scale_cpu.shape, (stride // 2, 1, 1), dtype=torch.float16, device="npu")
    key_scale.copy_(key_scale_cpu)
    candidates_cpu = torch.stack([torch.randperm(context // 8, generator=generator)[:2048] for _ in range(queries)])
    candidates_cpu = torch.nn.functional.pad(candidates_cpu, (0, 2048 - candidates_cpu.shape[1]), value=-1)
    candidates = candidates_cpu.int().unsqueeze(1).npu() if mode == 2 else None
    common = dict(
        cu_seqlens_q=torch.tensor([0, queries], dtype=torch.int32, device="npu"),
        seqused_k=torch.tensor([context], dtype=torch.int32, device="npu"),
        cmp_residual_k=torch.tensor([0], dtype=torch.int32, device="npu") if ratio == 2 else None,
        max_seqlen_q=queries,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=ratio,
    )
    metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
        heads, 1, width, topk, 2, batch_size=1, max_seqlen_k=context, device="npu:0", **common
    )
    selected, _, candidate_output = torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
        query_cpu.npu(),
        key,
        weights_cpu.npu(),
        query_scale_cpu.npu(),
        key_scale,
        topk,
        2,
        block_table=torch.arange(context // block_size, dtype=torch.int32, device="npu").unsqueeze(0),
        metadata=metadata,
        candidate_topk_index=candidates,
        candidate_mode=mode,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        **common,
    )
    dot = torch.einsum("qhd,kd->qhk", query_cpu.float(), key_cpu.reshape(context, width).float())
    relu = (dot / 1024).clamp_min(0).half().float()
    scores = (relu * (weights_cpu * query_scale_cpu).float().unsqueeze(-1)).sum(1)
    scores *= key_scale_cpu.reshape(context).float()
    selected = selected.cpu().reshape(queries, topk).long()
    for row in range(queries):
        visible = (context * ratio - queries + row + 1) // ratio
        scores[row, visible:] = -torch.inf
        if mode == 2:
            allowed = torch.isin(torch.arange(context) // 8, candidates_cpu[row])
            scores[row, ~allowed] = -torch.inf
        indices = selected[row]
        assert indices.unique().numel() == topk
        assert ((indices >= 0) & (indices < visible)).all()
        torch.testing.assert_close(
            scores[row, indices].sort(descending=True).values,
            scores[row].topk(topk).values,
            rtol=1e-4,
            atol=1e-6,
        )
        if mode == 1:
            blocks = candidate_output.cpu().reshape(queries, -1)[row]
            assert ((blocks == -1) | (blocks >= 0)).all()
            blocks = blocks[blocks >= 0].long()
            expected_blocks = (visible + 7) // 8
            assert blocks.unique().numel() == min(expected_blocks, 2048)
            assert ((blocks >= 0) & (blocks < expected_blocks)).all()
            if expected_blocks <= 2048:
                torch.testing.assert_close(blocks.sort().values, torch.arange(expected_blocks))


@pytest.mark.parametrize("topk", [1, 63, 65, 1024, 2048])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="QLI arch22 vector-repeat and adaptive Top-K tests require Ascend A2/A3",
)
def test_qli_adaptive_topk_boundaries(topk, ratio, mode):
    """Exercise partial 64-element widths and the unchanged maximum width."""
    test_qli_candidate_vector_repeats(queries=5, context=8192, ratio=ratio, mode=mode, topk=topk)


@pytest.mark.parametrize(
    "arch,enable_dcp",
    [
        pytest.param(AscendDeviceType.A3, False, id="a3-int8"),
        pytest.param(AscendDeviceType.A3, True, id="a3-int8-dcp"),
        pytest.param(AscendDeviceType.A5, False, id="a5-fp8"),
    ],
)
@torch.inference_mode()
def test_quant_lightning_indexer_v2_accuracy(arch, enable_dcp):
    """Compare paged QLI indices and scores with the operator's CPU formula."""
    if get_ascend_device_type() != arch:
        pytest.skip(f"requires {arch.name} NPU")
    ops = torch.ops._C_ascend
    if not hasattr(ops, "npu_quant_lightning_indexer_v2") or not hasattr(
        ops, "npu_quant_lightning_indexer_v2_metadata"
    ):
        pytest.skip("requires QLI v2 custom operators")

    generator = torch.Generator().manual_seed(20878)
    rows, heads, width, page, topk = 2, 64, 128, 128, 64
    world, rank, interleave = (2, 1, 8) if enable_dcp else (1, 0, 1)
    global_length = 1001 if enable_dcp else 1024
    local_positions = [index for index in range(global_length) if (index // interleave) % world == rank]
    local_length = len(local_positions)
    blocks = (local_length + page - 1) // page
    quant_mode = 1 if arch == AscendDeviceType.A5 else 2
    data_dtype = torch.float8_e4m3fn if quant_mode == 1 else torch.int8
    scale_dtype = torch.float32 if quant_mode == 1 else torch.float16
    if quant_mode == 1:
        query_cpu = (torch.randn((rows, heads, width), generator=generator) * 2).to(data_dtype)
        logical_key = (torch.randn((blocks, page, 1, width), generator=generator) * 2).to(data_dtype)
    else:
        query_cpu = torch.randint(-48, 48, (rows, heads, width), generator=generator, dtype=data_dtype)
        logical_key = torch.randint(-48, 48, (blocks, page, 1, width), generator=generator, dtype=data_dtype)
    weights_cpu = (torch.rand((rows, heads), generator=generator) + 0.25).to(scale_dtype)
    query_scale_cpu = (torch.rand((rows, heads), generator=generator) * 0.02 + 0.01).to(scale_dtype)
    logical_key_scale = (torch.rand((blocks, page, 1), generator=generator) + 0.5).to(scale_dtype)
    block_table_cpu = torch.randperm(blocks, generator=generator)
    key_cpu = torch.empty_like(logical_key)
    key_scale_cpu = torch.empty_like(logical_key_scale)
    key_cpu[block_table_cpu] = logical_key
    key_scale_cpu[block_table_cpu] = logical_key_scale

    query, key = query_cpu.npu(), key_cpu.npu()
    weights, query_scale = weights_cpu.npu(), query_scale_cpu.npu()
    key_scale = key_scale_cpu.npu()
    cu_seqlens_q = torch.tensor([0, rows], dtype=torch.int32, device="npu")
    seqused_k = torch.tensor([local_length], dtype=torch.int32, device="npu")
    block_table = block_table_cpu.to(torch.int32).unsqueeze(0).npu()
    dcp = (
        dict(
            enable_dcp=True,
            cp_world=world,
            cp_rank=rank,
            cp_kv_cache_interleave_size=interleave,
            causal_seqlens_kv_global=torch.tensor([global_length], dtype=torch.int32, device="npu"),
        )
        if enable_dcp
        else {}
    )
    common = dict(
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=rows,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
    )
    metadata = ops.npu_quant_lightning_indexer_v2_metadata(
        heads,
        1,
        width,
        topk,
        quant_mode,
        batch_size=1,
        max_seqlen_k=local_length,
        device=str(query.device),
        **common,
        **dcp,
    )
    indices, values = ops.npu_quant_lightning_indexer_v2(
        query,
        key,
        weights,
        query_scale,
        key_scale,
        topk,
        quant_mode,
        block_table=block_table,
        metadata=metadata,
        return_value=1,
        **common,
        **dcp,
    )

    logical_keys = logical_key.reshape(-1, width)[:local_length].float()
    dots = torch.einsum("qhd,kd->qhk", query_cpu.float(), logical_keys)
    if quant_mode == 2:
        dots = (dots / 1024).clamp_min(0).half().float()
    else:
        dots = dots.clamp_min(0)
    weighted_query = (weights_cpu * query_scale_cpu).float()
    scores = (dots * weighted_query.unsqueeze(-1)).sum(1)
    scores *= logical_key_scale.reshape(-1)[:local_length].float()
    if quant_mode == 1:
        scores = scores.bfloat16().float()

    indices_cpu = indices.cpu().reshape(rows, topk).long()
    values_cpu = values.cpu().reshape(rows, topk).float()
    assert values.dtype == (torch.bfloat16 if quant_mode == 1 else torch.float32)
    for row in range(rows):
        visible_end = global_length - rows + row + 1
        visible = sum(position < visible_end for position in local_positions)
        expected = scores[row, :visible].topk(topk).values
        selected = indices_cpu[row]
        assert selected.unique().numel() == topk
        assert ((selected >= 0) & (selected < visible)).all()
        torch.testing.assert_close(
            scores[row, selected].sort(descending=True).values,
            expected,
            rtol=2e-2,
            atol=2e-3,
        )
        torch.testing.assert_close(values_cpu[row], scores[row, selected], rtol=2e-2, atol=2e-3)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="FP8 QLI DCP requires Ascend A5",
)
@pytest.mark.parametrize("interleave", [1, 8, 768])
@pytest.mark.parametrize("rank", [0, 7])
def test_quant_lightning_indexer_v2_dcp_visibility_padding_and_changed_length_replay(interleave, rank):
    torch.manual_seed(42)
    world, heads, dim, page, rows, active, topk = 8, 32, 128, 128, 8, 4, 2048
    local_length = max(page, 2 * interleave)
    blocks = (local_length + page - 1) // page
    query = torch.randn(rows, heads, dim).to(torch.float8_e4m3fn).npu()
    # Key and scales use a shared physical page with non-contiguous block strides.
    page_bytes = page * (dim + 4)
    storage = torch.zeros(blocks * page_bytes, dtype=torch.uint8, device="npu")
    key = storage.view(query.dtype).as_strided((blocks, page, 1, dim), (page_bytes, dim, dim, 1))
    key.copy_(torch.randn(key.shape).to(query.dtype))
    scales = storage.view(torch.float32).as_strided((blocks, page, 1), (page_bytes // 4, 1, 1), page * dim // 4)
    scales.fill_(0.1)
    weights = torch.ones(rows, heads, device="npu")
    query_scale = torch.full_like(weights, 0.1)
    starts = torch.tensor([0, active], dtype=torch.int32, device="npu")
    local = torch.tensor([local_length], dtype=torch.int32, device="npu")
    global_len = torch.tensor([world * interleave + active], dtype=torch.int32, device="npu")
    table = torch.arange(blocks, dtype=torch.int32, device="npu").view(1, -1)
    dcp = dict(
        enable_dcp=True,
        cp_world=world,
        cp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        causal_seqlens_kv_global=global_len,
    )

    def call():
        build_local_key_lengths(global_len, local, world, rank, interleave)
        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            heads,
            1,
            dim,
            topk,
            1,
            cu_seqlens_q=starts,
            seqused_k=local,
            batch_size=1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
            **dcp,
        )
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            query,
            key,
            weights,
            query_scale,
            scales,
            topk,
            1,
            cu_seqlens_q=starts,
            seqused_k=local,
            block_table=table,
            metadata=metadata,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
            return_value=1,
            output_idx_offset=None,
            **dcp,
        )

    def check(outputs, causal_end):
        indices, scores = [x.cpu() for x in outputs]
        assert indices.dtype == torch.int32 and scores.dtype == torch.bfloat16
        assert (indices[active:] == -1).all() and torch.isneginf(scores[active:]).all()
        local_ids = torch.arange(local_length)
        global_ids = (local_ids // interleave * world + rank) * interleave + local_ids % interleave
        for row in range(active):
            expected = local_ids[global_ids < causal_end - active + row + 1]
            valid = indices[row, 0] >= 0
            torch.testing.assert_close(indices[row, 0, valid].sort().values.long(), expected)
            assert torch.isfinite(scores[row, 0, valid]).all()
            assert torch.isneginf(scores[row, 0, ~valid]).all()

    for _ in range(3):
        result = call()
    check(result, world * interleave + active)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = call()
    for length in (0, world * interleave - 1, world * interleave + 13):
        global_len.fill_(length)
        captured[0].fill_(0)
        captured[1].fill_(0)
        graph.replay()
        torch.npu.synchronize()
        check(captured, length)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="FP8 QLI DCP requires Ascend A5",
)
@pytest.mark.parametrize("local_length,rank", [(0, 7), (4097, 0), (4097, 7), (32769, 7)])
def test_quant_lightning_indexer_v2_dcp_matches_local_candidates_and_cpu_scores(local_length, rank):
    world, interleave, heads, dim, page, active, rows, topk = 8, 768, 32, 128, 128, 4, 8, 2048
    blocks = max(1, (local_length + page - 1) // page)
    token = torch.arange(blocks * page)
    # Binary fractions keep the BF16 comparison values exact, including ties.
    key_values = ((token * 17 % 33 - 8).float() / 8).to(torch.float8_e4m3fn).float()
    key_scales = 2.0 ** (token % 3 - 1)
    storage = torch.empty(blocks, page * (dim + 4), dtype=torch.uint8, device="npu")
    key = storage[:, : page * dim].view(torch.float8_e4m3fn).reshape(blocks, page, 1, dim)
    scales = storage[:, page * dim :].view(torch.float32).reshape(blocks, page, 1, 1)
    key.copy_(key_values[:, None].expand(-1, dim).reshape(key.shape))
    scales.copy_(key_scales.reshape(scales.shape))
    query = torch.ones(rows, heads, dim).to(key.dtype).npu()
    weights = torch.ones(rows, heads, device="npu")
    starts = torch.tensor([0, active], dtype=torch.int32, device="npu")
    local = torch.tensor([local_length], dtype=torch.int32, device="npu")
    causal_end = (4 if local_length < 10000 else 31) * world * interleave + 11
    global_len = torch.tensor([causal_end], dtype=torch.int32, device="npu")
    table = torch.arange(blocks, dtype=torch.int32, device="npu").view(1, -1)
    dcp = dict(global_seq_lens=global_len, cp_world=world, cp_rank=rank, cp_interleave=interleave)
    metadata = A5DeviceAdaptor.build_indexer_local_metadata(
        starts, local, num_heads_q=heads, head_dim=dim, topk=topk, **dcp
    )
    outputs = A5DeviceAdaptor.indexer_select_dcp_candidates(
        query, weights, key, scales, table, starts, local, weights, qli_metadata=metadata, topk=topk, **dcp
    )
    local_ids = torch.arange(local_length)
    global_ids = (local_ids // interleave * world + rank) * interleave + local_ids % interleave
    visible = [(global_ids < causal_end - active + row + 1).sum().item() for row in range(active)]
    lengths = torch.tensor(visible + [0] * (rows - active), dtype=torch.int32, device="npu").view(rows, 1)
    old_metadata = A5DeviceAdaptor.build_indexer_local_metadata(
        starts, local, num_heads_q=heads, head_dim=dim, topk=topk
    )
    old = A5DeviceAdaptor.indexer_select_local_candidates(
        query, weights, key, scales, table, starts, local, weights, lengths, qli_metadata=old_metadata, topk=topk
    )
    indices, scores = [tensor.cpu() for tensor in outputs]
    old_indices, old_scores = [tensor.cpu() for tensor in old]
    assert scores.dtype == old_scores.dtype == torch.bfloat16
    reference = (key_values * (heads * dim) * key_scales).clamp_min(0)
    for row, length in enumerate(visible + [0] * (rows - active)):
        valid = indices[row, 0] >= 0
        assert valid.sum().item() == min(length, topk)
        assert torch.isneginf(scores[row, 0, ~valid]).all()
        expected = torch.argsort(reference[:length], descending=True, stable=True)[:topk]
        torch.testing.assert_close(indices[row, 0, valid].sort().values.long(), expected.sort().values)
        torch.testing.assert_close(scores[row, 0, valid].float(), reference[indices[row, 0, valid].long()])
        torch.testing.assert_close(indices[row, 0].sort().values, old_indices[row, 0].sort().values)
        torch.testing.assert_close(scores[row, 0].float().sort().values, old_scores[row, 0].float().sort().values)
