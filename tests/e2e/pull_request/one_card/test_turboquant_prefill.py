# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NPU coverage of bounded TQ expansion with shared and recycled pages."""

import pytest
import torch
import torch_npu

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.turboquant_prefill import expand_turboquant_pages, workspace_shapes
from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import TurboQuantLatent
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("page", [64, 128, 256, 512])
@pytest.mark.parametrize("empty_request", [False, True])
@pytest.mark.parametrize("invalid_topk", [False, True])
@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("topk", [1024, 2048])
@torch.inference_mode()
def test_shared_prefix_pages_tail_and_attention(page, empty_request, invalid_topk, heads, topk):
    init_device_properties_triton()
    enable_custom_op()
    torch.manual_seed(81)
    lengths = [2051, 3073] + ([0] if empty_request else [])
    requests = len(lengths)
    columns = (max(lengths) + page - 1) // page
    physical_pages = requests * columns
    physical_rows = physical_pages * page
    store = TurboQuantLatent()
    packed = store.compress(torch.randn(physical_rows, 512, device="npu", dtype=torch.bfloat16)).reshape(-1, 258)
    scales = packed[:, 256:].contiguous().view(torch.float16)
    rope = (torch.randn(physical_rows, 64, device="npu") / scales.float()).bfloat16()
    cache = torch.cat((packed[:, :256], rope.view(torch.uint8), packed[:, 256:]), -1)
    cache = cache.view(physical_pages, page, 1, 386).view(torch.int8)
    before = cache.cpu().clone()
    # Permute physical pages and share a 1024-token immutable prefix. Poison
    # unused table entries so an unmasked tail read cannot silently pass.
    table = torch.full((requests, columns + 3), -1, dtype=torch.int32)
    table[:, :columns] = torch.randperm(physical_pages).reshape(requests, columns)
    table[1, : 1024 // page] = table[0, : 1024 // page]
    for request, length in enumerate(lengths):
        table[request, (length + page - 1) // page :] = -1
    table_npu = table.npu()[:, :columns]
    klens = torch.tensor(lengths, device="npu", dtype=torch.int32)
    shapes = workspace_shapes(requests, max(lengths), 128, 16 * 1024**2)
    assert shapes is not None
    nope, expanded_rope, output_table = [torch.empty(shape, dtype=dtype, device="npu") for shape, dtype in shapes]
    expand_turboquant_pages(cache, table_npu, klens, store.centroids, nope, expanded_rope, output_table)
    padded_length = output_table.shape[1] * 128
    raw = before.view(torch.uint8).reshape(-1, 386)
    codes = torch.stack((raw[:, :256] & 15, raw[:, :256] >> 4), -1).flatten(1).long()
    scale = raw[:, 384:].contiguous().view(torch.float16).float()
    expected_nope = (store.centroids.cpu().bfloat16().float()[codes] * scale).bfloat16()
    expected_rope = (raw[:, 256:384].contiguous().view(torch.bfloat16).float() * scale).bfloat16()
    actual_nope = nope.cpu().reshape(requests, padded_length, 512)
    actual_rope = expanded_rope.cpu().reshape(requests, padded_length, 64)
    for request, length in enumerate(lengths):
        position = torch.arange(length)
        physical = table[request, position // page].long() * page + position % page
        torch.testing.assert_close(actual_nope[request, :length], expected_nope[physical], rtol=0, atol=0)
        torch.testing.assert_close(actual_rope[request, :length], expected_rope[physical], rtol=0, atol=0)
        assert not actual_nope[request, length:].count_nonzero()
        assert not actual_rope[request, length:].count_nonzero()
    torch.testing.assert_close(output_table.cpu().flatten(), torch.arange(nope.shape[0], dtype=torch.int32))
    torch.testing.assert_close(cache.cpu(), before, rtol=0, atol=0)
    if empty_request:
        return
    q = torch.randn(64, heads, 512, device="npu", dtype=torch.bfloat16)
    qr = torch.randn(64, heads, 64, device="npu", dtype=torch.bfloat16)
    indices = torch.arange(topk, dtype=torch.int32, device="npu").repeat(64, 1).unsqueeze(1)
    if invalid_topk:
        indices[:, :, -17:] = -1
    common = dict(
        actual_seq_lengths_query=torch.tensor([32, 64], dtype=torch.int32, device="npu"),
        actual_seq_lengths_kv=klens,
        scale_value=576**-0.5,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
    )
    reference = torch_npu.npu_kv_quant_sparse_flash_attention(
        torch.cat((q, qr), -1), cache, cache, indices, block_table=table_npu,
        key_quant_mode=3, value_quant_mode=3, quant_scale_repo_mode=1, tile_size=128, rope_head_dim=64, **common,
    )
    actual = torch.ops._C_ascend.npu_sparse_flash_attention(
        q, nope, nope, indices, block_table=output_table, query_rope=qr, key_rope=expanded_rope, **common,
    )
    reference = (reference[0] if isinstance(reference, tuple) else reference).float().cpu()
    actual = (actual[0] if isinstance(actual, tuple) else actual).float().cpu()
    assert torch.isfinite(actual).all()
    if not invalid_topk:
        assert ((actual - reference).norm() / reference.norm()).item() < 0.02
    # The installed TQ kernel has a separately reproduced invalid-Top-K
    # discrepancy. Use direct attention as the correctness oracle, including
    # causal masking and -1 entries, rather than encoding that discrepancy.
    for request, length in enumerate(lengths):
        start, end = request * 32, (request + 1) * 32
        selected = indices[start, 0].long()
        valid = selected >= 0
        chosen = selected.clamp_min(0)
        keys = actual_nope[request].npu()[chosen].float()
        ropes = actual_rope[request].npu()[chosen].float()
        scores = torch.einsum("thd,kd->thk", q[start:end].float(), keys)
        scores += torch.einsum("thd,kd->thk", qr[start:end].float(), ropes)
        scores *= 576**-0.5
        causal = selected[None, :] <= (length - 32 + torch.arange(32, device="npu"))[:, None]
        scores.masked_fill_(~(valid[None, None, :] & causal[:, None, :]), float("-inf"))
        expected = torch.einsum("thk,kd->thd", scores.softmax(-1), keys).cpu()
        assert ((actual[start:end] - expected).norm() / expected.norm()).item() < 0.01


@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("clipped", [False, True])
@torch.inference_mode()
def test_current_bf16_mixed_with_quantized_history(page, history, clipped):
    """Check full prefill, prefix/chunks, rank-local lengths and poisoned fresh pages."""
    init_device_properties_triton()
    enable_custom_op()
    torch.manual_seed(91)
    full_lengths = [2051, 3073, 0]
    new_counts = [32, 65, 0] if history else full_lengths
    prefix_lengths = [length - count for length, count in zip(full_lengths, new_counts)]
    used_lengths = [prefix_lengths[0] + 16, prefix_lengths[1] + 22, 0] if clipped else full_lengths
    columns = (max(full_lengths) + page - 1) // page
    table = torch.arange(3 * columns, dtype=torch.int32).reshape(3, columns)
    if history:
        table[1, :1024 // page] = table[0, :1024 // page]
    store = TurboQuantLatent()
    nrows = 3 * columns * page
    packed = store.compress(torch.randn(nrows, 512, dtype=torch.bfloat16, device="npu")).reshape(-1, 258)
    scale = packed[:, 256:].contiguous().view(torch.float16)
    saved_rope = (torch.randn(nrows, 64, device="npu") / scale.float()).bfloat16()
    cache = torch.cat((packed[:, :256], saved_rope.view(torch.uint8), packed[:, 256:]), -1)
    cache = cache.view(3 * columns, page, 1, 386).view(torch.int8)
    before = cache.cpu().clone()
    codes = torch.stack((packed[:, :256] & 15, packed[:, :256] >> 4), -1).flatten(1).long()
    decoded = (store.centroids.bfloat16().float()[codes] * scale.float()).bfloat16().cpu()
    decoded_rope = (saved_rope.float() * scale.float()).bfloat16().cpu()
    # Poison every page that only contains fresh KV. Expansion must not read it.
    for i, prefix in enumerate(prefix_lengths):
        table[i, (prefix + page - 1) // page:] = -1
    current = torch.randn(sum(new_counts), 576, dtype=torch.bfloat16, device="npu")
    current_cpu = current.cpu()
    shapes = workspace_shapes(3, max(full_lengths), 128, 32 * 1024**2)
    nope, rope, out_table = [torch.empty(shape, dtype=dtype, device="npu") for shape, dtype in shapes]
    qends = torch.tensor(new_counts, dtype=torch.int32, device="npu").cumsum(0).int()
    used = torch.tensor(used_lengths, dtype=torch.int32, device="npu")
    expand_turboquant_pages(
        cache, table.npu(), used, store.centroids, nope, rope, out_table,
        current_kv=current, current_query_lens=qends,
        full_seq_lens=torch.tensor(full_lengths, dtype=torch.int32, device="npu"),
    )
    padded = out_table.shape[1] * 128
    expected = torch.zeros(3, padded, 576, dtype=torch.bfloat16)
    start = 0
    for i, (prefix, count, length) in enumerate(zip(prefix_lengths, new_counts, used_lengths)):
        pos = torch.arange(prefix)
        physical = table[i, pos // page].long() * page + pos % page
        expected[i, :prefix, :512] = decoded[physical]
        expected[i, :prefix, 512:] = decoded_rope[physical]
        expected[i, prefix:length] = current_cpu[start:start + length - prefix]
        start += count
    torch.testing.assert_close(nope.cpu().reshape(3, padded, 512), expected[:, :, :512], rtol=0, atol=0)
    torch.testing.assert_close(rope.cpu().reshape(3, padded, 64), expected[:, :, 512:], rtol=0, atol=0)
    torch.testing.assert_close(cache.cpu(), before, rtol=0, atol=0)
    torch.testing.assert_close(out_table.cpu().flatten(), torch.arange(nope.shape[0], dtype=torch.int32))

    # Sparse attention must consume the mixed cache in the same coordinates;
    # compare against independent FP32 attention rather than the old TQ kernel.
    q = torch.randn(32, 64, 512, dtype=torch.bfloat16, device="npu")
    qr = torch.randn(32, 64, 64, dtype=torch.bfloat16, device="npu")
    indices = torch.arange(2048, dtype=torch.int32, device="npu").repeat(32, 1).unsqueeze(1)
    # Include the fresh suffix of both requests, including the request whose
    # historical prefix exceeds Top-K. Keep future entries for causal masking.
    for i, length in enumerate(used_lengths[:2]):
        indices[i * 16:(i + 1) * 16] += max(length - 2048, 0)
    indices[:, :, -1] = -1
    output = torch.ops._C_ascend.npu_sparse_flash_attention(
        q, nope, nope, indices, block_table=out_table,
        actual_seq_lengths_query=torch.tensor([16, 32, 32], dtype=torch.int32, device="npu"),
        actual_seq_lengths_kv=used, query_rope=qr, key_rope=rope, scale_value=576**-0.5,
        sparse_block_size=1, layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2,
    )
    output = output[0] if isinstance(output, tuple) else output
    for i, length in enumerate(used_lengths[:2]):
        selected = indices[i * 16, 0].long()
        keys = expected[i].npu()[selected.clamp_min(0)].float()
        scores = torch.einsum("thd,kd->thk", q[i * 16:(i + 1) * 16].float(), keys[:, :512])
        scores += torch.einsum("thd,kd->thk", qr[i * 16:(i + 1) * 16].float(), keys[:, 512:])
        scores *= 576**-0.5
        visible = selected[None, :] <= (length - 16 + torch.arange(16, device="npu"))[:, None]
        scores.masked_fill_(~((selected >= 0)[None, None, :] & visible[:, None, :]), float("-inf"))
        reference = torch.einsum("thk,kd->thd", scores.softmax(-1), keys[:, :512])
        assert ((output[i * 16:(i + 1) * 16].float() - reference).norm() / reference.norm()).item() < 0.01
