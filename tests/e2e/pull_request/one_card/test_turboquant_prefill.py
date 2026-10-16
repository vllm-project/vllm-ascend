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


@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("empty_request", [False, True])
@pytest.mark.parametrize("invalid_topk", [False, True])
@torch.inference_mode()
def test_shared_prefix_pages_tail_and_attention(page, empty_request, invalid_topk):
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
    q = torch.randn(64, 64, 512, device="npu", dtype=torch.bfloat16)
    qr = torch.randn(64, 64, 64, device="npu", dtype=torch.bfloat16)
    indices = torch.arange(2048, dtype=torch.int32, device="npu").repeat(64, 1).unsqueeze(1)
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
