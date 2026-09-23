# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Official QLI pair accuracy with the framework custom OPP still enabled.

Run only on an available Ascend 950 device with matching CANN libraries.
This test does not remove custom OPP paths or import cann_ops_transformer.
"""

from functools import partial
from statistics import median

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("quant_mode", [1, 5])
@pytest.mark.parametrize("batch,qlen,klen", [(1, 1, 128), (1, 8, 4096), (2, 1, 4096)])
@torch.inference_mode()
def test_cann_qli_against_cpu(quant_mode, batch, qlen, klen, record_property):
    enable_custom_op()
    torch.manual_seed(123)
    heads, dim, block_size, topk = 64, 128, 128, 2048
    total_q = batch * qlen
    blocks_per_seq = klen // block_size

    def quantized(shape):
        if quant_mode == 1:
            reference = torch.randint(-4, 5, shape).float()
            return reference.to(torch.float8_e4m3fn).to("npu"), reference
        codes = torch.randint(0, 16, shape, dtype=torch.uint8)
        table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        reference = table[codes.long()]
        packed = codes[..., ::2] | (codes[..., 1::2] << 4)
        return packed.to("npu").view(torch.float4_e2m1fn_x2), reference

    query, query_ref = quantized((total_q, heads, dim))
    key, key_ref = quantized((batch * blocks_per_seq, block_size, 1, dim))
    weights = torch.full((total_q, heads), 1.0 / heads, device="npu")
    if quant_mode == 1:
        qscale = torch.ones((total_q, heads), device="npu")
        kscale = torch.ones((batch * blocks_per_seq, block_size, 1), device="npu")
    else:
        qscale = torch.full((total_q, heads, dim // 64, 2), 127, dtype=torch.uint8, device="npu")
        kscale = torch.full((batch * blocks_per_seq, block_size, 1, dim // 64, 2), 127, dtype=torch.uint8, device="npu")
        qscale = qscale.view(torch.float8_e8m0fnu)
        kscale = kscale.view(torch.float8_e8m0fnu)
    # Non-identity physical page mapping, including across batch boundaries.
    pages = torch.randperm(batch * blocks_per_seq).reshape(batch, blocks_per_seq)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device="npu") * qlen
    used_k = torch.full((batch,), klen, dtype=torch.int32, device="npu")
    make_metadata = partial(
        torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata_cann,
        heads,
        1,
        dim,
        topk,
        quant_mode,
        cu_seqlens_q=cu_q,
        seqused_k=used_k,
        batch_size=batch,
        max_seqlen_q=qlen,
        max_seqlen_k=klen,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
    )
    metadata = make_metadata()
    run_qli = partial(
        torch.ops._C_ascend.npu_quant_lightning_indexer_v2_cann,
        query,
        key,
        weights,
        qscale,
        kscale,
        topk,
        quant_mode,
        cu_seqlens_q=cu_q,
        seqused_k=used_k,
        block_table=pages.to(device="npu", dtype=torch.int32),
        metadata=metadata,
        max_seqlen_q=qlen,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=1,
        return_value=0,
    )
    indices, values = run_qli()
    torch.npu.synchronize()
    assert indices.shape == (total_q, 1, topk)
    assert indices.dtype == torch.int32
    assert values.numel() == 0
    indices = indices.cpu().long()
    for b in range(batch):
        logical_key = key_ref[pages[b]].reshape(klen, dim)
        for t in range(qlen):
            row = b * qlen + t
            valid_keys = klen - qlen + t + 1
            scores = (query_ref[row] @ logical_key[:valid_keys].T).relu().mean(0)
            selected = indices[row, 0]
            selected = selected[selected >= 0]
            count = min(topk, valid_keys)
            assert selected.numel() == count
            assert selected.unique().numel() == count
            assert (selected < valid_keys).all()
            # Compare selected scores, not the arbitrary ordering of tied keys.
            expected = scores.topk(count).values
            actual = scores[selected].sort(descending=True).values
            # FP4's quantized score accumulation may differ slightly at the
            # top-k cutoff from the CPU reference; 2% covers that rounding.
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)

    def run_pair():
        return run_qli(metadata=make_metadata())

    # Device-event samples after correctness checks and warmup. These are
    # single-operator measurements, not model throughput or speedup claims.
    for name, operation in (("metadata", make_metadata), ("qli", run_qli), ("pair", run_pair)):
        for _ in range(10):
            operation()
        torch.npu.synchronize()
        samples_us = []
        for _ in range(5):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                operation()
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000 / 50)
        record_property(f"{name}_median_us", median(samples_us))
        record_property(f"{name}_samples_us", str(samples_us))
