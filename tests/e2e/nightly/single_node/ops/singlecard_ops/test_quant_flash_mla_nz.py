# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.attention.flashmla import QuantFlashMLAConfig, QuantFlashMLAImpl, split_flashmla_cache
from vllm_ascend.device.paged_mla_cache_read import load_mla_cache_context_rows


@pytest.mark.parametrize(("causal", "slot_dtype"), [(True, torch.int64), (False, torch.int32)])
@torch.inference_mode()
def test_quant_flash_mla_nz_paged_replay(causal, slot_dtype):
    if not hasattr(torch.ops._C_ascend, "quant_flash_mla_nz"):
        pytest.skip("requires the Ascend 950 native FP8 decode op")

    dtype = torch.float8_e4m3fn
    device = torch.device("npu")
    torch.manual_seed(5823)
    pages = torch.randn(6, 128, 1, 576, dtype=torch.bfloat16).to(dtype)
    # Nonzero storage offset and page padding represent a cache in a shared block allocation.
    raw = torch.full((6, 2, 128 * 576), 91, device=device, dtype=torch.uint8)
    nz = raw[:, 1].view(dtype).view(6, 1, 18, 128, 32)
    nz.copy_(pages.view(6, 1, 128, 18, 32).transpose(2, 3).contiguous().to(device))
    latent, rope = split_flashmla_cache(nz, 512)
    slots = torch.tensor([4 * 128 + 127, -1, 0, 5 * 128 + 1], dtype=slot_dtype)
    replacement = torch.randn(4, 1, 576, dtype=torch.bfloat16).to(dtype)
    new_rows = replacement.to(device)

    def write():
        torch.ops._C_ascend.quant_flash_mla_nz_write(new_rows[..., :512], new_rows[..., 512:], latent, rope, slots_npu)

    slots_npu = slots.to(device)
    write()
    golden_rows = pages.view(torch.uint8).view(-1, 576)
    golden_rows[slots[slots >= 0].long()] = replacement.view(torch.uint8)[slots >= 0, 0]
    expected = pages.view(6, 1, 128, 18, 32).transpose(2, 3)
    assert torch.equal(raw[:, 0].cpu(), torch.full_like(raw[:, 0].cpu(), 91))
    torch.testing.assert_close(nz.view(torch.uint8).cpu(), expected.view(torch.uint8), rtol=0, atol=0)

    table = torch.tensor([[4, 0, 2], [5, 3, 1], [0, 0, 0]], dtype=torch.int32, device=device)
    prefix = load_mla_cache_context_rows(
        latent.view(torch.int8),
        rope.view(torch.int8),
        table,
        torch.tensor([3, 1, 0], dtype=torch.int32, device=device),
        torch.tensor([127, 1, 0], dtype=torch.int32, device=device),
        4,
    )
    torch.testing.assert_close(prefix.view(torch.uint8).cpu(), golden_rows[[4 * 128 + 127, 0, 1, 5 * 128 + 1]])

    config = QuantFlashMLAConfig(96, 0.05, mask_mode=3 if causal else 0)
    op = QuantFlashMLAImpl.load(config)
    q = torch.randn(12, 96, 576, dtype=torch.bfloat16).to(device).to(dtype)
    qs = torch.rand((12, 96), device=device) + 0.5
    ks = torch.tensor([0.75], device=device)
    lengths = torch.tensor([257, 133, 0], dtype=torch.int32, device=device)
    cu = torch.tensor([0, 7, 10, 12], dtype=torch.int32, device=device)
    used = torch.tensor([7, 3, 0], dtype=torch.int32, device=device)
    mask = torch.ones(2048, 2048, dtype=torch.int8).triu(1).to(device) if causal else None
    metadata = op.build_metadata(lengths, cu, used)
    kwargs = dict(
        q_descale=qs,
        k_descale=ks,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        seqused_q=used,
        block_table=table,
        metadata=metadata,
        attn_mask=mask,
    )
    bbnd = pages.to(device)
    expected, _ = op.attention(q, bbnd, **kwargs)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        write()
        result, _ = op.attention(q, nz, **kwargs)
    graph.replay()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    # Graph replay must observe refreshed lengths and page IDs at the captured addresses.
    table[0].copy_(torch.tensor([1, 2, 3], device=device, dtype=torch.int32))
    lengths.copy_(torch.tensor([130, 255, 0], device=device, dtype=torch.int32))
    metadata.copy_(op.build_metadata(lengths, cu, used))
    graph.replay()
    expected, _ = op.attention(q, bbnd, **kwargs)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
