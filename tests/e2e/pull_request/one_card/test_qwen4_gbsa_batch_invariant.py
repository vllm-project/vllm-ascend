# SPDX-License-Identifier: Apache-2.0
"""Real A5 GBSA batch/page-relocation regression; no model weights required."""

import pytest
import torch

from vllm_ascend.device.device_config import check_ascend_device_type, is_950
from vllm_ascend.models.qwen4_exp import gbsa
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.utils import enable_custom_op

PAGE_SIZE = 128
BLOCK_SIZE = 4
HEAD_DIM = 256
QUERY_HEADS = 12
BATCH_SIZE = 4


@pytest.fixture(scope="module", autouse=True)
def require_a5():
    if not torch.npu.is_available():
        pytest.skip("GBSA regression requires Ascend 950")
    check_ascend_device_type()
    if not is_950():
        pytest.skip("GBSA regression requires Ascend 950")
    enable_custom_op()
    init_device_properties_triton()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length,selected_blocks", [(305, 76), (20481, 511)])
def test_gbsa_batch_and_physical_page_invariance(monkeypatch, dtype, length, selected_blocks):
    monkeypatch.setattr(gbsa.envs, "VLLM_BATCH_INVARIANT", True)
    generator = torch.Generator().manual_seed(38198)
    pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
    backing = torch.randn(BATCH_SIZE * pages, 2, PAGE_SIZE, 1, HEAD_DIM, generator=generator).to(dtype)
    table = torch.randperm(BATCH_SIZE * pages, generator=generator).reshape(BATCH_SIZE, pages).int()
    query = torch.randn(BATCH_SIZE, QUERY_HEADS, HEAD_DIM, generator=generator).to(dtype)
    # LI-v2 indices contain compressed full blocks followed by three residual slots.
    compact = torch.full((BATCH_SIZE, selected_blocks + 3), -1, dtype=torch.int32)
    selected_tokens = []
    for row in range(BATCH_SIZE):
        blocks = torch.randperm(length // BLOCK_SIZE, generator=generator)[: selected_blocks - row]
        compact[row, : len(blocks)] = blocks.int()
        tokens = (blocks[:, None] * BLOCK_SIZE + torch.arange(BLOCK_SIZE)).flatten()
        selected_tokens.append(torch.cat((tokens, torch.arange(length // BLOCK_SIZE * BLOCK_SIZE, length))))

    # Independent dense attention over exactly the selected logical tokens.
    golden = []
    for row, tokens in enumerate(selected_tokens):
        logical_kv = backing[table[row, tokens // PAGE_SIZE].long(), :, tokens % PAGE_SIZE, 0].float()
        scores = query[row].float() @ logical_kv[:, 0].T / HEAD_DIM**0.5
        golden.append(torch.softmax(scores, dim=-1) @ logical_kv[:, 1])
    golden = torch.stack(golden)

    device_query = query.npu()
    device_compact = compact.npu()

    def run(pool, page_table, order):
        q = device_query[order]
        lens = torch.full((len(order),), length, dtype=torch.int32, device="npu")
        output = torch.empty_like(q)
        gbsa.qsa_gbsa_attention(
            q,
            pool[:, 0],
            pool[:, 1],
            device_compact[order],
            page_table[order],
            torch.arange(len(order), dtype=torch.int32, device="npu"),
            lens - 1,
            lens,
            output,
        )
        return output.cpu()

    pool = backing.npu()
    batch = run(pool, table.npu(), list(range(BATCH_SIZE)))
    assert torch.isfinite(batch).all()
    # This tolerance checks attention semantics; BI comparisons below are bitwise.
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(batch.float(), golden, rtol=tolerance, atol=tolerance)

    # Keep logical K/V unchanged while reversing every physical page address.
    page_order = torch.arange(backing.shape[0] - 1, -1, -1)
    relocated = torch.empty_like(backing)
    relocated[page_order] = backing
    relocated_pool = relocated.npu()
    relocated_table = page_order[table.long()].int().npu()
    order = [3, 0, 2, 1]
    mixed = run(relocated_pool, relocated_table, order)
    assert torch.equal(mixed.view(torch.int16), batch[order].view(torch.int16))
    for row in range(BATCH_SIZE):
        solo = run(relocated_pool, relocated_table, [row])
        assert torch.equal(solo.view(torch.int16), batch[row : row + 1].view(torch.int16))
