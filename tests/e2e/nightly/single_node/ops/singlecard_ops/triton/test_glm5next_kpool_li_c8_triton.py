# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Page-local C8 keys/scales and raw-tail cache writer regression tests."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

HEAD_DIM = 128
ROW_WIDTH = HEAD_DIM + 4
POOL_SIZE = 4
SENTINEL = 0x55


def _quantize_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for the writer's per-row INT8 keys and FP16 scales."""
    abs_max = rows.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(abs_max > 0, abs_max / 127.0, torch.zeros_like(abs_max))
    normalized = rows / torch.where(abs_max > 0, scale, torch.ones_like(scale))
    quantized = torch.round(normalized).clamp(-127.0, 127.0).to(torch.int8)
    return quantized, scale.squeeze(-1).half()


@pytest.mark.parametrize("quant", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("tail_padding", [0, 32])
@torch.inference_mode()
def test_cann_pooled_cache_writer_preserves_tail_and_padded_pages(quant, speculative, tail_padding):
    from vllm_ascend.ops.cann_keypool import prepare_keypool_state
    from vllm_ascend.ops.triton.glm5_next_cann_keypool import write_keypool_caches

    generator = torch.Generator().manual_seed(41)
    dim, capacity, page_rows = HEAD_DIM, 7, 16
    width = ROW_WIDTH if quant else HEAD_DIM
    dtype = torch.int8 if quant else torch.bfloat16
    page_elements = page_rows * 2 * dim + tail_padding
    tail_storage = torch.full((3 * page_elements,), 55.0, device="npu")
    tail = tail_storage.as_strided((3, capacity, 2 * dim), (page_elements, 2 * dim, 1))
    cache_storage = torch.full((3, 48, 1, width), SENTINEL, dtype=dtype, device="npu")
    cache = cache_storage[:, :32]
    pooled = torch.randn(4, dim, generator=generator).bfloat16()
    pooled[0] = 0
    keys = torch.randn(3, dim, generator=generator)
    gates = torch.randn(3, dim, generator=generator)
    state = torch.randn(4, 4, 2 * dim, generator=generator)
    writes = torch.tensor([[1, page_rows + 2, -1]], dtype=torch.int64, device="npu")
    state_rows = torch.tensor([[1, 6, 0]], dtype=torch.int64, device="npu")
    metadata = SimpleNamespace(pooled_slot_mapping=torch.tensor([0, 33, -1, 96], device="npu"))
    tail_metadata = SimpleNamespace(
        cann_inputs=SimpleNamespace(
            write_indices=writes,
            state_indices=state_rows,
            page_rows=page_rows,
        )
    )
    write_keypool_caches(
        tail,
        cache,
        keys.npu() if speculative else None,
        gates.npu() if speculative else None,
        pooled.npu(),
        metadata,
        tail_metadata,
        state.npu(),
    )
    expected_tail = torch.full_like(tail_storage.cpu(), 55.0)
    source = torch.cat((keys, gates), dim=-1) if speculative else state.view(-1, 2 * dim)[[1, 6]]
    expected_tail_view = expected_tail.as_strided((3, capacity, 2 * dim), (page_elements, 2 * dim, 1))
    expected_tail_view[0, 1] = source[0]
    expected_tail_view[1, 2] = source[1]
    torch.testing.assert_close(tail_storage.cpu(), expected_tail, rtol=0, atol=0)
    expected_cache = torch.full_like(cache_storage.cpu(), SENTINEL)
    actual_cache = cache_storage.cpu()
    if quant:
        from vllm_ascend.ops.triton.glm5_next_quant_indexer import view_quant_kpool_cache

        expected_keys, expected_scales = view_quant_kpool_cache(expected_cache[:, :32])
        actual_keys, actual_scales = view_quant_kpool_cache(actual_cache[:, :32])
        quantized, scales = _quantize_rows(pooled.float())
        for block, row, source_row in ((0, 0, 0), (1, 1, 1)):
            expected_keys[block, row, 0] = quantized[source_row]
            expected_scales[block, row, 0] = scales[source_row]
            actual = actual_keys[block, row, 0]
            expected = expected_keys[block, row, 0]
            assert (actual.to(torch.int16) - expected.to(torch.int16)).abs().max() <= 1
            scale = actual_scales[block, row, 0].float()
            error = (actual.float() * scale - pooled[source_row].float()).abs()
            assert torch.all(error <= scale * 0.57 + 1e-7)
            expected.copy_(actual)
    else:
        expected_cache[0, 0, 0] = pooled[0]
        expected_cache[1, 1, 0] = pooled[1]
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)

    history = torch.tensor([[0, 1, 2, 3], [page_rows + 2, page_rows + 3, page_rows, page_rows + 1]], device="npu")
    tail_metadata.cann_inputs.history_indices = history
    gathered = prepare_keypool_state(tail, torch.zeros(2, device="npu", dtype=torch.int32), tail_metadata, POOL_SIZE)
    expected_history = torch.stack((expected_tail_view[0, [0, 1, 2, 3]], expected_tail_view[1, [2, 3, 0, 1]]))
    torch.testing.assert_close(gathered[1:3].cpu(), expected_history, rtol=0, atol=0)
