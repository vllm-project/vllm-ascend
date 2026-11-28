# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.qwen4_exp.attention_dispatch import (
    build_qsa_request_plan,
    dispatch_qsa_attention,
    prepare_qsa_union_batches,
)
from vllm_ascend.models.qwen4_exp.qsa import AscendQSAImpl
from vllm_ascend.ops.triton.qwen4_exp.union_cache import prepare_union_pages

pytestmark = [
    pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton"),
    pytest.mark.skipif(
        get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5 MXFP8 and Union GBSA"
    ),
]


def test_padded_slots_cannot_overwrite_valid_rows():
    # The highest physical page is selected, page zero is absent from the layout.
    slots = torch.tensor([-1, 255, 256, 7, 128], device="npu")
    table = torch.tensor([[1]], dtype=torch.int32, device="npu")
    lengths = torch.tensor([128], dtype=torch.int32, device="npu")
    # Dense preparation uses the request table and lengths.
    indices = torch.empty((1, 4), dtype=torch.int32, device="npu")
    pages, current_rows, _ = prepare_union_pages(2, slots, indices, lengths - 1, lengths, lengths * 0, table, 1)
    assert pages.tolist() == [1]
    expected = torch.full((128,), -1, dtype=torch.int32)
    expected[0], expected[-1] = 4, 1
    torch.testing.assert_close(current_rows[:-1].cpu(), expected)


def _layer_cache(num_pages, dim, slots, generator):
    shape = (num_pages, 128, 1, dim)
    key = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    value = torch.randn(shape, dtype=torch.bfloat16, generator=generator)
    packed_k, k_scales = torch_npu.npu_dynamic_mx_quant(key.npu(), dst_type=torch.float8_e4m3fn)
    k_scales = k_scales.view(torch.uint8).reshape(*shape[:-1], dim // 32)
    channel_bytes = torch.randint(124, 130, (dim,), dtype=torch.uint8, generator=generator)
    channel_scale = torch.exp2(channel_bytes.float() - 127)
    packed_v = (value.float() / channel_scale).to(torch.float8_e4m3fn)
    v_scales = channel_bytes.repeat_interleave(2).expand(num_pages, 2, 1, dim * 2).contiguous()
    cache = packed_k, packed_v.npu(), k_scales, v_scales.npu()

    # Decode the full physical pool independently. Preserve original BF16 values for
    # rows written by this chunk, including the partial page shared with history.
    reference_k = (
        packed_k.cpu().float() * torch.exp2(k_scales.cpu().float() - 127).repeat_interleave(32, -1)
    ).bfloat16()
    reference_v = (packed_v.float() * channel_scale).bfloat16()
    current_k, current_v = key.flatten(0, 1)[slots], value.flatten(0, 1)[slots]
    reference_k.flatten(0, 1)[slots] = current_k
    reference_v.flatten(0, 1)[slots] = current_v
    return cache, (current_k.npu(), current_v.npu()), (reference_k.npu(), reference_v.npu())


@pytest.mark.parametrize("head_dim", [192, 256])
@torch.inference_mode()
def test_two_layers_share_dense_layout_with_independent_kv(head_dim):
    import vllm_ascend.vllm_ascend_C  # noqa: F401 - register Union operator

    generator = torch.Generator().manual_seed(453)
    counts = [515, 257]
    history = [389, 131]
    rows = sum(counts)
    lengths = torch.tensor([h + c for h, c in zip(history, counts)], dtype=torch.int32)
    positions = torch.cat([torch.arange(h, h + c, dtype=torch.int32) for h, c in zip(history, counts)])
    requests = torch.repeat_interleave(torch.arange(2, dtype=torch.int32), torch.tensor(counts))
    columns = (max(lengths.tolist()) + 127) // 128
    num_pages = 2 * columns
    table = torch.randperm(num_pages, generator=generator).int().reshape(2, columns)
    slots = (128 * table[requests.long(), positions.long() // 128] + positions % 128).long()

    table, requests, positions, lengths = (t.npu() for t in (table, requests, positions, lengths))
    starts = torch.tensor([0, counts[0], rows], dtype=torch.int32)
    plan = prepare_qsa_union_batches(
        build_qsa_request_plan(starts, rows, torch.ones(2, dtype=torch.bool)), positions + 1, lengths, table
    )
    metadata = SimpleNamespace(
        num_actual_tokens=rows,
        block_table=table,
        slot_mapping=slots.npu(),
        qsa_request_plan=plan,
        qsa_dense_page_layouts={},
    )
    impl = AscendQSAImpl(12, head_dim, head_dim**-0.5, 1, "mxfp8")
    query = torch.randn(rows, 12, head_dim, dtype=torch.bfloat16, generator=generator).npu()
    first_layout = None
    for layer_idx in range(2):
        cache, current_kv, (reference_k, reference_v) = _layer_cache(num_pages, head_dim, slots, generator)
        # Layer-specific selections include both history and current tokens.
        indices = torch.full((rows, 515), -1, dtype=torch.int32)
        for row, position in enumerate(positions.tolist()):
            groups = torch.randperm((position + 1) // 4, generator=generator)[:140]
            indices[row, : len(groups)] = groups
        indices = indices.npu()
        layer = SimpleNamespace(indexer=SimpleNamespace(token_topk=2048, compress_ratio=4), topk_indices_buffer=indices)
        expected = dispatch_qsa_attention(
            query, reference_k, reference_v, indices, table, requests, positions, lengths, torch.empty_like(query), plan
        )
        actual = impl.forward(
            layer,
            query,
            cache,
            metadata,
            torch.empty_like(query),
            token_to_req=requests,
            logical_positions=positions,
            seq_lens=lengths,
            current_kv=current_kv,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        (layout,) = metadata.qsa_dense_page_layouts.values()
        if layer_idx == 0:
            first_layout = layout
        else:
            assert layout is first_layout
