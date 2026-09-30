# SPDX-License-Identifier: Apache-2.0
"""Exercise compact-prefix lengths, raw softmax outputs and graph replay on A3."""

import json

import pytest
import torch
import torch_npu

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.SFA_C8_DCP_REPLICATED_INDEXER),
    reason="requires the custom SFA operators",
)

COUNTS = (0, 1, 127, 128, 129, 511, 512, 513, 2048)
HEAD_DIM = 512
ROPE_DIM = 64
TOPK = 2048


def make_inputs(dtype, layout, quantized=False, heads=64):
    torch.manual_seed(20260908)
    block_size, blocks_per_request = 128, 32
    query_lens = [3, 2, 4] if layout == "TND" else [3, 3, 3]
    rows = sum(query_lens)
    shape = (rows, heads, HEAD_DIM) if layout == "TND" else (3, 3, heads, HEAD_DIM)
    query = torch.randn(shape, dtype=dtype).npu()
    query_rope = torch.randn((*shape[:-1], ROPE_DIM), dtype=dtype).npu()
    key = torch.randn((3 * blocks_per_request, block_size, 1, HEAD_DIM), dtype=dtype).npu()
    key_rope = torch.randn((3 * blocks_per_request, block_size, 1, ROPE_DIM), dtype=dtype).npu()
    block_table = torch.randperm(3 * blocks_per_request, dtype=torch.int32).reshape(3, blocks_per_request)
    kv_lens = [4096, 3072, 2048]
    indices = torch.full((rows, 1, TOPK), -1, dtype=torch.int32)
    row = 0
    for batch, query_len in enumerate(query_lens):
        for _ in range(query_len):
            count = COUNTS[row]
            # Local IDs are deliberately not numerically sorted.
            indices[row, 0, :count] = torch.randperm(kv_lens[batch], dtype=torch.int32)[:count]
            row += 1
    if layout == "BSND":
        indices = indices.reshape(3, 3, 1, TOPK)
    actual_query_lens = torch.tensor(query_lens, dtype=torch.int32)
    if layout == "TND":
        actual_query_lens = actual_query_lens.cumsum(0).int()
    inputs = dict(
        query=query,
        key=key,
        value=key,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_indices=indices.npu(),
        block_table=block_table.npu(),
        actual_seq_lengths_query=actual_query_lens.npu(),
        actual_seq_lengths_kv=torch.tensor(kv_lens, dtype=torch.int32).npu(),
        scale_value=(HEAD_DIM + ROPE_DIM) ** -0.5,
        sparse_block_size=1,
        sparse_mode=0,
        attention_mode=2,
        layout_query=layout,
        layout_kv="PA_BSND",
        return_softmax_lse=True,
    )

    if quantized:
        # Packed C8 cache: int8 NoPE, BF16 RoPE bytes, FP32 scale bytes.
        tile_size = 128
        key_q = (key.float() * 32).round().clamp(-128, 127).to(torch.int8)
        scales = torch.full((*key.shape[:-1], HEAD_DIM // tile_size), 1 / 32, dtype=torch.float32, device=key.device)
        packed = torch.cat((key_q, key_rope.contiguous().view(torch.int8), scales.view(torch.int8)), dim=-1)
        inputs.update(
            query=torch.cat((query, query_rope), dim=-1),
            key=packed,
            value=packed,
            key_quant_mode=2,
            value_quant_mode=2,
            quant_scale_repo_mode=1,
            tile_size=tile_size,
            rope_head_dim=ROPE_DIM,
        )
        del inputs["query_rope"], inputs["key_rope"]
    return inputs


def run_op(inputs):
    op = (
        torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention
        if inputs["key"].dtype == torch.int8
        else torch.ops._C_ascend.npu_sparse_flash_attention
    )
    return op(**inputs)


def reference_attention(inputs):
    """Independent CPU attention over selected tokens, including C8 dequantization."""
    cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    dtype = cpu["query"].dtype
    if cpu["key"].dtype == torch.int8:
        packed = cpu["key"]
        rope_end = HEAD_DIM + ROPE_DIM * 2
        scales = packed[..., rope_end:].contiguous().view(torch.float32).repeat_interleave(128, dim=-1)
        value = (packed[..., :HEAD_DIM].float() * scales).to(dtype).float()
        rope = packed[..., HEAD_DIM:rope_end].contiguous().view(dtype).float()
        key = torch.cat((value, rope), dim=-1)
        query = cpu["query"].float()
    else:
        value = cpu["value"].float()
        key = torch.cat((cpu["key"], cpu["key_rope"]), dim=-1).float()
        query = torch.cat((cpu["query"], cpu["query_rope"]), dim=-1).float()
    query = query.reshape(-1, query.shape[-2], HEAD_DIM + ROPE_DIM)
    indices = cpu["sparse_indices"].reshape(query.shape[0], TOPK)
    qlens = cpu["actual_seq_lengths_query"]
    if cpu["layout_query"] == "TND":
        qlens = torch.diff(qlens, prepend=torch.zeros(1, dtype=qlens.dtype))
    output = torch.zeros((*query.shape[:-1], HEAD_DIM))
    lse = torch.full(query.shape[:-1], -torch.inf)
    row = 0
    for batch, qlen in enumerate(qlens.tolist()):
        kvlen = int(cpu["actual_seq_lengths_kv"][batch])
        for pos in range(qlen):
            selected = indices[row]
            selected = selected[selected >= 0].long()
            tokens = (selected[:, None] * cpu["sparse_block_size"] + torch.arange(cpu["sparse_block_size"])).flatten()
            limit = kvlen if cpu["sparse_mode"] == 0 else kvlen - qlen + pos + 1
            tokens = tokens[tokens < limit]
            if tokens.numel():
                block_size = key.shape[1]
                blocks = cpu["block_table"][batch, tokens // block_size].long()
                offsets = tokens % block_size
                k, v = key[blocks, offsets, 0], value[blocks, offsets, 0]
                scores = query[row] @ k.T * cpu["scale_value"]
                lse[row] = torch.logsumexp(scores, dim=-1)
                output[row] = torch.softmax(scores, dim=-1).to(dtype).float() @ v
            row += 1
    return output, lse


def assert_reference(actual, inputs):
    expected, expected_lse = reference_attention(inputs)
    output, maximum, total = [x.float().cpu() for x in actual]
    output = output.reshape_as(expected)
    lse = (maximum + total.log()).reshape_as(expected_lse)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, expected_lse, atol=1e-3, rtol=1e-3)
    empty = torch.isneginf(expected_lse)
    # Empty rows must keep the merge identity, including finite raw max and zero sum.
    assert torch.isfinite(maximum.reshape_as(expected_lse)[empty]).all()
    assert (total.reshape_as(expected_lse)[empty] == 0).all()


def assert_outputs(actual, expected):
    errors = {}
    for name, got, ref in zip(("output", "softmax_max", "softmax_sum"), actual, expected):
        got, ref = got.float().cpu(), ref.float().cpu()
        errors[name] = float((got - ref).abs().max())
        tolerance = 1e-2 if name == "output" else 1e-5
        torch.testing.assert_close(got, ref, atol=tolerance, rtol=tolerance)
    actual_lse = (actual[1].float() + actual[2].float().log()).cpu()
    expected_lse = (expected[1].float() + expected[2].float().log()).cpu()
    torch.testing.assert_close(actual_lse, expected_lse, atol=1e-5, rtol=1e-5)
    finite = torch.isfinite(expected_lse)
    errors["lse_finite"] = float((actual_lse[finite] - expected_lse[finite]).abs().max()) if finite.any() else 0.0
    print("VALID_COUNT_MAX_ABS_DIFF " + json.dumps(errors))


@pytest.mark.parametrize("dtype,quantized", [(torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True)])
@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize("heads", [16, 64, 128])
@torch.inference_mode()
def test_sparse_valid_count_lengths_and_softmax(dtype, quantized, layout, heads):
    inputs = make_inputs(dtype, layout, quantized, heads)
    assert_reference(run_op(inputs), inputs)


@pytest.mark.parametrize("dtype,quantized", [(torch.bfloat16, False), (torch.float16, False), (torch.bfloat16, True)])
@torch.inference_mode()
def test_sparse_valid_count_graph_replays_changed_indices(dtype, quantized):
    inputs = make_inputs(dtype, "TND", quantized)
    run_op(inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run_op(inputs)
    original = inputs["sparse_indices"].clone()
    for count in (TOPK, 0, 1, 128, 513, TOPK):
        changed = original.clone()
        changed[..., count:] = -1
        inputs["sparse_indices"].copy_(changed)
        expected = run_op(inputs)
        assert_reference(expected, inputs)
        graph.replay()
        torch.npu.synchronize()
        assert_outputs(captured, expected)


@pytest.mark.parametrize("quantized", [False, True])
@torch.inference_mode()
def test_sparse_valid_count_zero_kv_rows(quantized):
    inputs = make_inputs(torch.bfloat16, "TND", quantized)
    inputs["actual_seq_lengths_kv"][-1] = 0
    inputs["sparse_indices"][5:] = -1
    assert_reference(run_op(inputs), inputs)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("sparse_mode,block_size", [(3, 1), (0, 2)])
@torch.inference_mode()
def test_other_modes_keep_existing_semantics(quantized, sparse_mode, block_size):
    inputs = make_inputs(torch.bfloat16, "TND", quantized)
    inputs["sparse_mode"] = sparse_mode
    inputs["sparse_block_size"] = block_size
    if block_size > 1:
        indices = inputs["sparse_indices"]
        # Convert token IDs to block IDs without introducing duplicates.
        indices.div_(block_size, rounding_mode="floor")
        cpu = indices.cpu()
        for row in cpu:
            valid = torch.unique(row[row >= 0])
            row.fill_(-1)
            row[0, : valid.numel()] = valid.flip(0)
        indices.copy_(cpu)
    assert_reference(run_op(inputs), inputs)
