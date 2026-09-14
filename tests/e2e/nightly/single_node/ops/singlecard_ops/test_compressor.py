# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.models.deepseek_v4.compressor import Compressor

HIDDEN_SIZE = 4096
BLOCK_SIZE = 128
ROPE_HEAD_DIM = 64
NORM_EPS = 1e-6
PROFILES = [(4, 2, 128), (4, 2, 512), (128, 1, 512)]


def _inputs(lengths, ratio, coff, head_dim, dtype):
    """Allocate a reserved zero block and disjoint physical pages per request."""
    torch.manual_seed(23)
    max_blocks = (max(lengths) + BLOCK_SIZE - 1) // BLOCK_SIZE
    table = torch.arange(1, len(lengths) * max_blocks + 1, dtype=torch.int32).view(len(lengths), max_blocks)
    width = coff * head_dim
    return {
        "x": torch.randn(sum(lengths), HIDDEN_SIZE, dtype=dtype) * 0.1,
        "wkv": torch.randn(width, HIDDEN_SIZE, dtype=dtype) * 0.02,
        "wgate": torch.randn(width, HIDDEN_SIZE, dtype=dtype) * 0.01,
        "state_cache": torch.zeros(1 + table.numel(), BLOCK_SIZE, 2 * width),
        "ape": torch.randn(ratio, width) * 0.1,
        "state_block_table": table,
        "cu_seqlens": torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32),
        "start_pos": torch.zeros(len(lengths), dtype=torch.int32),
    }


def _reference(inputs, lengths, ratio, coff, head_dim):
    """Dense, unpaged reference; mask missing history before the first C4 block."""
    kv = inputs["x"].float() @ inputs["wkv"].float().T
    gates = inputs["x"].float() @ inputs["wgate"].float().T
    state = inputs["state_cache"].clone()
    outputs = []
    offset = 0
    for batch, length in enumerate(lengths):
        batch_kv = kv[offset : offset + length]
        batch_score = gates[offset : offset + length] + inputs["ape"][torch.arange(length) % ratio]
        for token in range(length):
            block = inputs["state_block_table"][batch, token // BLOCK_SIZE]
            state[block, token % BLOCK_SIZE] = torch.cat((batch_kv[token], batch_score[token]))
        for end in range(ratio, length + 1, ratio):
            # C4 uses the previous block's first projection and the current
            # block's second projection. C128 has just the current projection.
            values, scores = [], []
            for part in range(coff):
                begin = end - (coff - part) * ratio
                if begin < 0:
                    continue
                columns = slice(part * head_dim, (part + 1) * head_dim)
                values.append(batch_kv[begin : begin + ratio, columns])
                scores.append(batch_score[begin : begin + ratio, columns])
            outputs.append((torch.cat(values) * torch.cat(scores).softmax(dim=0)).sum(dim=0))
        offset += length
    return torch.stack(outputs).to(inputs["x"].dtype), state


def _run(inputs, ratio, coff):
    return torch.ops._C_ascend.compressor(**inputs, seqused=None, cmp_ratio=ratio, coff=coff, cache_mode=1)


def _postprocessor(head_dim):
    compressor = Compressor.__new__(Compressor)
    torch.nn.Module.__init__(compressor)
    compressor.rope_head_dim = ROPE_HEAD_DIM
    compressor.norm_eps = NORM_EPS
    compressor.norm = SimpleNamespace(weight=torch.ones(head_dim, device="npu"))
    return compressor


def _assert_close(actual, expected, dtype):
    tolerance = 2e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(("ratio", "coff", "head_dim"), PROFILES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compressor_prefill_and_postprocess(ratio, coff, head_dim, dtype):
    lengths = [2 * ratio + 1, ratio + 2]
    inputs = _inputs(lengths, ratio, coff, head_dim, dtype)
    expected, expected_state = _reference(inputs, lengths, ratio, coff, head_dim)
    npu_inputs = {name: tensor.npu() for name, tensor in inputs.items()}
    actual = _run(npu_inputs, ratio, coff)
    capacity = min(sum(lengths), sum(lengths) // ratio + len(lengths))
    assert actual.shape == (capacity, head_dim)
    _assert_close(actual[: len(expected)], expected, dtype)
    _assert_close(npu_inputs["state_cache"], expected_state, dtype)

    # Compare the complete migrated chain to an independent complex rotation.
    angles = torch.randn(capacity, ROPE_HEAD_DIM // 2)
    cos = angles.cos().repeat_interleave(2, dim=-1).unsqueeze(1).npu()
    sin = angles.sin().repeat_interleave(2, dim=-1).unsqueeze(1).npu()
    normalized = torch.nn.functional.rms_norm(expected.float(), (head_dim,), eps=NORM_EPS)
    tail = normalized[:, -ROPE_HEAD_DIM:].reshape(len(expected), -1, 2)
    rotated = torch.view_as_complex(tail.contiguous()) * torch.polar(
        torch.ones_like(angles[: len(expected)]), angles[: len(expected)]
    )
    normalized[:, -ROPE_HEAD_DIM:] = torch.view_as_real(rotated).flatten(1)
    processed = _postprocessor(head_dim)._postprocess(actual, cos, sin)
    _assert_close(processed[: len(expected)], normalized.to(dtype), dtype)


@pytest.mark.parametrize(("ratio", "coff", "head_dim"), PROFILES)
def test_compressor_chunked_prefill_decode_updates_state(ratio, coff, head_dim):
    dtype = torch.bfloat16
    lengths = [2 * ratio + 1]
    inputs = _inputs(lengths, ratio, coff, head_dim, dtype)
    expected, expected_state = _reference(inputs, lengths, ratio, coff, head_dim)
    npu_inputs = {name: tensor.npu() for name, tensor in inputs.items()}
    outputs = []
    start = 0
    # First call updates state without completing a compressed token. The
    # single-token decode completes it; the next chunk reads overlap history.
    for count in (ratio - 1, 1, ratio + 1):
        step = dict(npu_inputs)
        step["x"] = npu_inputs["x"][start : start + count]
        step["cu_seqlens"] = torch.tensor([0, count], dtype=torch.int32, device="npu")
        step["start_pos"] = torch.tensor([start], dtype=torch.int32, device="npu")
        raw = _run(step, ratio, coff)
        valid_rows = (start + count) // ratio - start // ratio
        outputs.append(raw[:valid_rows])
        start += count
    _assert_close(torch.cat(outputs), expected, dtype)
    _assert_close(npu_inputs["state_cache"], expected_state, dtype)


@pytest.mark.parametrize(("ratio", "coff", "head_dim"), PROFILES)
def test_compressor_padded_state_cache(ratio, coff, head_dim):
    """Acceptance check for the model's padded page stride and in-place writes."""
    lengths = [ratio + 1]
    inputs = _inputs(lengths, ratio, coff, head_dim, torch.bfloat16)
    expected, expected_state = _reference(inputs, lengths, ratio, coff, head_dim)
    npu_inputs = {name: tensor.npu() for name, tensor in inputs.items()}
    blocks, _, width = inputs["state_cache"].shape
    padding_value = -17.0
    backing = torch.full((blocks, BLOCK_SIZE + 1, width), padding_value, device="npu")
    state_view = backing[:, :BLOCK_SIZE]
    state_view.copy_(npu_inputs["state_cache"])
    npu_inputs["state_cache"] = state_view
    assert not state_view.is_contiguous()
    actual = _run(npu_inputs, ratio, coff)
    _assert_close(actual[: len(expected)], expected, torch.bfloat16)
    _assert_close(state_view, expected_state, torch.bfloat16)
    torch.testing.assert_close(backing[:, BLOCK_SIZE], torch.full_like(backing[:, BLOCK_SIZE], padding_value))


@pytest.mark.parametrize(("ratio", "coff", "head_dim"), PROFILES)
def test_compressor_empty_and_meta(ratio, coff, head_dim):
    inputs = _inputs([ratio], ratio, coff, head_dim, torch.bfloat16)
    npu_inputs = {name: tensor.npu() for name, tensor in inputs.items()}
    npu_inputs["x"] = npu_inputs["x"][:0]
    npu_inputs["cu_seqlens"].zero_()
    before = npu_inputs["state_cache"].clone()
    actual = _run(npu_inputs, ratio, coff)
    assert actual.shape == (0, head_dim)
    torch.testing.assert_close(npu_inputs["state_cache"], before)
    meta_inputs = {name: tensor.to("meta") for name, tensor in npu_inputs.items()}
    assert _run(meta_inputs, ratio, coff).shape == actual.shape


@pytest.mark.parametrize(("ratio", "coff", "head_dim"), PROFILES)
def test_compressor_graph_replay(ratio, coff, head_dim):
    inputs = _inputs([2 * ratio], ratio, coff, head_dim, torch.bfloat16)
    expected, expected_state = _reference(inputs, [2 * ratio], ratio, coff, head_dim)
    npu_inputs = {name: tensor.npu() for name, tensor in inputs.items()}
    eager = _run(npu_inputs, ratio, coff)
    cos = torch.ones(eager.shape[0], 1, ROPE_HEAD_DIM, device="npu")
    sin = torch.zeros_like(cos)
    compressor = _postprocessor(head_dim)
    expected_processed = compressor._postprocess(eager, cos, sin)[: len(expected)].clone()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = _run(npu_inputs, ratio, coff)
        processed = compressor._postprocess(actual, cos, sin)
    for _ in range(2):
        npu_inputs["state_cache"].zero_()
        graph.replay()
        _assert_close(actual[: len(expected)], expected, torch.bfloat16)
        _assert_close(processed[: len(expected)], expected_processed, torch.bfloat16)
        _assert_close(npu_inputs["state_cache"], expected_state, torch.bfloat16)


@pytest.mark.parametrize("cmp_ratio,coff", [(0, 1), (4, 0), (4, 3)])
def test_compressor_rejects_invalid_attributes(cmp_ratio, coff):
    inputs = _inputs([4], 4, 2, 128, torch.bfloat16)
    npu_inputs = {name: tensor.npu() for name, tensor in inputs.items()}
    with pytest.raises(RuntimeError, match="compressor (cmp_ratio|coff)"):
        _run(npu_inputs, cmp_ratio, coff)


@pytest.mark.parametrize(("ratio", "coff", "head_dim"), PROFILES)
def test_compressor_chain_latency(ratio, coff, head_dim, record_property):
    """Record full-chain latency and peak allocation for the target CANN build."""
    inputs = _inputs([2 * ratio], ratio, coff, head_dim, torch.bfloat16)
    npu_inputs = {name: tensor.npu() for name, tensor in inputs.items()}
    capacity = min(2 * ratio, 3)
    cos = torch.ones(capacity, 1, ROPE_HEAD_DIM, device="npu")
    sin = torch.zeros_like(cos)
    compressor = _postprocessor(head_dim)

    def run_chain():
        return compressor._postprocess(_run(npu_inputs, ratio, coff), cos, sin)

    for _ in range(3):
        run_chain()
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    iterations = 20
    start.record()
    for _ in range(iterations):
        run_chain()
    end.record()
    torch.npu.synchronize()
    record_property("compressor_chain_ms", start.elapsed_time(end) / iterations)
    record_property("peak_allocated_bytes", torch.npu.max_memory_allocated())
