# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import statistics

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

enable_custom_op()

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="Requires Ascend A5")

HEADS = 32
WIDTH = 128
PAGE_SIZE = 128
GLM_TOPK = 2048
# Empty requests exceed the AIC budget and force the original metadata schedule.
REFERENCE_REQUESTS = 128
CAPTURE_CALLS = 4


def _make_a5_case(
    query_counts, key_lengths, *, quant_mode=1, topk=GLM_TOPK, layout="TND", pattern="random", ratio=1, used_counts=None
):
    generator = torch.Generator().manual_seed(137)
    batch, tokens = len(query_counts), sum(query_counts)
    pages = max(1, (max(key_lengths) + PAGE_SIZE - 1) // PAGE_SIZE)
    dtype = torch.float8_e4m3fn if quant_mode == 1 else torch.int8
    scale_dtype = torch.float32 if quant_mode == 1 else torch.float16
    q = torch.randint(-8, 9, (tokens, HEADS, WIDTH), generator=generator).to(dtype)
    w = (torch.rand(tokens, HEADS, generator=generator) * 2 - 1).to(scale_dtype)
    qs = (torch.rand(tokens, HEADS, generator=generator) * 0.01 + 0.01).to(scale_dtype)
    if pattern == "zero":
        w.zero_()
    key_cpu = torch.randint(-8, 9, (batch * pages, PAGE_SIZE, 1, WIDTH), generator=generator).to(dtype)
    if pattern == "ramp":
        q.zero_()
        q[:, 0, 0] = 1
        w.zero_()
        w[:, 0] = 1
        qs.fill_(1)
        key_cpu.zero_()
        key_cpu[:, :, 0, 0] = 1
    key = torch.empty_strided(key_cpu.shape, ((PAGE_SIZE + 16) * WIDTH, WIDTH, WIDTH, 1), dtype=dtype, device="npu")
    key.copy_(key_cpu)
    scale = torch.empty_strided(key_cpu.shape[:-1], (PAGE_SIZE + 32, 1, 1), dtype=scale_dtype, device="npu")
    scale_cpu = torch.rand(key_cpu.shape[:-1], generator=generator).to(scale_dtype) * 0.01 + 0.01
    scale.copy_(scale_cpu)
    table = torch.randperm(batch * pages, generator=generator).int().reshape(batch, pages)
    prefix = [0]
    for count in query_counts:
        prefix.append(prefix[-1] + count)
    offsets = (torch.arange(tokens, dtype=torch.int32) * 17).reshape(tokens, 1)
    active = [q, w, qs, offsets]

    def arguments(reference):
        values = active
        if layout == "BSND" and not reference:
            values = []
            for value in active:
                padded = torch.zeros((batch, max(query_counts), *value.shape[1:]), dtype=value.dtype)
                for request, count in enumerate(query_counts):
                    padded[request, :count] = value[prefix[request] : prefix[request + 1]]
                values.append(padded)
        extra = REFERENCE_REQUESTS - batch if reference and max(key_lengths) > 0 else 0
        current_table = torch.cat((table, torch.zeros(extra, pages, dtype=torch.int32)))
        lengths = list(key_lengths) + [0] * extra
        common = dict(
            seqused_k=torch.tensor(lengths, dtype=torch.int32, device="npu"),
            max_seqlen_q=max(query_counts),
            layout_q="TND" if reference else layout,
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=ratio,
        )
        if common["layout_q"] == "TND":
            common["cu_seqlens_q"] = torch.tensor(prefix + [tokens] * extra, dtype=torch.int32, device="npu")
            if used_counts is not None:
                common["seqused_q"] = torch.tensor(list(used_counts) + [0] * extra, dtype=torch.int32, device="npu")
        else:
            common["seqused_q"] = torch.tensor(query_counts, dtype=torch.int32, device="npu")
        if ratio > 1:
            common["cmp_residual_k"] = torch.ones(batch + extra, dtype=torch.int32, device="npu")
        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            HEADS,
            1,
            WIDTH,
            topk,
            quant_mode,
            batch_size=batch + extra,
            max_seqlen_k=max(1, max(key_lengths) * ratio + int(ratio > 1)),
            **common,
        )
        return [value.npu() for value in values], current_table.npu(), metadata, common

    tested, reference = arguments(False), arguments(True)

    def run(*, serial=False, return_value=0):
        values, current_table, metadata, common = reference if serial else tested
        q, w, qs, offsets = values
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v2(
            q,
            key,
            w,
            qs,
            scale,
            topk,
            quant_mode,
            block_table=current_table,
            output_idx_offset=offsets,
            metadata=metadata,
            return_value=return_value,
            **common,
        )

    def active_rows(output):
        if layout == "BSND":
            for request, count in enumerate(query_counts):
                assert (output[0][request, count:].cpu() == -1).all()
            output = tuple(
                torch.cat([tensor[request, :count] for request, count in enumerate(query_counts)])
                if tensor.numel()
                else tensor
                for tensor in output
            )
        indices = output[0].cpu()
        for request, count in enumerate(query_counts):
            used = count if used_counts is None else used_counts[request]
            for row in range(count):
                token = prefix[request] + row
                if row >= used:
                    assert (indices[token] == -1).all()
                    continue
                valid_length = max(0, (key_lengths[request] * ratio + int(ratio > 1) - used + row + 1) // ratio)
                valid_length = min(key_lengths[request], valid_length)
                selected = indices[token, 0]
                valid = selected != -1
                assert valid.sum().item() == min(topk, valid_length)
                selected = selected[valid] - offsets[token, 0]
                assert ((selected >= 0) & (selected < valid_length)).all()
                if pattern == "ramp":
                    # Only head 0 contributes: Q.K == 1, weight*Q-scale == 1.
                    # INT8's existing vector path also divides Q.K by 1024.
                    scores = scale_cpu[table[request].long()].flatten()[:valid_length].float()
                    if quant_mode == 2:
                        scores = scores / 1024
                    scores = scores.bfloat16()
                    returned = scores[selected.long()]
                    golden = scores.float().topk(min(topk, valid_length)).values.bfloat16()
                    torch.testing.assert_close(returned.sort().values, golden.sort().values, rtol=0, atol=0)
        return output

    def update(lengths):
        key_lengths[:] = lengths
        for old, fresh in ((tested, arguments(False)), (reference, arguments(True))):
            old[2].copy_(fresh[2])
            for name, tensor in old[3].items():
                if isinstance(tensor, torch.Tensor):
                    tensor.copy_(fresh[3][name])

    return run, active_rows, update


def _check_a5_result(actual, expected):
    actual_indices = actual[0].cpu().flatten(0, -2)
    expected_indices = expected[0].cpu().flatten(0, -2)
    torch.testing.assert_close(actual_indices.sort().values, expected_indices.sort().values, rtol=0, atol=0)
    for ai in actual_indices:
        valid = ai >= 0
        assert ai[valid].unique().numel() == valid.sum().item()


@pytest.mark.parametrize("quant_mode", [1, 2])
@pytest.mark.parametrize("pattern", ["random", "zero"])
@pytest.mark.parametrize(
    "queries,lengths,layout,topk,ratio",
    [
        ([1], [4096], "TND", 2048, 1),
        ([6], [32769], "TND", 2048, 1),
        ([1, 5, 2], [8193, 32769, 4097], "TND", 513, 1),
        ([0, 3, 0, 1], [0, 8193, 0, 4097], "TND", 2048, 1),
        ([1] * 13, [8193] * 13, "TND", 2048, 1),
        ([49], [8193], "TND", 2048, 1),
        ([1, 5], [513, 8193], "TND", 2048, 1),
        ([1, 1], [0, 8193], "TND", 2048, 1),
        ([1], [0], "TND", 2048, 1),
        ([3, 3], [8193, 4097], "BSND", 65, 1),
        ([1, 5], [8193, 16385], "BSND", 2048, 1),
        ([1, 5], [8193, 16385], "TND", 2048, 4),
        ([1], [7], "TND", 2048, 1),
        ([1], [32769], "TND", 8192, 1),
    ],
)
@torch.inference_mode()
def test_a5_split_k_matches_metadata(queries, lengths, layout, topk, ratio, pattern, quant_mode):
    run, active_rows, _ = _make_a5_case(
        queries, lengths, quant_mode=quant_mode, topk=topk, layout=layout, pattern=pattern, ratio=ratio
    )
    expected = run(serial=True)
    _check_a5_result(active_rows(run()), expected)


@pytest.mark.parametrize("quant_mode", [1, 2])
@torch.inference_mode()
def test_a5_split_k_graph_changes_dispatch(quant_mode):
    run, active_rows, update = _make_a5_case([1, 5], [8193, 32769], quant_mode=quant_mode)
    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [run() for _ in range(CAPTURE_CALLS)]
    for lengths in ([8193, 32769], [4097, 8193], [513, 8193], [0, 4097], [8193, 32769]):
        update(lengths)
        expected = run(serial=True)
        for _ in range(4):
            graph.replay()
            for actual in outputs:
                _check_a5_result(active_rows(actual), expected)


@pytest.mark.parametrize("quant_mode", [1, 2])
@torch.inference_mode()
def test_a5_split_k_preserves_tnd_padding(quant_mode):
    run, active_rows, _ = _make_a5_case([5, 3], [8193, 16385], quant_mode=quant_mode, used_counts=[1, 0])
    _check_a5_result(active_rows(run()), run(serial=True))


@pytest.mark.parametrize("quant_mode", [1, 2])
@torch.inference_mode()
def test_a5_split_k_matches_cpu_topk(quant_mode):
    run, active_rows, _ = _make_a5_case([1, 5], [8193, 32769], quant_mode=quant_mode, pattern="ramp")
    active_rows(run())


@pytest.mark.parametrize("quant_mode", [1, 2])
@torch.inference_mode()
def test_a5_split_k_glm_long_context_performance(quant_mode, record_property):
    run, active_rows, _ = _make_a5_case([1], [262144], quant_mode=quant_mode)
    _check_a5_result(active_rows(run()), run(serial=True))
    graphs, outputs = {}, {}
    for name, serial in (("current", False), ("metadata", True)):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs[name] = [run(serial=serial, return_value=0) for _ in range(CAPTURE_CALLS)]
        for _ in range(30):
            graph.replay()
        graphs[name] = graph
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    raw = {name: [] for name in graphs}
    for iteration in range(30):
        for name in list(graphs) if iteration % 2 == 0 else list(reversed(graphs)):
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            raw[name].append(start.elapsed_time(end) * 1000 / CAPTURE_CALLS)
    medians = {name: statistics.median(values) for name, values in raw.items()}
    record_property("split_k_a5_performance", json.dumps({"quant_mode": quant_mode, "median_us": medians}))
    assert medians["current"] < 0.8 * medians["metadata"], medians
