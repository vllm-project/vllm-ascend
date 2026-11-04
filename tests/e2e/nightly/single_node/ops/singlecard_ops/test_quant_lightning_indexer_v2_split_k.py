# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
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

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="QLI split-K requires Ascend A2/A3",
)

# 补充空请求，使请求数超过两种支持架构的AIC数量，强制使用原metadata调度。
# 有效请求的Q、K、候选列表和因果坐标保持不变。
REFERENCE_REQUESTS = 64
HEADS = 32
WIDTH = 128
PAGE_SIZE = 128
CANDIDATE_BLOCKS = 2048
CANDIDATE_BLOCK_SIZE = 8
CAPTURE_CALLS = 4
TIMING_WARMUP = 30
TIMING_SAMPLES = 30
MAX_SPLIT_K_LATENCY_RATIO = 0.8


def _make_split_k_case(query_counts, key_lengths, layout, ratio, mask, topk, *, pattern="random", hybrid_stride=False):
    """构造变长请求，使用互不重叠的乱序物理页和逐行掩码。"""
    key_lengths = list(key_lengths)
    generator = torch.Generator().manual_seed(137)
    batch, tokens = len(query_counts), sum(query_counts)
    pages = max(1, (max(key_lengths) + PAGE_SIZE - 1) // PAGE_SIZE)
    query = torch.randint(-8, 9, (tokens, HEADS, WIDTH), dtype=torch.int8, generator=generator)
    weights = (torch.rand(tokens, HEADS, generator=generator) * 2 - 1).half()
    query_scale = torch.full((tokens, HEADS), 0.01, dtype=torch.float16)
    if pattern == "zero":
        weights.zero_()
    key_cpu = torch.randint(-8, 9, (batch * pages, PAGE_SIZE, 1, WIDTH), dtype=torch.int8, generator=generator)
    key_stride = 147712 if hybrid_stride else (PAGE_SIZE + 16) * WIDTH
    scale_stride = 73856 if hybrid_stride else PAGE_SIZE + 32
    key = torch.empty_strided(key_cpu.shape, (key_stride, WIDTH, WIDTH, 1), dtype=torch.int8, device="npu")
    key.copy_(key_cpu)
    scale = torch.empty_strided(key_cpu.shape[:-1], (scale_stride, 1, 1), dtype=torch.float16, device="npu")
    scale.fill_(0.01)
    table = torch.randperm(batch * pages, generator=generator).int().reshape(batch, pages)
    candidates = torch.full((tokens, 1, CANDIDATE_BLOCKS), -1, dtype=torch.int32)
    prefix = [0]
    for count, length in zip(query_counts, key_lengths):
        for row in range(prefix[-1], prefix[-1] + count):
            blocks = torch.randperm((length + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE, generator=generator)
            blocks = blocks[:CANDIDATE_BLOCKS]
            candidates[row, 0, : blocks.numel()] = blocks.int()
        prefix.append(prefix[-1] + count)
    if pattern == "duplicate":
        candidates.zero_()
    elif pattern == "empty":
        candidates.fill_(-1)
    offsets = (torch.arange(tokens, dtype=torch.int32) * 17).reshape(tokens, 1)
    active = [query, weights, query_scale, candidates, offsets]

    def arguments(reference):
        # BSND参考调用使用TND布局下相同的有效行。
        # 被测BSND输出的填充行在下方单独检查。
        values = active
        if layout == "BSND" and not reference:
            values = []
            for tensor in active:
                padded = torch.zeros((batch, max(query_counts), *tensor.shape[1:]), dtype=tensor.dtype)
                for request, count in enumerate(query_counts):
                    padded[request, :count] = tensor[prefix[request] : prefix[request + 1]]
                values.append(padded)
        common = dict(
            seqused_k=torch.tensor(key_lengths, dtype=torch.int32),
            max_seqlen_q=max(query_counts),
            layout_q="TND" if reference else layout,
            layout_k="PA_BBND",
            mask_mode=mask,
            cmp_ratio=ratio,
        )
        current_table = table
        if common["layout_q"] == "TND":
            common["cu_seqlens_q"] = torch.tensor(prefix, dtype=torch.int32)
        else:
            common["seqused_q"] = torch.tensor(query_counts, dtype=torch.int32)
        if reference:
            extra = REFERENCE_REQUESTS - batch if max(key_lengths) > 0 else 0
            current_table = torch.cat((table, torch.zeros(extra, pages, dtype=torch.int32)))
            common["seqused_k"] = torch.cat((common["seqused_k"], torch.zeros(extra, dtype=torch.int32)))
            common["cu_seqlens_q"] = torch.tensor(prefix + [tokens] * extra, dtype=torch.int32)
        if ratio > 1 and mask == 3:
            common["cmp_residual_k"] = torch.ones(current_table.shape[0], dtype=torch.int32)
        common = {name: value.npu() if isinstance(value, torch.Tensor) else value for name, value in common.items()}
        metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            HEADS,
            1,
            WIDTH,
            topk,
            2,
            batch_size=current_table.shape[0],
            max_seqlen_k=max(1, max(key_lengths) * ratio + int(ratio > 1 and mask == 3)),
            **common,
        )
        return [tensor.npu() for tensor in values], current_table.npu(), metadata, common

    tested = arguments(False)
    reference = arguments(True)

    def run(mode, *, serial=False):
        values, current_table, metadata, common = reference if serial else tested
        q, w, qs, blocks, row_offsets = values
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
            q,
            key,
            w,
            qs,
            scale,
            topk,
            2,
            block_table=current_table,
            output_idx_offset=row_offsets,
            metadata=metadata,
            candidate_topk_index=blocks if mode == 2 else None,
            candidate_mode=mode,
            candidate_topk_blocks=CANDIDATE_BLOCKS,
            candidate_block_size=CANDIDATE_BLOCK_SIZE,
            **common,
        )[0]

    def active_rows(output):
        if layout == "TND":
            return output
        for request, count in enumerate(query_counts):
            assert (output[request, count:] == -1).all(), "BSND padding must remain invalid"
        return torch.cat([output[request, :count] for request, count in enumerate(query_counts)])

    def update(lengths, iteration):
        """在原地址更新图捕获的张量（含metadata），无需重新构图。"""
        assert len(lengths) == batch and max(lengths) <= pages * PAGE_SIZE
        key_lengths[:] = lengths
        generator.manual_seed(137 + iteration)
        query.copy_(torch.randint(-8, 9, query.shape, dtype=query.dtype, generator=generator))
        weights.copy_((torch.rand(weights.shape, generator=generator) * 2 - 1).half())
        if iteration % 2:
            weights.zero_()
        key.copy_(torch.randint(-8, 9, key_cpu.shape, dtype=key_cpu.dtype, generator=generator))
        table.copy_(torch.randperm(batch * pages, generator=generator).int().reshape(batch, pages))
        for old, fresh in ((tested, arguments(False)), (reference, arguments(True))):
            for dest, src in zip(old[0], fresh[0]):
                dest.copy_(src)
            old[1].copy_(fresh[1])
            old[2].copy_(fresh[2])
            for name, tensor in old[3].items():
                if isinstance(tensor, torch.Tensor):
                    tensor.copy_(fresh[3][name])

    return run, active_rows, update


@pytest.mark.parametrize("mode", [2, 3])
@pytest.mark.parametrize("pattern", ["random", "zero"])
@pytest.mark.parametrize(
    "query_counts,key_lengths,layout,ratio,mask,topk",
    [
        ([1], [4096], "TND", 1, 3, 1),
        ([6], [4097], "TND", 1, 3, 65),
        ([1, 5, 2], [8193, 16385, 4097], "TND", 2, 3, 513),
        ([0, 3, 0, 1], [0, 8193, 0, 4097], "TND", 1, 3, 512),
        ([1] * 8, [8193] * 8, "TND", 4, 0, 1024),
        ([1] * 12, [8193] * 12, "TND", 1, 3, 2048),
        ([1] * 13, [8193] * 13, "TND", 1, 3, 512),
        ([1] * 24, [4097] * 24, "TND", 1, 3, 512),
        ([47], [8193], "TND", 1, 3, 63),
        ([49], [8193], "TND", 1, 3, 512),
        ([1, 5], [513, 8193], "TND", 1, 3, 512),
        ([1, 1], [0, 8193], "TND", 1, 3, 512),
        ([1], [0], "TND", 1, 3, 512),
        ([3, 3], [8193, 4097], "BSND", 1, 3, 512),
        ([1, 5], [8193, 16385], "BSND", 1, 3, 512),
        ([1], [7], "TND", 1, 3, 512),
    ],
)
@torch.inference_mode()
def test_split_k_matches_metadata_schedule(query_counts, key_lengths, layout, ratio, mask, topk, mode, pattern):
    """比较索引多重集合，严格验证同分选择和输出偏移。"""
    run, active_rows, _ = _make_split_k_case(query_counts, key_lengths, layout, ratio, mask, topk, pattern=pattern)
    expected = run(mode, serial=True).cpu().sort(dim=-1).values
    for _ in range(2):
        actual = active_rows(run(mode).cpu()).sort(dim=-1).values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("pattern", ["duplicate", "empty"])
@torch.inference_mode()
def test_split_k_preserves_insufficient_candidate_fill(pattern):
    run, _, _ = _make_split_k_case([1, 5], [8193, 16385], "TND", 1, 3, 512, pattern=pattern)
    expected = run(2, serial=True).cpu().sort(dim=-1).values
    torch.testing.assert_close(run(2).cpu().sort(dim=-1).values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [2, 3])
@pytest.mark.parametrize("query_counts", [[1], [1, 1], [6]])
@torch.inference_mode()
def test_split_k_long_context_repeated(query_counts, mode):
    """反复执行长上下文计算图，检验多轮分片归并的稳定性。"""
    run, _, _ = _make_split_k_case(query_counts, [262144] * len(query_counts), "TND", 1, 3, 512, hybrid_stride=True)
    expected = run(mode, serial=True).cpu().sort(dim=-1).values
    torch.testing.assert_close(run(mode).cpu().sort(dim=-1).values, expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [run(mode) for _ in range(4)]
    for _ in range(64):
        graph.replay()
        for output in outputs:
            torch.testing.assert_close(output.cpu().sort(dim=-1).values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [2, 3])
@torch.inference_mode()
def test_split_k_graph_replay_changes_dispatch(mode):
    run, _, update = _make_split_k_case([1, 5], [8193, 16385], "TND", 1, 3, 512)
    run(mode)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = run(mode)
    # 在同一捕获图中跨越K阈值、加入零K请求，
    # 并更新权重、query、缓存内容和页映射。
    for iteration, lengths in enumerate(([8193, 16385], [4097, 8193], [513, 8193], [0, 4097], [8193, 16385])):
        update(lengths, iteration)
        expected = run(mode, serial=True).cpu().sort(dim=-1).values
        graph.replay()
        torch.testing.assert_close(actual.cpu().sort(dim=-1).values, expected, rtol=0, atol=0)


def _measure_split_k_case(
    query_counts, key_lengths, mode, *, warmup=TIMING_WARMUP, samples=TIMING_SAMPLES, compare_reference=True
):
    """先校验结果，再测量图重放中的算子耗时，排除输入准备和metadata构建。"""
    run, _, _ = _make_split_k_case(query_counts, key_lengths, "TND", 1, 3, 512, hybrid_stride=True)
    expected = run(mode, serial=True).cpu().sort(dim=-1).values
    torch.testing.assert_close(run(mode).cpu().sort(dim=-1).values, expected, rtol=0, atol=0)
    schedules = {"current": False}
    if compare_reference:
        schedules["metadata"] = True
    graphs, outputs = {}, {}
    for name, serial in schedules.items():
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs[name] = [run(mode, serial=serial) for _ in range(CAPTURE_CALLS)]
        for _ in range(warmup):
            graph.replay()
        graphs[name] = graph
    torch.npu.synchronize()
    raw: dict[str, list[float]] = {name: [] for name in schedules}
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    for iteration in range(samples):
        # 交替测量顺序，减小频率和温度漂移对两种调度的偏置。
        order = list(schedules) if iteration % 2 == 0 else list(reversed(schedules))
        for name in order:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            raw[name].append(start.elapsed_time(end) * 1000 / CAPTURE_CALLS)
    for values in outputs.values():
        for output in values:
            torch.testing.assert_close(output.cpu().sort(dim=-1).values, expected, rtol=0, atol=0)
    return {
        "queries": query_counts,
        "lengths": key_lengths,
        "mode": mode,
        "samples_us": raw,
        "median_us": {name: statistics.median(values) for name, values in raw.items()},
        "output_sha256": hashlib.sha256(expected.contiguous().numpy().tobytes()).hexdigest(),
    }


@pytest.mark.parametrize("mode", [2, 3])
@pytest.mark.parametrize("query_counts", [[1], [1, 1], [6]])
@torch.inference_mode()
def test_split_k_performance(query_counts, mode, record_property):
    """长K场景应明显快于metadata调度，防止小query路径退回原调度。"""
    result = _measure_split_k_case(query_counts, [262144] * len(query_counts), mode)
    result["speedup"] = result["median_us"]["metadata"] / result["median_us"]["current"]
    record_property("split_k_performance", json.dumps(result))
    print(json.dumps(result), flush=True)
    # 参考调用仅补充空请求以关闭split-K，用于检验关键场景的加速收益。
    assert result["median_us"]["current"] <= MAX_SPLIT_K_LATENCY_RATIO * result["median_us"]["metadata"], (
        f"Split-K lost its long-context speedup: {result['speedup']:.3f}x; "
        f"expected at least {1 / MAX_SPLIT_K_LATENCY_RATIO:.2f}x"
    )
