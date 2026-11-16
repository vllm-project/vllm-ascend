# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real A3 operator/adapter tests; run with --noconftest for isolated bring-up.

Select a device with ASCEND_RT_VISIBLE_DEVICES before starting pytest. These
tests require the flash-attention-npu 910 MLA release; no numerical mock is
used. They do not replace a K3.1 model or multi-rank serving test.
"""

import ast
import importlib.util
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(
    not torch_npu.npu.is_available() or "Ascend910" not in torch_npu.npu.get_device_name(),
    reason="requires an A3/910 NPU",
)


@pytest.fixture(scope="module")
def sink():
    # Load the production boundary without importing an installed vLLM plugin
    # with a potentially different version. Native provider imports stay real.
    root = Path(__file__).resolve().parents[4]
    path = root / "vllm_ascend/attention/a3_mla_sink.py"
    spec = importlib.util.spec_from_file_location("a3_native_test_adapter", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_case(sink, dtype, qlens, klens, causal, *, heads=8, scale=192**-0.5, padded=False):
    torch.manual_seed(31415)
    page = 128
    columns = max(1, math.ceil(max(klens) / page))
    batch, tokens = len(qlens), sum(qlens)
    pages = batch * columns
    config = sink.A3MLAConfig(
        heads,
        scale,
        mask_mode=3 if causal else 0,
        max_seqlen_q=tokens,
        max_seqlen_k=columns * page,
        dtype=dtype,
    )
    adapter = sink.A3MLAAdapter.load(config, "external")
    q_cpu = (torch.randn(tokens, heads, 576) * 0.2).to(dtype)
    cache_cpu = (torch.randn(pages, page, 1, 576) * 0.2).to(dtype)
    table_cpu = torch.arange(pages - 1, -1, -1, dtype=torch.int32).reshape(batch, columns)
    prefixes = [0]
    for length in qlens:
        prefixes.append(prefixes[-1] + length)
    q = q_cpu.npu()
    if padded:
        # Same physical component layout as the A3 combined KV pool: packed
        # within each page, strided between pages, including hybrid padding.
        stride = page * 576 + 32
        raw = torch.zeros(pages * stride, dtype=dtype, device="npu")
        content = torch.as_strided(raw, (pages, page, 1, 512), (stride, 512, 512, 1))
        rope = torch.as_strided(raw, (pages, page, 1, 64), (stride, 64, 64, 1), page * 512)
    else:
        content, rope = adapter.allocate_cache(pages, device="npu")
    content.copy_(cache_cpu[..., :512])
    rope.copy_(cache_cpu[..., 512:])
    data = dict(
        q=q,
        k_cache=(content, rope),
        block_table=table_cpu.npu(),
        cache_seqlens=torch.tensor(klens, dtype=torch.int32, device="npu"),
        cu_seqlens_q=torch.tensor(prefixes, dtype=torch.int32, device="npu"),
        seqused_q=torch.tensor(qlens, dtype=torch.int32, device="npu"),
    )
    return adapter, data, (q_cpu, cache_cpu, table_cpu, qlens, klens, causal, scale)


def reference(q, cache, table, qlens, klens, causal, scale):
    output = torch.zeros((*q.shape[:2], 512), dtype=torch.float64)
    lse = torch.full((q.shape[1], q.shape[0]), -torch.inf, dtype=torch.float64)
    start = 0
    for request, (qlen, klen) in enumerate(zip(qlens, klens)):
        keys = cache[table[request].long()].reshape(-1, 576)[:klen].double()
        for offset in range(qlen):
            visible = min(klen, max(0, klen - qlen + offset + 1)) if causal else klen
            if visible:
                scores = q[start + offset].double() @ keys[:visible].T * scale
                output[start + offset] = scores.softmax(-1) @ keys[:visible, :512]
                lse[:, start + offset] = scores.logsumexp(-1)
        start += qlen
    return output.transpose(0, 1), lse


def invoke(adapter, data):
    metadata = adapter.build_metadata(data["cache_seqlens"], data["cu_seqlens_q"], data["seqused_q"])
    assert metadata.dtype == torch.uint8 and metadata.numel() == 2376
    return adapter.attention(**data, metadata=metadata)


def compare(actual, expected):
    # Fixed before testing: BF16/FP16 kernel vs independent FP64 reference.
    torch.testing.assert_close(actual[0].cpu().double(), expected[0], atol=0.003, rtol=0.02)
    torch.testing.assert_close(actual[1].cpu().double(), expected[1], atol=0.003, rtol=0.001)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qlens,klens,padded",
    [((1,), (129,), False), ((3, 1), (257, 9), True), ((3, 1, 0), (2, 0, 0), True), ((1,), (16384,), False)],
)
def test_native_matches_fp64(sink, dtype, causal, qlens, klens, padded):
    adapter, data, oracle = make_case(sink, dtype, qlens, klens, causal, padded=padded)
    before = tuple(cache.clone() for cache in data["k_cache"])
    compare(invoke(adapter, data), reference(*oracle))
    for cache, snapshot in zip(data["k_cache"], before):
        torch.testing.assert_close(cache, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_native_explicit_scale_and_dcp_head_count(sink, causal):
    adapter, data, oracle = make_case(sink, torch.bfloat16, (2, 1), (257, 33), causal, heads=64, scale=0.37)
    compare(invoke(adapter, data), reference(*oracle))


@pytest.mark.parametrize("causal", [False, True])
def test_native_graph_refresh_lengths_and_reorder(sink, causal):
    adapter, data, oracle = make_case(sink, torch.bfloat16, (2, 1), (257, 33), causal, padded=True)
    metadata = adapter.build_metadata(data["cache_seqlens"], data["cu_seqlens_q"], data["seqused_q"])
    for _ in range(3):
        adapter.attention(**data, metadata=metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = adapter.attention(**data, metadata=metadata)
    pointer = metadata.data_ptr()
    for qlens, klens in [((2, 1), (257, 33)), ((1, 2), (9, 143)), ((2, 1), (129, 0))]:
        data["cache_seqlens"].copy_(torch.tensor(klens, dtype=torch.int32, device="npu"))
        data["cu_seqlens_q"].copy_(torch.tensor([0, qlens[0], 3], dtype=torch.int32, device="npu"))
        data["seqused_q"].copy_(torch.tensor(qlens, dtype=torch.int32, device="npu"))
        metadata.copy_(adapter.build_metadata(data["cache_seqlens"], data["cu_seqlens_q"], data["seqused_q"]))
        graph.replay()
        torch.npu.synchronize()
        assert metadata.data_ptr() == pointer
        compare(output, reference(*oracle[:3], qlens, klens, causal, oracle[-1]))


def test_native_eager_does_not_read_device_scalars(sink):
    adapter, data, _ = make_case(sink, torch.bfloat16, (2, 1), (257, 33), False)
    invoke(adapter, data)
    torch.npu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        invoke(adapter, data)
    torch.npu.synchronize()
    events = {event.key: event.count for event in profiler.key_averages()}
    scalar_reads = {
        name: count for name, count in events.items() if name in ("aten::item", "aten::_local_scalar_dense")
    }
    assert not scalar_reads, f"native MLA still reads device scalars on Host: {scalar_reads}"


def test_native_production_builder_refreshes_padded_batch(sink):
    # Execute the production builder, isolating only module imports from the
    # environment's installed vLLM. Its Tensor operations and provider are real.
    root = Path(__file__).resolve().parents[4]
    path = root / "vllm_ascend/attention/flashmla.py"
    names = {
        "FLASHMLA_QK_DIM",
        "FLASHMLA_V_DIM",
        "FLASHMLA_BLOCK_SIZE",
        "FlashMLACurrent",
        "FlashMLADecode",
        "FlashMLAMetadataBuilder",
    }
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if getattr(node, "name", None) in names
        or isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ]
    scope = dict(torch=torch, dataclass=dataclass, replace=replace, FlashMLAImpl=object)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), scope)
    adapter, data, oracle = make_case(sink, torch.bfloat16, (2, 1), (257, 33), False, padded=True)
    impl = SimpleNamespace(
        a3_mla_adapter=adapter,
        num_heads=8,
        scale=adapter.config.softmax_scale,
        dtype=torch.bfloat16,
        use_mla_rope=False,
        vllm_config=SimpleNamespace(speculative_config=object()),
    )
    builder = scope["FlashMLAMetadataBuilder"](impl, torch.device("npu"), 4, torch.empty(0, device="npu"))
    common = SimpleNamespace(
        num_actual_tokens=3,
        num_input_tokens=8,
        causal=False,
        seq_lens=data["cache_seqlens"],
        query_start_loc=data["cu_seqlens_q"],
        block_table_tensor=data["block_table"],
        slot_mapping=torch.arange(3, device="npu"),
        positions=torch.arange(3, device="npu"),
    )
    first = builder.build(common, 2, 3, False, retain_for_graph=True)
    pointer = first.schedule.data_ptr()
    first.query.zero_()
    first.query[:3].copy_(data["q"])
    for qlens, klens in [((2, 1), (257, 33)), ((1, 2), (37, 143))]:
        common.seq_lens.copy_(torch.tensor(klens, dtype=torch.int32, device="npu"))
        common.query_start_loc.copy_(torch.tensor([0, qlens[0], 3], dtype=torch.int32, device="npu"))
        batch = builder.build(common, 2, 3, False)
        assert batch is first and batch.schedule.data_ptr() == pointer
        out, lse = batch.adapter.attention(
            batch.query,
            data["k_cache"],
            block_table=batch.block_table,
            cache_seqlens=batch.cache_lens,
            cu_seqlens_q=batch.cu,
            seqused_q=batch.used_q,
            metadata=batch.schedule,
        )
        compare((out[:, :3], lse[:, :3]), reference(*oracle[:3], qlens, klens, False, oracle[-1]))
        assert torch.all(out[:, 3:] == 0).item()
        assert torch.isneginf(lse[:, 3:]).all().item()
