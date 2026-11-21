# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the production indexer pipeline on CPU with device APIs simulated.

Run with --confcutdir=tests/ut/attention. Stream tests check submission ordering,
not device concurrency, graph replay or CANN numerical accuracy.
"""

import ast
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def quantize(x, dst_type):
    limit = 127 if dst_type == torch.int8 else 448
    scale = x.float().abs().amax(-1).clamp_min(1e-12) / limit
    values = x.float() / scale[..., None]
    if dst_type == torch.int8:
        values = values.round().clamp(-128, 127)
    return values.to(dst_type), scale


def scatter(cache, updates, slots):
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            block, token = divmod(slot, cache.shape[1])
            cache[block, token, 0].copy_(updates[row].reshape(-1))


@pytest.fixture
def runtime(monkeypatch):
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(f"{self.name}:wait:{other.name}")

    main, aux = Stream("main"), Stream("q")
    current = [main]

    @contextmanager
    def stream(selected):
        prior = current[0]
        current[0] = selected
        events.append(f"enter:{selected.name}")
        yield
        current[0] = prior
        events.append(f"exit:{selected.name}")

    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, selected: events.append(f"record:{selected.name}"))
    matmul = torch.Tensor.__matmul__

    def hadamard(x, other):
        events.append(f"{current[0].name}:hadamard:{x.shape[1]}")
        return matmul(x, other)

    def dynamic_quant(x, dst_type):
        events.append(f"{current[0].name}:quant:{x.shape[0]}")
        return quantize(x, dst_type)

    monkeypatch.setattr(torch.Tensor, "__matmul__", hadamard)
    npu = SimpleNamespace(
        float4_e2m1fn_x2=object(),
        float8_e8m0fnu=object(),
        npu_dynamic_quant=Mock(side_effect=dynamic_quant),
        npu_quant_matmul=Mock(side_effect=lambda x, *args, output_dtype, **kwargs: x.to(output_dtype)),
        npu=SimpleNamespace(current_stream=lambda: current[0], stream=stream, Stream=lambda: aux),
    )
    scope = dict(
        torch=torch,
        torch_npu=npu,
        nn=nn,
        AttentionBackend=object,
        INDEXER_K_CACHE_SLOT=0,
        INDEXER_SCALE_CACHE_SLOT=1,
        HAS_TRITON=False,
    )
    helpers = ast.parse((ROOT / "vllm_ascend/attention/context_parallel/sfa_dcp_utils.py").read_text())
    helpers.body = [
        node
        for node in helpers.body
        if isinstance(node, ast.FunctionDef) and node.name in {"as_byte_view", "_index_select_kv_blocks"}
    ]
    exec(compile(helpers, "sfa_dcp_utils.py", "exec"), scope)
    tree = ast.parse((ROOT / "vllm_ascend/attention/indexer.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAIndexerBackend")
    stream_nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_indexer_qk_stream")
        or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "_INDEXER_QK_STREAM")
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, *stream_nodes, node], type_ignores=[])),
            "indexer.py",
            "exec",
        ),
        scope,
    )
    cls = scope["AscendSFAIndexerBackend"]

    def make(*, fused=False, overlap=False, paired=False, dtype=torch.float8_e4m3fn):
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.head_dim, model.n_head = 128, 2
        model.qk_rope_head_dim = 0
        model.enable_sparse_li_c8, model.enable_sparse_li_c4 = True, False
        model.c8_k_cache_dtype = dtype
        model.c8_k_scale_cache_dtype = torch.float32 if dtype == torch.float8_e4m3fn else torch.float16
        model.q_hadamard = torch.eye(128, dtype=torch.bfloat16)
        model.k_hadamard = model.q_hadamard
        model._pcp_active = model._dsa_cp_active = model._sharded_dcp = False
        model._q_stream = aux if overlap else None
        model.k_norm = nn.LayerNorm(128, dtype=torch.bfloat16)
        model.k_norm.register_forward_hook(lambda *args: events.append(f"{current[0].name}:k_norm"))
        if paired:
            # Independent K/scale planes in each padded physical page.
            page_bytes = 4 * (128 + 4) + 64
            backing = torch.full((3 * page_bytes,), 17, dtype=torch.uint8)
            key = backing.view(dtype).as_strided((3, 4, 1, 128), (page_bytes, 128, 128, 1))
            scale = backing.view(torch.float32).as_strided((3, 4, 1, 1), (page_bytes // 4, 1, 1, 1), 128)
        else:
            backing = None
            key = torch.zeros(3, 4, 1, 128, dtype=dtype)
            scale = torch.zeros(3, 4, 1, 1, dtype=model.c8_k_scale_cache_dtype)
        model.k_cache = SimpleNamespace(kv_cache=(key, scale))

        def wk(x):
            events.append("k_cube")
            return torch.cat((x[:, :128], x[:, :2]), dim=-1), None

        def wq(x):
            events.append("q_cube")
            return x, None

        model.wk_weights_proj, model.wq_b = Mock(side_effect=wk), Mock(side_effect=wq)

        def write(k, s, slots):
            events.append("cache_write")
            scatter(key, k, slots)
            scatter(scale, s, slots)

        model.write_cache = Mock(side_effect=write)

        def fused_write(k_cache, s_cache, x, slots, *, quant_mode, round_scale=True):
            assert quant_mode == "fp8" and slots.dtype == torch.int32
            assert round_scale is False, "Preserve npu_dynamic_quant's unrounded FP8 scale"
            assert k_cache is key and s_cache is scale
            events.append("fused_cache_write")
            k, s = quantize(x, key.dtype)
            scatter(key, k, slots)
            scatter(scale, s, slots)

        model._fused_k_cache = Mock(side_effect=fused_write) if fused else None
        return model, backing

    return SimpleNamespace(make=make, events=events, npu=npu, scope=scope)


def inputs():
    torch.manual_seed(2)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    q = torch.randn(4, 256, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        kv_gather_block_ids=None,
        cos=torch.empty(0),
        sin=torch.empty(0),
        slot_mapping=torch.tensor([1, 5, -1, 11]),
        actual_seq_lengths_query=torch.arange(1, 5),
        actual_seq_lengths_key=torch.arange(1, 5),
    )

    def select(q, weights, backend, meta, scales):
        return q.float().clone(), weights.clone(), scales.clone()

    metadata.topk_selector = select
    return x, q, metadata


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_fused_k_scale_cache_preserves_slots_pages_and_q(runtime, paired, overlap):
    baseline, old_backing = runtime.make(paired=paired)
    candidate, new_backing = runtime.make(fused=True, overlap=overlap, paired=paired)
    x, q, metadata = inputs()
    expected = baseline(x, q, x, metadata)
    actual = candidate(x, q, x, metadata)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(candidate.k_cache.kv_cache, baseline.k_cache.kv_cache):
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)
    if paired:
        torch.testing.assert_close(new_backing, old_backing, rtol=0, atol=0)
    candidate._fused_k_cache.assert_called_once()
    candidate.write_cache.assert_not_called()
    candidate.wk_weights_proj.assert_called_once()
    candidate.wq_b.assert_called_once()


def test_q_stream_waits_for_k_projection_and_joins_before_topk(runtime):
    model, _ = runtime.make(fused=True, overlap=True)
    x, q, metadata = inputs()
    metadata.topk_selector = lambda *args: runtime.events.append("topk")
    handle = SimpleNamespace(wait=lambda: runtime.events.append("attn_q_gather_wait"))
    model(x, q, x, metadata, attn_q_gather_handle=handle)
    events = runtime.events
    assert events.index("k_cube") < events.index("q:wait:main") < events.index("q_cube")
    assert events.index("attn_q_gather_wait") < events.index("fused_cache_write")
    main_waits = [i for i, event in enumerate(events) if event == "main:wait:q"]
    aux_waits = [i for i, event in enumerate(events) if event == "q:wait:main"]
    assert len(main_waits) == len(aux_waits) == 2
    assert events.index("q:hadamard:2") < main_waits[0] < aux_waits[1]
    assert events.index("main:k_norm") < main_waits[0]
    assert aux_waits[1] < events.index("q:hadamard:1")
    assert main_waits[0] < events.index("main:quant:8") < main_waits[1]
    assert events.index("q:hadamard:1") < main_waits[1] < events.index("fused_cache_write")
    assert events.index("fused_cache_write") < events.index("topk")
    assert not any(":wait:" in event for event in events[events.index("fused_cache_write") + 1 :])
    assert events.count("record:q") == events.count("record:main") == 2


@pytest.mark.parametrize("overlap", [False, True])
def test_shared_topk_layer_only_updates_k(runtime, overlap):
    model, _ = runtime.make(fused=True, overlap=overlap)
    x, q, metadata = inputs()
    assert model(x, q, x, metadata, compute_topk=False) is None
    model.wq_b.assert_not_called()
    model.wk_weights_proj.assert_called_once()
    model._fused_k_cache.assert_called_once()
    assert "enter:q" not in runtime.events


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("local_queries", [0, 4])
@pytest.mark.parametrize("shared_topk", [False, True])
def test_full_history_read_preserves_sharded_writes_and_empty_senders(runtime, paired, local_queries, shared_topk):
    model, _ = runtime.make(overlap=True, paired=paired)
    model._sharded_dcp = True
    model.use_torch_npu_lightning_indexer = False
    model._get_local_qli_inputs = Mock(side_effect=AssertionError("full history must not use owner-local lengths"))
    x, q, metadata = inputs()
    metadata.topk_selector = None
    metadata.num_actual_tokens = local_queries
    metadata.kv_gather_block_ids = torch.tensor([2, 0], dtype=torch.int32)
    metadata.block_table = torch.tensor([[0, 2, 1, 3]], dtype=torch.int32)
    original_cache = model.k_cache.kv_cache
    calls = []

    def gather(blocks, group):
        assert runtime.events[-1] == "cache_write"
        slot = len(calls)
        cache = original_cache[slot]
        source = cache.view(torch.uint8) if slot == 0 else cache
        expected = source.index_select(0, metadata.kv_gather_block_ids)
        torch.testing.assert_close(blocks, expected, rtol=0, atol=0)
        result = torch.empty((4, *blocks.shape[1:]), dtype=blocks.dtype)

        def wait():
            # A real asynchronous gather is not readable until joined.
            result.copy_(torch.cat((blocks, blocks.flip(0))))

        handle = SimpleNamespace(wait=Mock(side_effect=wait))
        calls.append((result, handle))
        return result, handle

    def select(*args):
        cache = args[4]
        assert all(handle.wait.call_count == 1 for _, handle in calls)
        for slot, tensor in enumerate(cache):
            expected = calls[slot][0].view(original_cache[slot].dtype)
            torch.testing.assert_close(tensor.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
        assert args[7] is metadata
        assert args[9] is metadata.actual_seq_lengths_key
        return torch.tensor([[[0, 7, 8, 15]]], dtype=torch.int32)

    selector = Mock(side_effect=select)
    runtime.scope.update(
        all_gather_async=gather,
        get_dcp_group=lambda: object(),
        DeviceOperator=SimpleNamespace(indexer_select_post_process=selector),
    )
    result = model(x, q, x, metadata, compute_topk=not shared_topk)
    model.write_cache.assert_called_once()
    assert model.k_cache.kv_cache is original_cache
    assert len(calls) == (0 if shared_topk else 2)
    if local_queries and not shared_topk:
        selector.assert_called_once()
        assert result.tolist() == [[[0, 7, 8, 15]]]
    else:
        assert result is None
        selector.assert_not_called()
        model.wq_b.assert_not_called()


@pytest.mark.parametrize(
    "reason", ["pcp", "dsa_cp", "dtype", "empty", "cache_dtype", "scale_dtype", "cache_shape", "no_op"]
)
def test_ineligible_cache_contracts_keep_split_path(runtime, reason):
    model, _ = runtime.make(fused=True)
    x, _, _ = inputs()
    if reason == "pcp":
        model._pcp_active = True
    elif reason == "dsa_cp":
        model._dsa_cp_active = True
    elif reason == "dtype":
        x = x.float()
    elif reason == "empty":
        x = x[:0]
    elif reason == "no_op":
        model._fused_k_cache = None
    elif reason == "cache_dtype":
        model.k_cache.kv_cache = (model.k_cache.kv_cache[0].to(torch.int8), model.k_cache.kv_cache[1])
    elif reason == "scale_dtype":
        model.k_cache.kv_cache = (model.k_cache.kv_cache[0], model.k_cache.kv_cache[1].half())
    elif reason == "cache_shape":
        model.k_cache.kv_cache = tuple(t.squeeze(2) for t in model.k_cache.kv_cache)
    assert not model._can_fuse_k_cache(x)


def test_int8_q_overlap_keeps_native_k_write(runtime):
    baseline, _ = runtime.make(dtype=torch.int8)
    model, _ = runtime.make(dtype=torch.int8, overlap=True)
    x, q, metadata = inputs()
    expected = baseline(x, q, x, metadata)
    result = model(x, q, x, metadata)
    for a, b in zip(result, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    model.write_cache.assert_called_once()
    assert "main:wait:q" in runtime.events


@pytest.mark.parametrize("case", ["a5", "a3", "lora", "other_model", "rope", "subclass", "missing_op"])
def test_pipeline_preparation_is_scoped_and_allocates_stream_once(runtime, monkeypatch, case):
    model, _ = runtime.make(dtype=torch.int8 if case == "a3" else torch.float8_e4m3fn)
    model._is_kimi_k31 = case != "other_model"
    model.qk_rope_head_dim = 64 if case == "rope" else 0
    if case == "subclass":
        model.__class__ = type("OtherIndexer", (type(model),), {})
    config = SimpleNamespace(lora_config=object() if case == "lora" else None)
    runtime.scope["get_current_vllm_config"] = lambda: config
    runtime.scope["get_current_hardware_profile"] = lambda: SimpleNamespace(supports=lambda cap: case != "a3")
    runtime.scope["HardwareCapability"] = SimpleNamespace(INDEXER_QUANT_CACHE=object())
    runtime.scope["find_spec"] = lambda name: None if case == "missing_op" else object()
    op = Mock()
    monkeypatch.setitem(
        sys.modules, "cann_ops_transformer", SimpleNamespace(ops=SimpleNamespace(indexer_quant_cache=op))
    )
    create_stream = Mock(wraps=runtime.npu.npu.Stream)
    runtime.npu.npu.Stream = create_stream
    model.process_weights_after_loading()
    model.process_weights_after_loading()
    eligible = case in ("a5", "a3", "missing_op")
    assert create_stream.call_count == int(eligible)
    assert (model._q_stream is not None) is eligible
    assert (model._fused_k_cache is op) is (case == "a5")


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_indexer_layers_share_stream_across_reload_and_forward(runtime, dtype):
    runtime.scope["get_current_vllm_config"] = lambda: SimpleNamespace(lora_config=None)
    runtime.scope["get_current_hardware_profile"] = lambda: SimpleNamespace(supports=lambda cap: False)
    runtime.scope["HardwareCapability"] = SimpleNamespace(INDEXER_QUANT_CACHE=object())
    create_stream = Mock(wraps=runtime.npu.npu.Stream)
    runtime.npu.npu.Stream = create_stream
    layers = [runtime.make(dtype=dtype)[0] for _ in range(24)]
    for layer in layers:
        layer._is_kimi_k31 = True
        layer.process_weights_after_loading()
    assert create_stream.call_count == 1
    assert all(layer._q_stream is layers[0]._q_stream for layer in layers)
    x, q, metadata = inputs()
    for _ in range(2):
        for layer in layers:
            layer.process_weights_after_loading()
            layer(x, q, x, metadata)
    assert create_stream.call_count == 1


@pytest.mark.parametrize("overlap", [False, True])
def test_cp_cache_collectives_remain_after_attention_query_dependency(runtime, overlap):
    model, _ = runtime.make(fused=True, overlap=overlap)
    model._dsa_cp_active = True
    runtime.scope["get_tp_group"] = lambda: object()

    def gather(value, group, async_op):
        runtime.events.append("cache_gather")
        return value, SimpleNamespace(wait=lambda: runtime.events.append("cache_gather_wait"))

    runtime.scope["all_gather_async"] = gather
    x, q, metadata = inputs()
    handle = SimpleNamespace(wait=lambda: runtime.events.append("attn_q_wait"))
    model(x, q, x, metadata, attn_q_gather_handle=handle)
    events = runtime.events
    assert events.index("attn_q_wait") < events.index("cache_gather")
    assert events.count("cache_gather") == events.count("cache_gather_wait") == 2
    assert max(i for i, event in enumerate(events) if event == "cache_gather_wait") < events.index("cache_write")
    model._fused_k_cache.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_prequantized_latent_q_keeps_matmul_contract_and_stream_lifetime(runtime, dtype):
    model, _ = runtime.make(overlap=True)
    model.wq_b.quant_method = SimpleNamespace(quant_method=SimpleNamespace(group_size=32))
    x, q, metadata = inputs()
    q = q.to(dtype)
    scale = torch.ones(4, 1) if dtype == torch.int8 else torch.ones(4, 8)
    model(x, (q, scale), x, metadata)
    model.wq_b.assert_not_called()
    runtime.npu.npu_quant_matmul.assert_called_once()
    kwargs = runtime.npu.npu_quant_matmul.call_args.kwargs
    assert kwargs["output_dtype"] == torch.bfloat16
    assert kwargs["pertoken_scale"].shape == ((4,) if dtype == torch.int8 else (4, 4, 2))
    assert runtime.events.count("record:q") == 3
