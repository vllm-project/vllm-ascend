# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; also runnable without vLLM/torch_npu using --confcutdir.

Numerical oracle deliberately unpages one request at a time and uses logsumexp;
it does not share mock gather, masking, or normalization helpers.
"""

import ast
import importlib.util
import math
import sys
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mock_op = load_source("vllm_ascend/attention/a3_mla_mock.py", "vllm_ascend.attention.a3_mla_mock")
sink = load_source("vllm_ascend/attention/a3_mla_sink.py", "vllm_ascend.attention.a3_mla_sink")


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def case(dtype=torch.bfloat16, heads=2, q_lengths=(2, 1), kv_lengths=(143, 57), causal=False, strided=False):
    generator = torch.Generator().manual_seed(812)
    page = 128
    columns = max(1, math.ceil(max(kv_lengths, default=0) / page))
    batch = len(q_lengths)
    physical = max(1, batch * columns)
    cache = torch.randn((physical, page, 1, 576), generator=generator).to(dtype) * 0.2
    if strided:
        backing = torch.empty((physical, page * 2, 1, 576), dtype=dtype)
        backing[:, ::2].copy_(cache)
        cache = backing[:, ::2]
    table = torch.arange(physical - 1, -1, -1, dtype=torch.int32)[: batch * columns].reshape(batch, columns)
    q = torch.randn((sum(q_lengths), heads, 576), generator=generator).to(dtype) * 0.2
    cu = torch.tensor((0, *torch.tensor(q_lengths).cumsum(0).tolist()), dtype=torch.int32)
    lengths = torch.tensor(kv_lengths, dtype=torch.int32)
    used = torch.tensor(q_lengths, dtype=torch.int32)
    cfg = sink.A3MLAConfig(heads, 192**-0.5, mask_mode=3 if causal else 0, dtype=dtype)
    adapter = sink.A3MLAAdapter.load(cfg, "mock").for_capacity(columns, q.shape[0])
    return adapter, q, cache, table, lengths, cu, used


def reference(q, cache, table, lengths, cu, used, scale, causal):
    out = torch.zeros((*q.shape[:2], 512), dtype=q.dtype)
    lse = torch.full((q.shape[1], q.shape[0]), -torch.inf, dtype=torch.float32)
    for request, length in enumerate(lengths.tolist()):
        start, end = int(cu[request]), int(cu[request + 1])
        if not length or not int(used[request]):
            continue
        pages = [cache[int(page), :, 0, :] for page in table[request].tolist()]
        kv = torch.cat(pages)[:length].double()
        for token in range(start, start + int(used[request])):
            visible = length if not causal else max(0, min(length, length - (end - start) + token - start + 1))
            if not visible:
                continue
            scores = q[token].double() @ kv[:visible].T * scale
            normalizer = torch.logsumexp(scores, dim=-1)
            value = torch.exp(scores - normalizer[:, None]) @ kv[:visible, :512]
            out[token] = value.to(q.dtype)
            lse[:, token] = normalizer.float()
    return out.transpose(0, 1), lse


def invoke(data, *, split_cache=False, metadata=None):
    adapter, q, cache, table, lengths, cu, used = data
    metadata = adapter.build_metadata(lengths, cu, used) if metadata is None else metadata
    cache_arg = (cache[..., :512], cache[..., 512:]) if split_cache else cache
    return adapter.attention(
        q, cache_arg, block_table=table, cache_seqlens=lengths, cu_seqlens_q=cu, seqused_q=used, metadata=metadata
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads", [1, 12])
@pytest.mark.parametrize(
    "q_lengths,kv_lengths", [((1,), (1,)), ((4, 2), (143, 57)), ((4, 1), (2, 0)), ((0, 2), (0, 129))]
)
def test_numerical_contract(dtype, causal, heads, q_lengths, kv_lengths):
    data = case(dtype, heads, q_lengths, kv_lengths, causal)
    adapter, q, cache, table, lengths, cu, used = data
    before = cache.clone()
    actual = invoke(data, split_cache=True)
    expected = reference(q, cache, table, lengths, cu, used, adapter.config.softmax_scale, causal)
    torch.testing.assert_close(actual[0], expected[0], atol=0.002, rtol=0.01)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-5)
    assert not torch.isnan(actual[0]).any()
    assert not torch.isnan(actual[1]).any()
    torch.testing.assert_close(cache, before, rtol=0, atol=0)


@pytest.mark.parametrize("split_cache", [False, True])
def test_strided_cache_and_explicit_scale(split_cache):
    data = list(case(strided=True))
    data[0] = replace(data[0], config=replace(data[0].config, softmax_scale=0.37))
    actual = invoke(data, split_cache=split_cache)
    adapter, q, cache, table, lengths, cu, used = data
    expected = reference(q, cache, table, lengths, cu, used, 0.37, False)
    torch.testing.assert_close(actual[0], expected[0], atol=0.002, rtol=0.01)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-5)


def test_unused_cache_nan_and_dummy_queries_do_not_leak():
    data = list(case(q_lengths=(2, 3), kv_lengths=(5, 0)))
    adapter, q, cache, table, lengths, cu, used = data
    cache[int(table[0, 0]), 5:] = torch.nan
    cache[int(table[1, 0])] = torch.nan
    used[1] = 0
    q[2:] = torch.nan
    output, lse = invoke(data)
    assert torch.isfinite(output[:, :2]).all()
    assert torch.equal(output[:, 2:], torch.zeros_like(output[:, 2:]))
    assert torch.isneginf(lse[:, 2:]).all()


def test_metadata_rejects_stale_lengths_prefixes_and_mask():
    data = list(case())
    adapter, _, _, _, lengths, cu, used = data
    old = adapter.build_metadata(lengths, cu, used)
    lengths[0] -= 1
    with pytest.raises(RuntimeError, match="stale"):
        invoke(data, metadata=old)
    invoke(data)
    lengths[0] += 1
    cu[1] = 1
    used[:] = cu[1:] - cu[:-1]
    with pytest.raises(RuntimeError, match="stale"):
        invoke(data, metadata=old)
    cu[1] = 2
    used[:] = cu[1:] - cu[:-1]
    data[0] = replace(adapter, config=replace(adapter.config, mask_mode=3))
    with pytest.raises(RuntimeError, match="stale"):
        invoke(data, metadata=old)


def test_no_host_length_reads(monkeypatch):
    data = case()

    def forbidden(*args, **kwargs):
        pytest.fail("hot path tried to read a Tensor on the host")

    with monkeypatch.context() as context:
        for name in ("item", "tolist", "cpu", "numpy", "__bool__", "__int__"):
            context.setattr(torch.Tensor, name, forbidden)
        invoke(data)


@pytest.mark.parametrize("failure", ["page", "length", "dtype", "partial_q", "capacity", "return_count", "lse"])
def test_rejects_invalid_contract(failure):
    data = list(case())
    adapter, _, _, table, lengths, _, used = data
    if failure == "page":
        table[0, 0] = -1
    elif failure == "length":
        lengths[0] = 999
    elif failure == "dtype":
        data[4] = lengths.long()
    elif failure == "partial_q":
        used[0] = 1
    elif failure == "capacity":
        with pytest.raises(ValueError, match="capacity"):
            adapter.for_capacity(129, 3)
        return
    elif failure == "return_count":
        data[0] = replace(adapter, attention_op=lambda *a, **kw: (None, None))
    else:

        def bad_lse(*args, **kwargs):
            out, lse, x, y = mock_op.flash_attn_with_kvcache(*args, **kwargs)
            return out, lse.half(), x, y

        data[0] = replace(adapter, attention_op=bad_lse)
    with pytest.raises((ValueError, RuntimeError)):
        invoke(data)


def test_external_provider_is_replaceable_and_never_silently_falls_back(monkeypatch):
    calls = []

    def attention(
        q,
        k_cache,
        v_cache,
        *,
        qv,
        cache_seqlens,
        page_table,
        cu_seqlens_q,
        max_seqlen_q,
        softmax_scale,
        causal,
        scheduler_metadata,
        num_splits,
        return_softmax_lse,
    ):
        calls.append((q.shape[-1], qv.shape[-1], softmax_scale, num_splits))
        return mock_op.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            qv=qv,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=causal,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            return_softmax_lse=return_softmax_lse,
        )

    module = SimpleNamespace(flash_attn_with_kvcache=attention, get_scheduler_metadata=mock_op.get_scheduler_metadata)
    monkeypatch.setattr(sink, "import_module", lambda name: module)
    data = list(case())
    native = sink.A3MLAAdapter.load(data[0].config, "external", "future_a3_operator")
    with pytest.raises(ValueError, match="NPU"):
        native.build_metadata(data[4], data[5], data[6])
    # CPU ABI simulation only; production external mode retains its NPU guard.
    data[0] = replace(native, is_mock=True)
    invoke(data)
    assert calls == [(64, 512, 192**-0.5, 1)]
    monkeypatch.setattr(sink, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ImportError):
        sink.A3MLAAdapter.load(data[0].config, "external", "incomplete")


def ast_symbols(path, names, scope):
    nodes = ast.parse((ROOT / path).read_text(encoding="utf-8")).body
    chosen = []
    for node in nodes:
        name = getattr(node, "name", None)
        if (
            name in names
            or isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        ):
            chosen.append(node)
    assert len(chosen) == len(names), (path, names)
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(ROOT / path), "exec"), scope)
    return scope


def ast_method(path, owner, name, scope):
    module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == owner)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    # Keep implementation intact; remove annotations requiring unavailable vLLM types.
    fn.returns = None
    for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
        arg.annotation = None
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT / path), "exec"), scope)
    return scope[name]


def builder(dcp=1, rank=0, causal=False):
    data = case(causal=causal)

    def local_lengths(lengths, *, dcp_size, dcp_rank, cp_kv_cache_interleave_size):
        chunk = cp_kv_cache_interleave_size
        return lengths // (chunk * dcp_size) * chunk + (lengths % (chunk * dcp_size) - dcp_rank * chunk).clamp(0, chunk)

    scope = dict(
        torch=torch,
        dataclass=dataclass,
        replace=replace,
        Callable=Callable,
        FlashMLAImpl=object,
        get_dcp_local_seq_lens=local_lengths,
    )
    names = [
        "FLASHMLA_QK_DIM",
        "FLASHMLA_V_DIM",
        "FLASHMLA_BLOCK_SIZE",
        "FlashMLACurrent",
        "FlashMLADecode",
        "FlashMLAMetadataBuilder",
    ]
    ast_symbols("vllm_ascend/attention/flashmla.py", names, scope)
    impl = SimpleNamespace(
        dcp_size=dcp,
        dcp_rank=rank,
        a3_mla_adapter=data[0],
        num_heads=2,
        dtype=torch.bfloat16,
        use_mla_rope=False,
        scale=192**-0.5,
        vllm_config=SimpleNamespace(
            speculative_config=object(), parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=2)
        ),
    )
    return scope["FlashMLAMetadataBuilder"](impl, torch.device("cpu"), 4, torch.empty(0))


def common(lengths=(15, 7), prefixes=(0, 2, 3), capacity=8):
    return SimpleNamespace(
        num_actual_tokens=prefixes[-1],
        num_input_tokens=capacity,
        causal=False,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        query_start_loc=torch.tensor(prefixes, dtype=torch.int32),
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.arange(prefixes[-1]),
        positions=torch.arange(prefixes[-1]),
    )


def test_real_builder_reuses_graph_addresses_and_refreshes_after_rejection_reordering():
    state = builder()
    first = state.build(common(), 2, 3, False, retain_for_graph=True)
    pointers = tuple(
        getattr(first, name).data_ptr() for name in ("query", "schedule", "cache_lens", "cu", "block_table")
    )
    old = first.schedule.clone()
    second = state.build(common((6, 13), (0, 1, 3)), 2, 3, False)
    assert second is first
    assert pointers == tuple(
        getattr(second, name).data_ptr() for name in ("query", "schedule", "cache_lens", "cu", "block_table")
    )
    assert second.cache_lens.tolist() == [6, 13, 0, 0, 0]
    assert second.used_q.tolist() == [1, 2, 0, 0, 0]
    assert not torch.equal(old, second.schedule)
    second.query.zero_()
    cache = torch.randn(2, 128, 1, 576).to(torch.bfloat16)
    out, lse = second.adapter.attention(
        second.query,
        cache,
        block_table=second.block_table,
        cache_seqlens=second.cache_lens,
        cu_seqlens_q=second.cu,
        seqused_q=second.used_q,
        metadata=second.schedule,
    )
    assert torch.equal(out[:, 3:], torch.zeros_like(out[:, 3:]))
    assert torch.isneginf(lse[:, 3:]).all()
    with pytest.raises(ValueError, match="decode-only"):
        state.build(common(), 2, 3, True, retain_for_graph=True)


@pytest.mark.parametrize("world", [1, 2, 8])
def test_dcp_local_lengths_and_lse_merge_match_full_attention(world):
    generator = torch.Generator().manual_seed(100)
    q = torch.randn(2, 2, 576, generator=generator).to(torch.bfloat16) * 0.1
    global_kv = torch.randn(13, 576, generator=generator).to(torch.bfloat16) * 0.1
    outputs, statistics = [], []
    for rank in range(world):
        indices = [i for i in range(13) if (i // 2) % world == rank]
        cache = torch.zeros(1, 128, 1, 576, dtype=torch.bfloat16)
        cache[0, : len(indices), 0] = global_kv[indices]
        data = list(case(heads=2, q_lengths=(2,), kv_lengths=(len(indices),)))
        data[1], data[2], data[3] = q, cache, torch.zeros(1, 1, dtype=torch.int32)
        output, lse = invoke(data)
        outputs.append(output.float())
        statistics.append(lse)
        state = builder(world, rank)
        meta = state.build(common((13, 0)), 2, 3, False)
        assert int(meta.cache_lens[0]) == len(indices)
    lse = torch.stack(statistics)
    maximum = lse.amax(0)
    weights = (lse - maximum).exp()
    merged = (torch.stack(outputs) * weights[..., None]).sum(0) / weights.sum(0)[..., None]
    full = torch.zeros(1, 128, 1, 576, dtype=torch.bfloat16)
    full[0, :13, 0] = global_kv
    expected, _ = reference(
        q,
        full,
        torch.zeros(1, 1, dtype=torch.int32),
        torch.tensor([13]),
        torch.tensor([0, 2]),
        torch.tensor([2]),
        192**-0.5,
        False,
    )
    torch.testing.assert_close(merged, expected.float(), atol=0.001, rtol=0.015)


def selection_impl():
    return SimpleNamespace(
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        use_pcp=False,
        vllm_config=SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=SimpleNamespace(method="dspark", enforce_eager=True),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="kimi_k31"), max_model_len=16384, enforce_eager=True
            ),
            cache_config=SimpleNamespace(block_size=128),
            parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        ),
    )


@pytest.mark.parametrize(
    "mode,is_a3,is_draft,expected",
    [
        ("disabled", True, True, False),
        ("mock", False, True, False),
        ("external", False, True, False),
        ("mock", True, False, False),
        ("mock", True, True, True),
        ("external", True, True, True),
    ],
)
def test_selection_is_explicit_and_a5_unchanged(mode, is_a3, is_draft, expected):
    assert (
        sink.validate_sink_selection(selection_impl(), mode=mode, is_a3=is_a3, is_draft=is_draft, allow_graph=False)
        is expected
    )


@pytest.mark.parametrize("bad", ["length", "page", "quant", "nz", "pcp", "pp", "graph", "dimensions", "method"])
def test_selection_rejects_unsupported_shapes_and_features(bad):
    impl = selection_impl()
    if bad == "length":
        impl.vllm_config.model_config.max_model_len = 131072
    elif bad == "page":
        impl.vllm_config.cache_config.block_size = 192
    elif bad == "quant":
        impl.fa_quant_layer = True
    elif bad == "nz":
        impl.enable_kv_nz = True
    elif bad == "pcp":
        impl.use_pcp = True
    elif bad == "pp":
        impl.vllm_config.parallel_config.pipeline_parallel_size = 2
    elif bad == "graph":
        impl.vllm_config.model_config.enforce_eager = False
        impl.vllm_config.speculative_config.enforce_eager = False
    elif bad == "dimensions":
        impl.qk_rope_head_dim = 0
    else:
        impl.vllm_config.speculative_config.method = "eagle"
    with pytest.raises(ValueError):
        sink.validate_sink_selection(impl, mode="mock", is_a3=True, is_draft=True, allow_graph=False)


def runner(ready=True):
    backend = type("UnknownBackend", (), {})
    provider = SimpleNamespace(supports_spec_device_lengths=lambda: ready)
    group = SimpleNamespace(layer_names=("target.0",), get_metadata_builder=lambda index: provider)
    state = SimpleNamespace(supports_spec_device_lengths=lambda: True)
    speculator = SimpleNamespace(
        use_flashmla=False, use_a3_mla_sink=True, draft_attn_layer_names={"draft.0"}, attn_backends={"draft.0": backend}
    )
    spec = SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(architectures=["K3DSparkModel"]),
        enable_adaptive_verification=False,
    )
    instance = SimpleNamespace(
        sync_spec_pp_cpu_counts=False,
        model_state=state,
        attn_groups=[[group]],
        speculator=speculator,
        speculative_config=spec,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), architecture="KimiK31ForConditionalGeneration"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=8
        ),
        ascend_config=SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False)),
        vllm_config=SimpleNamespace(kv_transfer_config=None),
    )
    scope = dict(
        torch=torch,
        is_deepseek_v41=lambda cfg: False,
        AscendFlashAttentionBackend=backend,
        a3_target_device_lengths_ready=sink.a3_target_device_lengths_ready,
    )
    path = "vllm_ascend/worker/v2/model_runner.py"
    fn = ast_method(path, "NPUModelRunner", "_can_skip_spec_cpu_length_sync", scope)
    instance._can_skip_spec_cpu_length_sync = lambda: fn(instance)
    return instance, scope


@pytest.mark.parametrize("ready", [False, True])
def test_real_runner_copy_and_wait_share_the_same_predicate(monkeypatch, ready):
    instance, scope = runner(ready)
    assert instance._can_skip_spec_cpu_length_sync() is ready
    instance.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([98, 195])),
        num_computed_tokens_cpu=torch.tensor([100, 200]),
        req_id_to_index={"a": 0, "b": 1},
    )
    instance.num_computed_tokens_cpu = torch.full((2,), -99)
    instance.num_computed_tokens_stream = Mock()
    instance.num_computed_tokens_event = Mock()
    instance.input_buffers = SimpleNamespace(seq_lens_cpu=torch.zeros(2, dtype=torch.int32))
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"a": 4, "b": 4}, scheduled_cached_reqs=SimpleNamespace(req_ids=["b", "a"])
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(stream=lambda stream: nullcontext()), raising=False)
    path = "vllm_ascend/worker/v2/model_runner.py"
    copy = ast_method(path, "NPUModelRunner", "_copy_num_computed_tokens_to_cpu", scope)
    update = ast_method(path, "NPUModelRunner", "_update_seq_lens_cpu", scope)
    copy(instance)
    update(instance, scheduler, ["b", "a"])
    assert instance.num_computed_tokens_event.synchronize.call_count == int(not ready)
    assert instance.num_computed_tokens_event.record.call_count == int(not ready)
    assert instance.num_computed_tokens_cpu.tolist() == ([-99, -99] if ready else [98, 195])
    assert instance.input_buffers.seq_lens_cpu.tolist() == ([204, 104] if ready else [199, 102])


@pytest.mark.parametrize(
    "reason", ["unknown", "no_groups", "only_draft", "pp", "pcp", "offload", "adaptive", "pd", "fia"]
)
def test_sync_fallback_for_every_unsupported_consumer(reason):
    instance, _ = runner()
    if reason == "unknown":
        instance.attn_groups[0][0].get_metadata_builder = lambda index: object()
    elif reason == "no_groups":
        instance.attn_groups = []
    elif reason == "only_draft":
        instance.attn_groups[0][0].layer_names = ("draft.0",)
    elif reason == "pp":
        instance.parallel_config.pipeline_parallel_size = 2
    elif reason == "pcp":
        instance.parallel_config.prefill_context_parallel_size = 2
    elif reason == "offload":
        instance.ascend_config.sparse_kv_offload_config.enabled = True
    elif reason == "adaptive":
        instance.speculative_config.enable_adaptive_verification = True
    elif reason == "pd":
        instance.vllm_config.kv_transfer_config = object()
    else:
        instance.speculator.use_a3_mla_sink = False
    assert instance._can_skip_spec_cpu_length_sync() is False


@pytest.mark.parametrize(
    "path,owner,field,ready,blocked",
    [
        ("attention/sfa_v1.py", "AscendSFAMetadataBuilder", "use_pcp", False, True),
        ("attention/context_parallel/sfa_cp.py", "AscendSFADCPMetadataBuilder", "use_pcp", False, True),
        ("attention/indexer.py", "AscendSFAIndexerMetadataBuilder", "use_pcp", False, True),
        ("ops/kda_pcp.py", "KimiKDAHostMetadataBuilder", "pcp_size", 1, 2),
        ("worker/v2/model_states/mamba_hybrid.py", "AscendMambaHybridModelState", "recoverssm", None, object()),
    ],
)
def test_reviewed_consumers_require_exact_type_and_supported_mode(path, owner, field, ready, blocked):
    cls = type(owner, (), {})
    method = ast_method("vllm_ascend/" + path, owner, "supports_spec_device_lengths", {owner: cls})
    instance = cls()
    setattr(instance, field, ready)
    assert method(instance) is True
    setattr(instance, field, blocked)
    assert method(instance) is False
    unknown = type("UnreviewedSubclass", (cls,), {})()
    setattr(unknown, field, ready)
    assert method(unknown) is False


def test_unknown_model_state_keeps_cpu_sync():
    instance, _ = runner()
    instance.model_state = SimpleNamespace()
    assert instance._can_skip_spec_cpu_length_sync() is False


@pytest.mark.parametrize("has_device_metadata", [False, True])
def test_a5_sync_policy_retains_original_target_producer_requirement(has_device_metadata):
    instance, _ = runner(False)
    instance.speculator.use_a3_mla_sink = False
    instance.speculator.use_flashmla = True
    instance.model_state = SimpleNamespace(device_metadata=object() if has_device_metadata else None)
    # A5 must not inspect the A3 consumer declarations or require A3 groups.
    del instance.attn_groups
    assert instance._can_skip_spec_cpu_length_sync() is has_device_metadata


@pytest.mark.parametrize("host_mirror", [None, 9999])
def test_sfa_build_consumes_device_lengths_without_exact_cpu_mirror(monkeypatch, host_mirror):
    cls = type("AscendSFAMetadataBuilder", (), {})
    scope = {
        "AscendSFAMetadataBuilder": cls,
        "get_cos_and_sin_mla": lambda *args, **kwargs: (torch.zeros(3, 64), torch.zeros(3, 64)),
        "split_decodes_and_prefills": lambda *args, **kwargs: (2, 0, 3, 0),
    }
    cls.supports_spec_device_lengths = ast_method(
        "vllm_ascend/attention/sfa_v1.py",
        cls.__name__,
        "supports_spec_device_lengths",
        scope,
    )
    build = ast_method("vllm_ascend/attention/sfa_v1.py", cls.__name__, "_build", scope)
    instance = cls()
    instance.use_pcp = instance.nope = instance.use_rope = False
    instance.allow_a3_device_lengths = True
    instance.kernel_block_size = 128
    instance.decode_threshold = 8
    instance.metadata_cls = SimpleNamespace
    instance.model_config = SimpleNamespace(get_head_size=lambda: 576)
    instance.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda *args: None)
    inputs = common(capacity=3)
    inputs.num_reqs = 2
    inputs._seq_lens_cpu = None if host_mirror is None else torch.full((2,), host_mirror)
    inputs.seq_lens_cpu = None
    inputs.attn_state = object()
    inputs.max_query_len = 2
    inputs.max_seq_len = 16384
    original_to = torch.Tensor.to

    def no_cpu_copy(tensor, *args, **kwargs):
        if args and str(args[0]) == "cpu" or str(kwargs.get("device")) == "cpu":
            pytest.fail("SFA rebuilt an exact CPU sequence-length mirror")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", no_cpu_copy)
    actual = build(instance, inputs)
    assert actual.seq_lens.tolist() == [15, 7]
    assert actual.cum_query_lens.tolist() == [2, 3]
    assert actual.seq_lens.data_ptr() == inputs.seq_lens.data_ptr()
    if host_mirror is None:
        assert actual.seq_lens_cpu is None
        # The default disabled configuration retains the pre-existing fallback.
        instance.allow_a3_device_lengths = False
        monkeypatch.setattr(torch.Tensor, "to", original_to)
        fallback = build(instance, inputs)
        assert fallback.seq_lens_cpu.tolist() == [15, 7]
    else:
        assert actual.seq_lens_cpu.tolist() == [9999, 9999]


@pytest.mark.parametrize("dcp", [False, True])
def test_dspark_does_not_read_host_lengths_when_a3_selected(dcp):
    method = ast_method(
        "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py",
        "AscendDSparkSpeculator",
        "_prepare_draft_dcp_metadata_inputs",
        {"torch": torch},
    )
    # No input buffers or DCP manager are provided: touching either fails.
    instance = SimpleNamespace(use_a3_mla_sink=True, use_dcp=dcp)
    lengths, prefilling = method(instance, 2, 4, 8)
    assert lengths is None
    assert prefilling.tolist() == [False] * 4
    # Keep the old FIA path's exact-length copy and the A5 no-copy behavior.
    instance.use_a3_mla_sink = False
    instance.use_dcp = False
    instance.use_flashmla = False
    instance.attn_architecture = "MLA"
    instance.input_buffers = SimpleNamespace(seq_lens=torch.tensor([17, 33, 9999, 9999]))
    lengths, _ = method(instance, 2, 4, 8)
    assert lengths.tolist() == [17, 33, 0, 0]
    instance.use_flashmla = True
    lengths, _ = method(instance, 2, 4, 8)
    assert lengths is None


def test_real_forward_packs_split_query_and_cache_then_calls_v_up():
    data = case()
    adapter, query, cache, table, lengths, cu, used = data
    flash = SimpleNamespace(
        query=torch.empty_like(query),
        adapter=adapter,
        block_table=table,
        cache_lens=lengths,
        cu=cu,
        used_q=used,
        schedule=adapter.build_metadata(lengths, cu, used),
        attn_mask=None,
    )
    forward = ast_method(
        "vllm_ascend/attention/mla_v1.py",
        "AscendMLAImpl",
        "_forward_external_flashmla",
        {"torch": torch, "record_attention_compute_start": Mock()},
    )
    impl = SimpleNamespace(use_a3_mla_sink=True, kv_lora_rank=512, _v_up_proj=Mock(side_effect=lambda x: x))
    preprocessed = SimpleNamespace(ql_nope=query[..., :512], q_pe=query[..., 512:])
    metadata = SimpleNamespace(external_flashmla=flash)
    split_cache = (cache[..., :512], cache[..., 512:])
    result = forward(impl, preprocessed, split_cache, metadata)
    expected, _ = reference(query, cache, table, lengths, cu, used, adapter.config.softmax_scale, False)
    torch.testing.assert_close(result, expected, atol=0.002, rtol=0.01)
    impl._v_up_proj.assert_called_once()
    torch.testing.assert_close(flash.query, query, atol=0, rtol=0)
    impl.use_a3_mla_sink = False
    with pytest.raises(RuntimeError, match="original fused"):
        forward(impl, preprocessed, split_cache, metadata)


@pytest.mark.parametrize("qlens,kvlens", [((), ()), ((0,), (0,)), ((0, 0), (0, 17))])
def test_empty_batch_and_empty_queries(qlens, kvlens):
    output, lse = invoke(case(q_lengths=qlens, kv_lengths=kvlens))
    assert output.shape == (2, 0, 512)
    assert lse.shape == (2, 0)


def test_capacity_boundary_and_mock_workload_limit():
    adapter = sink.A3MLAAdapter.load(sink.A3MLAConfig(1, 0.1), "mock").for_capacity(128, 1)
    lengths = torch.tensor([16384], dtype=torch.int32)
    cu = torch.tensor([0, 1], dtype=torch.int32)
    adapter.build_metadata(lengths, cu)
    with pytest.raises(ValueError, match="16384"):
        adapter.for_capacity(129, 1)
    data = list(case(heads=1, q_lengths=(33,), kv_lengths=(1025,)))
    with pytest.raises(RuntimeError, match="workload limit"):
        invoke(data)


def test_manager_block_1536_uses_existing_128_token_component_views():
    impl = selection_impl()
    impl.vllm_config.cache_config.block_size = 1536
    assert sink.validate_sink_selection(impl, mode="mock", is_a3=True, is_draft=True, allow_graph=False)
    scope = {"torch": torch, "math": math, "get_dtype_size": lambda dtype: torch.empty(0, dtype=dtype).element_size()}
    ast_symbols("vllm_ascend/worker/v2/attn_utils.py", ["_reshape_combined_mla_kv_cache"], scope)
    data = case(q_lengths=(2,), kv_lengths=(1500,))
    adapter, query, fused, table, lengths, cu, used = data
    page_bytes = 128 * 576 * 2 + 64  # Hybrid-pool padding is not KV data.
    raw = torch.full((12 * page_bytes,), 91, dtype=torch.uint8)
    content, rope = scope["_reshape_combined_mla_kv_cache"](
        raw,
        (12, 128, 1, 512),
        (12, 128, 1, 64),
        torch.bfloat16,
        12 * page_bytes,
        12,
    )
    content.copy_(fused[..., :512])
    rope.copy_(fused[..., 512:])
    before = raw.clone()
    actual = adapter.attention(
        query,
        (content, rope),
        block_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
        seqused_q=used,
        metadata=adapter.build_metadata(lengths, cu, used),
    )
    expected = reference(query, fused, table, lengths, cu, used, adapter.config.softmax_scale, False)
    torch.testing.assert_close(actual[0], expected[0], atol=0.002, rtol=0.01)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-5, rtol=2e-5)
    assert torch.equal(raw, before)
    assert torch.all(raw.view(12, page_bytes)[:, -64:] == 91)
