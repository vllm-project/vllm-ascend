# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU selection, cache layout and host-sync policy checks.

These checks execute framework code without calling an attention kernel.
Native numerical and graph coverage lives in test_a3_mla_sink_native.py.
"""

import ast
import importlib.util
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sink = load_source("vllm_ascend/attention/a3_mla_sink.py", "a3_mla_cpu_contract")


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


supports_component_major_mla_pd = ast_symbols(
    "vllm_ascend/core/kv_cache_interface.py",
    {"supports_component_major_mla_pd"},
    {"VllmConfig": object, "AscendDeviceType": SimpleNamespace(A3="a3"), "get_ascend_device_type": lambda: "a3"},
)["supports_component_major_mla_pd"]


@pytest.mark.parametrize("connector,expected", [("MooncakeConnectorV2", True), ("MultiConnector", False)])
def test_a5_preserves_baseline_pd_layout(monkeypatch, connector, expected):
    monkeypatch.setitem(supports_component_major_mla_pd.__globals__, "get_ascend_device_type", lambda: "a5")
    config = selection_impl().vllm_config
    config.kv_transfer_config = SimpleNamespace(
        kv_connector=connector,
        kv_connector_extra_config={"use_layerwise": True, "connectors": [{"kv_connector": "MooncakeConnectorV2"}]},
    )
    assert supports_component_major_mla_pd(config) is expected


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("outside_scope", ["model", "runner", "draft", "pcp", "pp"])
def test_other_a3_deployments_keep_baseline_pd_policy(role, outside_scope):
    config = selection_impl().vllm_config
    if outside_scope == "model":
        config.model_config.hf_config.model_type = "other"
    elif outside_scope == "runner":
        config.use_v2_model_runner = False
    elif outside_scope == "draft":
        config.speculative_config = None
    elif outside_scope == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    else:
        config.parallel_config.pipeline_parallel_size = 2
    child = {"kv_connector": "MooncakeConnectorV2", "kv_connector_extra_config": {"use_layerwise": True}}
    config.kv_transfer_config = SimpleNamespace(kv_role=role, **child)
    assert supports_component_major_mla_pd(config)
    config.kv_transfer_config = SimpleNamespace(
        kv_role=role, kv_connector="MultiConnector", kv_connector_extra_config={"connectors": [child]}
    )
    assert not supports_component_major_mla_pd(config)


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
            kv_transfer_config=None,
            parallel_config=SimpleNamespace(pipeline_parallel_size=1, prefill_context_parallel_size=1),
        ),
    )


@pytest.mark.parametrize("is_a3,is_draft", [(False, False), (False, True), (True, False), (True, True)])
def test_selection_is_hardware_and_model_scoped(is_a3, is_draft):
    assert sink.supports_a3_mla(selection_impl(), is_a3=is_a3, is_draft=is_draft) is (is_a3 and is_draft)


@pytest.mark.parametrize(
    "reason",
    [
        "model",
        "runner",
        "method",
        "no_spec",
        "page",
        "quant",
        "dtype",
        "nz",
        "pcp",
        "pp",
        "pd",
        "dimensions",
        "kv_heads",
    ],
)
def test_unsupported_config_retains_fia(reason):
    impl = selection_impl()
    cfg = impl.vllm_config
    if reason == "model":
        cfg.model_config.hf_config.model_type = "deepseek_v3"
    elif reason == "runner":
        cfg.use_v2_model_runner = False
    elif reason == "method":
        cfg.speculative_config.method = "eagle"
    elif reason == "no_spec":
        cfg.speculative_config = None
    elif reason == "page":
        cfg.cache_config.block_size = 192
    elif reason == "quant":
        impl.fa_quant_layer = True
    elif reason == "dtype":
        impl.dtype = torch.float32
    elif reason == "nz":
        impl.enable_kv_nz = True
    elif reason == "pcp":
        impl.use_pcp = True
    elif reason == "pp":
        cfg.parallel_config.pipeline_parallel_size = 2
    elif reason == "pd":
        cfg.kv_transfer_config = object()
    elif reason == "dimensions":
        impl.qk_rope_head_dim = 0
    else:
        impl.num_kv_heads = 2
    assert not sink.supports_a3_mla(impl, is_a3=True, is_draft=True)


@pytest.mark.parametrize("eager", [False, True])
def test_uses_existing_execution_config_and_page_capacity(eager):
    impl = selection_impl()
    impl.vllm_config.model_config.enforce_eager = eager
    impl.vllm_config.speculative_config.enforce_eager = eager
    impl.vllm_config.model_config.max_model_len = 131072
    impl.vllm_config.cache_config.block_size = 1536
    assert sink.supports_a3_mla(impl, is_a3=True, is_draft=True)
    adapter = sink.A3MLAAdapter(sink.A3MLAConfig(8, 192**-0.5), None, None).for_capacity(1024, 16)
    assert adapter.config.max_seqlen_k == 131072
    assert adapter.config.max_seqlen_q == 16


def test_missing_package_retains_fia(monkeypatch):
    def unavailable(name):
        assert name == sink.OPERATOR_MODULE
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(sink, "import_module", unavailable)
    assert sink.get_a3_mla_ops() is None
    with pytest.raises(ImportError, match="A3 MLA requires"):
        sink.A3MLAAdapter.load(sink.A3MLAConfig(8, 0.125))


def test_installed_package_dependency_error_is_not_swallowed(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError(name="flash_attn_3_npu")

    monkeypatch.setattr(sink, "import_module", broken)
    with pytest.raises(ModuleNotFoundError):
        sink.get_a3_mla_ops()


def test_load_binds_only_the_fixed_public_api(monkeypatch):
    def never_execute(*args, **kwargs):
        pytest.fail("CPU contract checks must not execute an attention kernel")

    module = SimpleNamespace(flash_attn_with_kvcache=never_execute, get_scheduler_metadata=never_execute)
    monkeypatch.setattr(sink, "import_module", lambda name: module)
    adapter = sink.A3MLAAdapter.load(sink.A3MLAConfig(8, 0.125))
    assert adapter.attention_op is module.flash_attn_with_kvcache
    assert adapter.metadata_op is module.get_scheduler_metadata
    del module.get_scheduler_metadata
    assert sink.get_a3_mla_ops() is None


def test_cpu_inputs_cannot_execute_native_metadata():
    adapter = sink.A3MLAAdapter(sink.A3MLAConfig(8, 0.125), None, None)
    with pytest.raises(ValueError, match="NPU"):
        adapter.build_metadata(torch.tensor([3], dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(ValueError, match="nonempty"):
        adapter.build_metadata(torch.empty(0, dtype=torch.int32), torch.zeros(1, dtype=torch.int32))


@pytest.mark.parametrize("causal", [False, True])
def test_two_operator_boundary_reuses_lengths_and_normalizes_empty_rows(monkeypatch, causal):
    # CPU ABI instrumentation only: device rejection is tested above; this does
    # not simulate or validate the native attention kernel's arithmetic.
    monkeypatch.setattr(sink.A3MLAAdapter, "_check_lengths", lambda self, lengths, cu, used: lengths.numel())
    lengths = torch.tensor([1, 0], dtype=torch.int32)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    used = torch.tensor([2, 0], dtype=torch.int32)
    calls = []
    provider_outputs = []

    def metadata_op(*args, **kwargs):
        assert args[6] is lengths
        assert kwargs["cu_seqlens_q"] is cu
        calls.append("metadata")
        return torch.zeros(32, dtype=torch.uint8)

    def attention_op(q, rope, content, **kwargs):
        assert kwargs["cache_seqlens"] is lengths
        assert kwargs["cu_seqlens_q"] is cu
        assert kwargs["scheduler_metadata"] is metadata
        assert q.shape == (4, 8, 64) and q.is_contiguous()
        assert kwargs["qv"].shape == (4, 8, 512) and kwargs["qv"].is_contiguous()
        calls.append("attention")
        output = torch.ones(4, 8, 512, dtype=q.dtype)
        lse = torch.full((8, 4), torch.inf)
        provider_outputs.append((output, lse))
        return output, lse, None, None

    adapter = sink.A3MLAAdapter(sink.A3MLAConfig(8, 0.125, mask_mode=3 if causal else 0), attention_op, metadata_op)

    def forbidden(*args, **kwargs):
        pytest.fail("the adapter must not read host values or remask normalized lengths")

    with monkeypatch.context() as context:
        for name in ("cpu", "item", "tolist"):
            context.setattr(torch.Tensor, name, forbidden)
        context.setattr(torch, "where", forbidden)
        metadata = adapter.build_metadata(lengths, cu, used)
        output_mask = adapter.build_output_mask(lengths, cu, 4)
        context.setattr(torch, "searchsorted", forbidden)
        context.setattr(torch.Tensor, "masked_fill", forbidden)
        output, lse = adapter.attention(
            torch.zeros(4, 8, 576, dtype=torch.bfloat16),
            adapter.allocate_cache(2, device="cpu"),
            block_table=torch.zeros(2, 1, dtype=torch.int32),
            cache_seqlens=lengths,
            cu_seqlens_q=cu,
            seqused_q=used,
            metadata=metadata,
            output_mask=output_mask,
        )
    assert calls == ["metadata", "attention"]
    assert output.data_ptr() == provider_outputs[0][0].data_ptr()
    assert lse.data_ptr() == provider_outputs[0][1].data_ptr()
    assert torch.all(output[:, 2:] == 0) and torch.isneginf(lse[:, 2:]).all()
    assert torch.all(output[:, 1] == 1)
    assert torch.all(output[:, 0] == (0 if causal else 1))
    if causal:
        assert torch.isneginf(lse[:, 0]).all()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [((0, 3, 0, 2, 4), (0, 1, 9, 0, 2)), ((1, 0, 3), (0, 0, 0)), ((0, 0), (0, 3)), ((2, 1), (3, 1))],
)
def test_output_mask_matches_visible_kv_rows(causal, qlens, klens):
    adapter = sink.A3MLAAdapter(sink.A3MLAConfig(8, 0.125, mask_mode=3 if causal else 0), None, None)
    cu = torch.tensor([0, *torch.tensor(qlens).cumsum(0).tolist()], dtype=torch.int32)
    expected = [
        (min(klen, max(0, klen - qlen + offset + 1)) if causal else klen) == 0
        for qlen, klen in zip(qlens, klens)
        for offset in range(qlen)
    ]
    actual = adapter.build_output_mask(torch.tensor(klens, dtype=torch.int32), cu, sum(qlens))
    assert actual.dtype == torch.bool and actual.tolist() == expected


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dcp_size", [1, 2])
def test_a3_builder_normalizes_once_and_reuses_schedule_storage(monkeypatch, causal, dcp_size):
    monkeypatch.setattr(sink.A3MLAAdapter, "_check_lengths", lambda self, lengths, cu, used: lengths.numel())
    calls = []

    def metadata_op(*args, **kwargs):
        lengths, cu = args[6], kwargs["cu_seqlens_q"]
        assert lengths.device.type == "cpu"  # A3 must not use Meta shape inference.
        calls.append((args[1], args[2], lengths, cu))
        return torch.cat((lengths, cu)).to(torch.uint8)

    names = {
        "FLASHMLA_QK_DIM",
        "FLASHMLA_V_DIM",
        "FLASHMLA_BLOCK_SIZE",
        "FlashMLACurrent",
        "FlashMLADecode",
        "FlashMLAMetadataBuilder",
    }
    scope = dict(
        torch=torch,
        dataclass=dataclass,
        replace=replace,
        FlashMLAImpl=type("UnusedA5", (), {}),
        A3MLAAdapter=sink.A3MLAAdapter,
        get_dcp_local_seq_lens=lambda lengths, **kwargs: (lengths + 1) // 2,
    )
    ast_symbols("vllm_ascend/attention/flashmla.py", names, scope)
    impl = SimpleNamespace(
        a3_mla_adapter=sink.A3MLAAdapter(sink.A3MLAConfig(8, 0.125), None, metadata_op),
        num_heads=8,
        scale=0.125,
        dtype=torch.bfloat16,
        use_mla_rope=False,
        dcp_size=dcp_size,
        dcp_rank=0,
        vllm_config=SimpleNamespace(
            speculative_config=object(),
            parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        ),
    )
    builder = scope["FlashMLAMetadataBuilder"](impl, torch.device("cpu"), 4, torch.empty(0))
    inputs = common()
    inputs.causal = causal
    first = builder.build(inputs, 2, 3, False, retain_for_graph=True)
    schedule_ptr = first.schedule.data_ptr()
    mask_ptr = first.output_mask.data_ptr()
    current_ptr = first.current.schedule.data_ptr() if first.current else None
    current_mask_ptr = first.current.output_mask.data_ptr() if first.current else None
    split = causal and dcp_size > 1
    assert len(calls) == 1 + int(split)
    assert first.cache_lens.tolist() == (
        [7, 3, 0, 0, 0] if split else [8, 4, 0, 0, 0] if dcp_size > 1 else [15, 7, 0, 0, 0]
    )
    assert all(tokens == 8 and capacity == 128 for tokens, capacity, _, _ in calls)
    assert calls[0][2] is first.cache_lens  # Keep the builder's length buffer.
    inputs.seq_lens.copy_(torch.tensor([1, 0], dtype=torch.int32))
    second = builder.build(inputs, 2, 3, False)
    assert len(calls) == 2 * (1 + int(split))
    assert second is first and second.schedule.data_ptr() == schedule_ptr
    assert second.output_mask.data_ptr() == mask_ptr
    expected_live = [False, False] if split else [not causal, True]
    assert second.output_mask.tolist() == [not live for live in expected_live] + [True] * 6
    assert second.used_q.tolist() == [2, 0, 0, 0, 0]
    assert second.cache_lens.tolist() == ([0, 0, 0, 0, 0] if split else [1, 0, 0, 0, 0])
    torch.testing.assert_close(second.schedule, torch.cat((second.cache_lens, second.cu)).to(torch.uint8))
    if split:
        assert second.current.schedule.data_ptr() == current_ptr
        assert second.current.output_mask.data_ptr() == current_mask_ptr
        assert second.current.output_mask.tolist() == [False, False] + [True] * 6
        assert isinstance(second.current.cache, tuple)
        torch.testing.assert_close(second.current.schedule, torch.cat((second.used_q, second.cu)).to(torch.uint8))

    # Reorder query boundaries and reactivate a previously empty request in
    # the same retained buffers; masks must not be stale across graph replays.
    inputs.query_start_loc.copy_(torch.tensor([0, 1, 3], dtype=torch.int32))
    inputs.seq_lens.copy_(torch.tensor([0, 4], dtype=torch.int32))
    with patch.object(torch, "searchsorted", wraps=torch.searchsorted) as lookup:
        third = builder.build(inputs, 2, 3, False)
    assert lookup.call_count == 1
    assert third.output_mask.data_ptr() == mask_ptr
    assert third.output_mask.tolist() == [True, False, False] + [True] * 5
    if split:
        assert third.current.output_mask.data_ptr() == current_mask_ptr
        assert third.current.output_mask.tolist() == third.output_mask.tolist()

        inputs.query_start_loc.copy_(torch.tensor([0, 0, 3], dtype=torch.int32))
        with_empty_request = builder.build(inputs, 2, 3, False)
        assert with_empty_request.current.slots.tolist() == [0, 1, 2] + [-1] * 5
        assert with_empty_request.current.output_mask.tolist() == [False] * 3 + [True] * 5

    for _ in range(2):
        calls.clear()
        mixed = builder.build(inputs, 2, 3, True)
        assert mixed is not first and mixed.schedule is not None
        assert len(calls) == 1 + int(split)


@pytest.mark.parametrize("path", ["ordinary", "history", "current"])
def test_a3_callers_forward_the_matching_output_mask(monkeypatch, path):
    monkeypatch.setattr(sink.A3MLAAdapter, "_check_lengths", lambda self, lengths, cu, used: lengths.numel())

    def provider(q, *args, **kwargs):
        assert q.data_ptr() == q_pe.data_ptr()
        assert kwargs["qv"].data_ptr() == q_nope.data_ptr()
        # Finite invalid-row sentinels must be masked by the eventual consumer.
        output = torch.ones(2, 8, 512, dtype=q.dtype)
        assert kwargs["return_softmax_lse"] is (path != "ordinary")
        return (output, torch.zeros(8, 2), None, None) if kwargs["return_softmax_lse"] else output

    adapter = sink.A3MLAAdapter(sink.A3MLAConfig(8, 0.125), provider, None)
    cache = adapter.allocate_cache(1, device="cpu")
    flash = SimpleNamespace(
        adapter=adapter,
        query=torch.zeros(2, 8, 576, dtype=torch.bfloat16),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        cache_lens=torch.zeros(1, dtype=torch.int32),
        cu=torch.tensor([0, 2], dtype=torch.int32),
        used_q=torch.tensor([2], dtype=torch.int32),
        schedule=torch.zeros(32, dtype=torch.uint8),
        attn_mask=None,
        output_mask=torch.ones(2, dtype=torch.bool),
    )
    flash.current = SimpleNamespace(
        adapter=adapter,
        query=flash.query.clone(),
        cache=cache,
        schedule=flash.schedule,
        block_table=flash.block_table,
        attn_mask=None,
        output_mask=torch.zeros(2, dtype=torch.bool),
    )
    q_nope, q_pe = flash.query[..., :512].contiguous(), flash.query[..., 512:].contiguous()
    if path == "ordinary":
        forward = ast_method(
            "vllm_ascend/attention/mla_v1.py",
            "AscendMLAImpl",
            "_forward_external_flashmla",
            {"torch": torch, "record_attention_compute_start": lambda: None},
        )
        impl = SimpleNamespace(kv_lora_rank=512, use_a3_mla_sink=True, _v_up_proj=lambda output: output)
        out = forward(impl, SimpleNamespace(ql_nope=q_nope, q_pe=q_pe), cache, SimpleNamespace(external_flashmla=flash))
        assert torch.all(out == 0)
    else:

        def forbidden(*args, **kwargs):
            pytest.fail("DCP must defer masking and avoid fused Query copies")

        monkeypatch.setattr(torch.Tensor, "masked_fill_", forbidden)
        monkeypatch.setattr(torch.Tensor, "copy_", forbidden)
        kinds = SimpleNamespace(HISTORY=object(), CURRENT=object())
        forward = ast_method(
            "vllm_ascend/attention/context_parallel/mla_cp.py",
            "AscendMlaDCPImpl",
            "_run_dcp_flashmla_attention",
            {"CPKVScope": kinds},
        )
        kind = kinds.CURRENT if path == "current" else kinds.HISTORY
        out, lse = forward(SimpleNamespace(kv_lora_rank=512), flash, q_nope, q_pe, cache, None, attention_kind=kind)
        assert torch.all(out == 1)
        assert torch.all(lse == 0)


def test_current_and_persistent_caches_preserve_packed_components():
    adapter = sink.A3MLAAdapter(sink.A3MLAConfig(8, 0.125), None, None)
    content, rope = adapter.allocate_cache(12, device="cpu")
    assert content.shape == (12, 128, 1, 512) and rope.shape == (12, 128, 1, 64)
    assert content.is_contiguous() and rope.is_contiguous()
    assert content.untyped_storage().data_ptr() != rope.untyped_storage().data_ptr()
    scope = {"torch": torch, "math": math, "get_dtype_size": lambda dtype: torch.empty(0, dtype=dtype).element_size()}
    ast_symbols("vllm_ascend/worker/v2/attn_utils.py", ["_reshape_combined_mla_kv_cache"], scope)
    page_bytes = 128 * 576 * 2 + 64
    raw = torch.full((12 * page_bytes,), 91, dtype=torch.uint8)
    content, rope = scope["_reshape_combined_mla_kv_cache"](
        raw, (12, 128, 1, 512), (12, 128, 1, 64), torch.bfloat16, 12 * page_bytes, 12
    )
    assert content.stride()[1:] == (512, 512, 1)
    assert rope.stride()[1:] == (64, 64, 1)
    content.fill_(1)
    rope.fill_(2)
    assert torch.all(content == 1) and torch.all(rope == 2)
    assert torch.all(raw.view(12, page_bytes)[:, -64:] == 91)


def runner(ready=True):
    backend = type("UnknownBackend", (), {})
    state = SimpleNamespace()
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
        _a3_target_device_lengths=ready,
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
    )
    path = "vllm_ascend/worker/v2/model_runner.py"
    fn = ast_method(path, "NPUModelRunner", "_can_skip_spec_cpu_length_sync", scope)
    instance._can_skip_spec_cpu_length_sync = lambda: fn(instance)
    return instance, scope


@pytest.mark.parametrize("connector", [None, "MooncakeConnectorV2", "MooncakePullConnector", "MultiConnector"])
@pytest.mark.parametrize("ready", [False, True])
def test_real_runner_copy_and_wait_share_the_same_predicate(monkeypatch, ready, connector):
    instance, scope = runner(ready)
    if connector is not None:
        instance.vllm_config.kv_transfer_config = SimpleNamespace(kv_connector=connector)
        if connector == "MultiConnector":
            instance.vllm_config.kv_transfer_config.kv_connector_extra_config = {
                "connectors": [{"kv_connector": "MooncakeConnectorV2"}]
            }
    assert instance._can_skip_spec_cpu_length_sync() is ready
    instance.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([98, 195])),
        num_computed_tokens_cpu=torch.tensor([100, 200]),
        req_id_to_index={"a": 0, "b": 1},
    )
    instance.num_computed_tokens_cpu = torch.full((2,), -99)
    events = []
    instance.num_computed_tokens_stream = SimpleNamespace(wait_stream=lambda stream: events.append("wait"))
    instance.num_computed_tokens_event = SimpleNamespace(
        record=lambda: events.append("record"), synchronize=lambda: events.append("synchronize")
    )
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
    assert events.count("synchronize") == int(not ready)
    assert events.count("record") == int(not ready)
    assert instance.num_computed_tokens_cpu.tolist() == ([-99, -99] if ready else [98, 195])
    assert instance.input_buffers.seq_lens_cpu.tolist() == ([204, 104] if ready else [199, 102])


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize(
    "connector,module,expected",
    [
        ("MooncakeConnectorV2", None, True),
        ("MooncakePullConnector", None, True),
        ("MooncakeConnectorV1", None, False),
        ("MooncakePushConnector", None, False),
        ("AscendStoreConnector", None, False),
        ("MultiConnector", None, False),
        ("MooncakeConnectorV2", "custom.connector", False),
    ],
)
def test_pd_selection_and_length_sync_share_a5_transfer_gate(role, connector, module, expected):
    transfer = SimpleNamespace(kv_connector=connector, kv_connector_module_path=module, kv_role=role)
    impl = selection_impl()
    impl.vllm_config.kv_transfer_config = transfer
    assert (
        sink.supports_a3_mla(
            impl, is_a3=True, is_draft=True, pd_supported=supports_component_major_mla_pd(impl.vllm_config)
        )
        is expected
    )
    instance, _ = runner()
    instance.vllm_config.kv_transfer_config = transfer
    instance.speculator.use_a3_mla_sink = sink.supports_a3_mla(
        impl, is_a3=True, is_draft=True, pd_supported=supports_component_major_mla_pd(impl.vllm_config)
    )
    assert instance._can_skip_spec_cpu_length_sync() is expected


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize(
    "children,module,expected",
    [
        ([{"kv_connector": "MooncakeConnectorV2"}], None, True),
        ([{"kv_connector": "MooncakePullConnector"}], None, True),
        ([{"kv_connector": "MooncakeConnectorV2"}], "custom.wrapper", False),
        ([{"kv_connector": "MooncakeConnectorV2", "kv_connector_module_path": "custom.child"}], None, False),
        ([{"kv_connector": "MooncakeConnectorV2"}, {"kv_connector": "MooncakePullConnector"}], None, False),
        ([{"kv_connector": "MooncakeConnectorV1"}], None, False),
        ([{"kv_connector": "MultiConnector"}], None, False),
        ([{}], None, False),
        ([None], None, False),
        ([], None, False),
        (None, None, False),
        ({"kv_connector": "MooncakeConnectorV2"}, None, False),
    ],
)
def test_single_child_multi_pd_selection_and_sync(role, children, module, expected):
    transfer = SimpleNamespace(
        kv_connector="MultiConnector",
        kv_connector_module_path=module,
        kv_role=role,
        kv_connector_extra_config={"connectors": children},
    )
    impl = selection_impl()
    impl.vllm_config.kv_transfer_config = transfer
    assert (
        sink.supports_a3_mla(
            impl, is_a3=True, is_draft=True, pd_supported=supports_component_major_mla_pd(impl.vllm_config)
        )
        is expected
    )
    instance, _ = runner()
    instance.vllm_config.kv_transfer_config = transfer
    instance.speculator.use_a3_mla_sink = sink.supports_a3_mla(
        impl, is_a3=True, is_draft=True, pd_supported=supports_component_major_mla_pd(impl.vllm_config)
    )
    assert instance._can_skip_spec_cpu_length_sync() is expected
    # Inspecting child capability must not replace the user's connector config.
    assert transfer.kv_connector == "MultiConnector"
    assert transfer.kv_connector_extra_config["connectors"] is children


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("layerwise", [False, True])
def test_mooncake_layerwise_retains_legacy_layout_and_sync(wrapped, layerwise):
    child = {"kv_connector": "MooncakeConnectorV2", "kv_connector_extra_config": {"use_layerwise": layerwise}}
    transfer = SimpleNamespace(
        **(
            {"kv_connector": "MultiConnector", "kv_connector_extra_config": {"connectors": [child]}}
            if wrapped
            else child
        )
    )
    impl = selection_impl()
    impl.vllm_config.kv_transfer_config = transfer
    assert sink.supports_a3_mla(
        impl, is_a3=True, is_draft=True, pd_supported=supports_component_major_mla_pd(impl.vllm_config)
    ) is (not layerwise)
    instance, _ = runner()
    instance.vllm_config.kv_transfer_config = transfer
    instance.speculator.use_a3_mla_sink = sink.supports_a3_mla(
        impl, is_a3=True, is_draft=True, pd_supported=supports_component_major_mla_pd(impl.vllm_config)
    )
    assert instance._can_skip_spec_cpu_length_sync() is (not layerwise)


@pytest.mark.parametrize("reason", ["pp", "pcp", "offload", "adaptive", "fia"])
def test_sync_fallback_for_every_unsupported_consumer(reason):
    instance, _ = runner()
    if reason == "pp":
        instance.parallel_config.pipeline_parallel_size = 2
    elif reason == "pcp":
        instance.parallel_config.prefill_context_parallel_size = 2
    elif reason == "offload":
        instance.ascend_config.sparse_kv_offload_config.enabled = True
    elif reason == "adaptive":
        instance.speculative_config.enable_adaptive_verification = True
    else:
        instance.speculator.use_a3_mla_sink = False
    assert instance._can_skip_spec_cpu_length_sync() is False


@pytest.mark.parametrize("case", ["supported", "unknown", "empty", "draft_only", "state", "recover", "pcp"])
def test_bind_target_lengths_once(case, monkeypatch):
    from types import ModuleType

    modules = {}
    for path, name in (
        ("attention.context_parallel.sfa_cp", "AscendSFADCPMetadataBuilder"),
        ("attention.sfa_v1", "AscendSFAMetadataBuilder"),
        ("attention.indexer", "AscendSFAIndexerMetadataBuilder"),
        ("ops.kda_pcp", "KimiKDAHostMetadataBuilder"),
        ("worker.v2.model_states.mamba_hybrid", "AscendMambaHybridModelState"),
    ):
        module = ModuleType("vllm_ascend." + path)
        modules[name] = type(name, (), {})
        setattr(module, name, modules[name])
        monkeypatch.setitem(sys.modules, module.__name__, module)
    state = modules["AscendMambaHybridModelState"]()
    state.recoverssm = object() if case == "recover" else None
    builders = [modules[name]() for name in modules if name.endswith("Builder")]
    if case == "unknown":
        builders.append(object())
    group = SimpleNamespace(layer_names=["draft" if case == "draft_only" else "target"], metadata_builders=builders)
    runner = SimpleNamespace(
        model_state=object() if case == "state" else state,
        attn_groups=[] if case == "empty" else [[group]],
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2 if case == "pcp" else 1),
    )
    configure = ast_method(
        "vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner", "_configure_a3_target_lengths", {}
    )
    assert configure(runner, {"draft"}) is (case == "supported")
    for sfa in builders[:2]:
        assert getattr(sfa, "allow_a3_device_lengths", False) is (case == "supported")


@pytest.mark.parametrize("has_device_metadata", [False, True])
def test_a5_sync_policy_retains_original_target_producer_requirement(has_device_metadata):
    instance, _ = runner(False)
    instance.speculator.use_a3_mla_sink = False
    instance.speculator.use_flashmla = True
    instance.model_state = SimpleNamespace(device_metadata=object() if has_device_metadata else None)
    # A5 must not inspect the A3 consumer declarations or require A3 groups.
    assert instance._can_skip_spec_cpu_length_sync() is has_device_metadata


@pytest.mark.parametrize("host_mirror", [None, 9999])
def test_sfa_build_consumes_device_lengths_without_exact_cpu_mirror(monkeypatch, host_mirror):
    cls = type("AscendSFAMetadataBuilder", (), {})
    scope = {
        "AscendSFAMetadataBuilder": cls,
        "get_cos_and_sin_mla": lambda *args, **kwargs: (torch.zeros(3, 64), torch.zeros(3, 64)),
        "split_decodes_and_prefills": lambda *args, **kwargs: (2, 0, 3, 0),
    }
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
        # Other model/hardware paths retain the pre-existing fallback.
        del instance.allow_a3_device_lengths
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
