# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU selection, cache layout and draft attention checks.

These checks execute framework code without calling an attention kernel.
"""

import ast
import importlib.util
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
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
    {"VllmConfig": object},
)["supports_component_major_mla_pd"]

use_component_major_mla_pd_layout = ast_symbols(
    "vllm_ascend/attention/utils.py",
    {"use_component_major_mla_pd_layout"},
    {
        "VllmConfig": object,
        "AscendDeviceType": SimpleNamespace(A3="a3"),
        "get_ascend_device_type": lambda: "a3",
        "supports_component_major_mla_pd": supports_component_major_mla_pd,
    },
)["use_component_major_mla_pd_layout"]


@pytest.mark.parametrize("connector,expected", [("MooncakeConnectorV2", False), ("MultiConnector", False)])
def test_a5_preserves_baseline_pd_layout(monkeypatch, connector, expected):
    monkeypatch.setitem(use_component_major_mla_pd_layout.__globals__, "get_ascend_device_type", lambda: "a5")
    config = selection_impl().vllm_config
    config.kv_transfer_config = SimpleNamespace(
        kv_connector=connector,
        kv_connector_extra_config={"use_layerwise": True, "connectors": [{"kv_connector": "MooncakeConnectorV2"}]},
    )
    assert use_component_major_mla_pd_layout(config) is expected


@pytest.mark.parametrize("outside_scope", ["model", "no_draft", "draft_method", "pcp", "pp"])
def test_multi_connector_rejects_unsupported_a3_deployment(outside_scope):
    config = selection_impl().vllm_config
    child = {"kv_connector": "MooncakeConnectorV2", "kv_connector_extra_config": {"use_layerwise": False}}
    config.kv_transfer_config = SimpleNamespace(
        kv_role="kv_consumer", kv_connector="MultiConnector", kv_connector_extra_config={"connectors": [child]}
    )
    assert use_component_major_mla_pd_layout(config)

    if outside_scope == "model":
        config.model_config.hf_config.model_type = "other"
    elif outside_scope == "no_draft":
        config.speculative_config = None
    elif outside_scope == "draft_method":
        config.speculative_config.method = "mtp"
    elif outside_scope == "pcp":
        config.parallel_config.prefill_context_parallel_size = 2
    else:
        config.parallel_config.pipeline_parallel_size = 2
    assert supports_component_major_mla_pd(config)  # deployment changes do not change the connector ABI
    assert not use_component_major_mla_pd_layout(config)


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
        use_dcp=False,
        use_pcp=False,
        vllm_config=SimpleNamespace(
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
    assert sink.supports_a3_mla(
        selection_impl(), is_a3=is_a3, is_draft=is_draft, supports_pd_layout=use_component_major_mla_pd_layout
    ) is (is_a3 and is_draft)


@pytest.mark.parametrize("version, eligible", [("0.5.0", False), ("0.5.1.post1", True)])
def test_dflash_window_requires_compatible_provider(monkeypatch, version, eligible):
    impl = selection_impl()
    impl.vllm_config.speculative_config.method = "dflash"
    monkeypatch.setattr(sink, "version", lambda name: version)
    assert (
        sink.supports_a3_mla(
            impl, is_a3=True, is_draft=True, dflash_draft_window=True, supports_pd_layout=lambda _: True
        )
        is eligible
    )
    assert not sink.supports_a3_mla(impl, is_a3=True, is_draft=True, supports_pd_layout=lambda _: True)


def test_multi_rank_selection_requires_split_dcp_implementation():
    impl = selection_impl()
    impl.use_dcp = True
    assert not sink.supports_a3_mla(
        impl, is_a3=True, is_draft=True, supports_pd_layout=use_component_major_mla_pd_layout
    )
    impl.supports_split_query_dcp = True
    assert sink.supports_a3_mla(impl, is_a3=True, is_draft=True, supports_pd_layout=use_component_major_mla_pd_layout)


@pytest.mark.parametrize(
    "reason",
    [
        "model",
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
    assert not sink.supports_a3_mla(
        impl, is_a3=True, is_draft=True, supports_pd_layout=use_component_major_mla_pd_layout
    )


@pytest.mark.parametrize("eager", [False, True])
def test_uses_existing_execution_config_and_page_capacity(eager):
    impl = selection_impl()
    impl.vllm_config.model_config.enforce_eager = eager
    impl.vllm_config.speculative_config.enforce_eager = eager
    impl.vllm_config.model_config.max_model_len = 131072
    impl.vllm_config.cache_config.block_size = 1536
    assert sink.supports_a3_mla(impl, is_a3=True, is_draft=True, supports_pd_layout=use_component_major_mla_pd_layout)
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


@pytest.mark.parametrize("name", ["get_scheduler_metadata", "flash_attn_with_kvcache"])
def test_unavailable_entry_point_retains_fia(monkeypatch, name):
    module = SimpleNamespace(get_scheduler_metadata=lambda: None, flash_attn_with_kvcache=lambda: None)
    monkeypatch.setattr(sink, "import_module", lambda _: module)
    delattr(module, name)
    assert sink.get_a3_mla_ops() is None
    setattr(module, name, None)
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
    provider_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

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
        "FLASHMLA_NON_CAUSAL_MASK_MODE",
        "FLASHMLA_CAUSAL_MASK_MODE",
        "FLASHMLA_BAND_MASK_MODE",
    }
    scope = dict(
        torch=torch,
        dataclass=dataclass,
        replace=replace,
        Any=Any,
        Iterator=Iterator,
        contextmanager=contextmanager,
        FlashMLAImpl=type("UnusedA5", (), {}),
        QuantFlashMLAImpl=type("UnusedQuant", (), {}),
        A3MLAAdapter=sink.A3MLAAdapter,
        get_dcp_local_seq_lens=lambda lengths, **kwargs: (lengths + 1) // 2,
    )
    ast_symbols("vllm_ascend/attention/flashmla.py", names, scope)
    if dcp_size > 1 and not hasattr(scope["FlashMLACurrent"], "allocate_split"):
        pytest.skip("requires split-query DCP support")
    impl = SimpleNamespace(
        use_quant_flashmla=False,
        a3_mla_adapter=sink.A3MLAAdapter(sink.A3MLAConfig(8, 0.125), None, metadata_op),
        num_heads=8,
        scale=0.125,
        dtype=torch.bfloat16,
        use_mla_rope=False,
        dcp_size=dcp_size,
        pcp_group=SimpleNamespace(world_size=1),
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
        query=None,
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
        query=None,
        cache=cache,
        schedule=flash.schedule,
        block_table=flash.block_table,
        attn_mask=None,
        output_mask=torch.zeros(2, dtype=torch.bool),
    )
    q_nope = torch.zeros(2, 8, 512, dtype=torch.bfloat16)
    q_pe = torch.zeros(2, 8, 64, dtype=torch.bfloat16)
    if path == "ordinary":
        forward = ast_method(
            "vllm_ascend/attention/mla_v1.py",
            "AscendMLAImpl",
            "_forward_external_flashmla",
            {"torch": torch, "record_attention_compute_start": lambda: None},
        )
        impl = SimpleNamespace(
            kv_lora_rank=512, use_a3_mla_sink=True, use_quant_flashmla=False, _v_up_proj=lambda output: output
        )
        out = forward(impl, SimpleNamespace(ql_nope=q_nope, q_pe=q_pe), cache, SimpleNamespace(external_flashmla=flash))
        assert torch.all(out == 0)
    else:
        current = ast_symbols(
            "vllm_ascend/attention/flashmla.py", {"FlashMLACurrent"}, dict(dataclass=dataclass, Any=Any, torch=torch)
        )["FlashMLACurrent"]
        if not hasattr(current, "allocate_split"):
            pytest.skip("requires split-query DCP support")

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
        out, lse = forward(
            SimpleNamespace(kv_lora_rank=512, use_quant_flashmla=False),
            flash,
            q_nope,
            q_pe,
            cache,
            None,
            attention_kind=kind,
        )
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
def test_pd_selection_uses_transfer_gate(role, connector, module, expected):
    transfer = SimpleNamespace(kv_connector=connector, kv_connector_module_path=module, kv_role=role)
    impl = selection_impl()
    impl.vllm_config.kv_transfer_config = transfer
    assert (
        sink.supports_a3_mla(impl, is_a3=True, is_draft=True, supports_pd_layout=use_component_major_mla_pd_layout)
        is expected
    )


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
def test_single_child_multi_pd_selection(role, children, module, expected):
    transfer = SimpleNamespace(
        kv_connector="MultiConnector",
        kv_connector_module_path=module,
        kv_role=role,
        kv_connector_extra_config={"connectors": children},
    )
    impl = selection_impl()
    impl.vllm_config.kv_transfer_config = transfer
    assert (
        sink.supports_a3_mla(impl, is_a3=True, is_draft=True, supports_pd_layout=use_component_major_mla_pd_layout)
        is expected
    )
    # Inspecting child capability must not replace the user's connector config.
    assert transfer.kv_connector == "MultiConnector"
    assert transfer.kv_connector_extra_config["connectors"] is children


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("layerwise", [False, True])
def test_mooncake_layerwise_retains_legacy_layout(wrapped, layerwise):
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
    assert supports_component_major_mla_pd(impl.vllm_config) is (not layerwise)
    assert sink.supports_a3_mla(
        impl, is_a3=True, is_draft=True, supports_pd_layout=use_component_major_mla_pd_layout
    ) is (not layerwise)


@pytest.mark.parametrize("dcp", [False, True])
def test_dspark_does_not_read_host_lengths_when_a3_selected(dcp):
    method = ast_method(
        "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py",
        "AscendDSparkSpeculator",
        "_prepare_draft_dcp_metadata_inputs",
        ast_symbols(
            "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py",
            {"prepare_draft_fia_lengths"},
            {"torch": torch, "Any": Any},
        ),
    )
    # No input buffers or DCP manager are provided: touching either fails.
    instance = SimpleNamespace(use_a3_mla_sink=True, use_flashmla=False, attn_architecture="MLA", use_dcp=dcp)
    lengths, prefilling = method(instance, 2, 4, 8)
    assert lengths is None
    assert prefilling.tolist() == [False] * 4
    # Keep the old FIA path's exact-length copy and the A5 no-copy behavior.
    instance.use_a3_mla_sink = False
    instance.use_dcp = False
    instance._proposal_host_lengths = None
    instance.input_buffers = SimpleNamespace(seq_lens=torch.tensor([17, 33, 9999, 9999]))
    lengths, _ = method(instance, 2, 4, 8)
    assert lengths.tolist() == [17, 33, 0, 0]
    instance.use_flashmla = True
    lengths, _ = method(instance, 2, 4, 8)
    assert lengths is None
