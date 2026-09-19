# SPDX-License-Identifier: Apache-2.0
"""Check MRv2 draft config isolation and attention context on CPU."""

import ast
import copy
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch


def mixin_node():
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    tree = ast.parse((root / "worker/v2/spec_decode/dcp_utils.py").read_text())
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DCPDraftReplicatedMixin")


def test_mrv2_pd_config_isolates_recompute_and_preserves_target():
    from dataclasses import dataclass, replace

    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    helper_tree = ast.parse((root / "worker/v2/spec_decode/dcp_utils.py").read_text())
    helpers = [node for node in helper_tree.body if getattr(node, "name", None) == "draft_additional_config"]
    tree = ast.parse((root / "worker/v2/spec_decode/dspark/speculator.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [node for node in cls.body if getattr(node, "name", None) == "__init__"]

    @dataclass
    class Config:
        parallel_config: SimpleNamespace
        cache_config: SimpleNamespace
        kv_transfer_config: object
        additional_config: dict[str, Any]

        def __post_init__(self):
            if self.kv_transfer_config is None:
                assert not self.additional_config["scheduler_config"]["recompute_scheduler_enable"]

    class Parent:
        def __init__(self, config, device):
            self.vllm_config = config
            self.device = device

    scope: dict[str, Any] = dict(
        cast=cast,
        contextmanager=contextmanager,
        copy=copy,
        replace=replace,
        DSparkSpeculator=Parent,
        uses_dcp_replicated_gqa_draft=lambda config: True,
        prepare_replicated_pcp_config=lambda config: (config, False),
    )
    exec(
        compile(
            "from __future__ import annotations\n"
            + ast.unparse(ast.Module(body=[*helpers, mixin_node(), cls], type_ignores=[])),
            "mrv2_config",
            "exec",
        ),
        scope,
    )
    connector = SimpleNamespace(kv_connector="MooncakeConnectorV2", kv_role="kv_consumer")
    target = Config(
        SimpleNamespace(
            tensor_parallel_size=8,
            data_parallel_size=4,
            decode_context_parallel_size=8,
            cp_kv_cache_interleave_size=768,
        ),
        SimpleNamespace(block_size=768),
        connector,
        {"multistream_overlap_shared_expert": True, "recompute_scheduler_enable": True},
    )
    draft = scope[cls.name](target, "cpu")
    assert draft.target_vllm_config is target
    assert draft.vllm_config.parallel_config.decode_context_parallel_size == 1
    assert target.parallel_config.decode_context_parallel_size == 8
    assert draft.vllm_config.parallel_config.tensor_parallel_size == 8
    assert draft.vllm_config.kv_transfer_config is None and target.kv_transfer_config is connector
    assert draft.vllm_config.additional_config["multistream_overlap_shared_expert"]
    assert target.additional_config["recompute_scheduler_enable"]
    draft.vllm_config.cache_config.block_size = 128
    assert target.cache_config.block_size == 768


@pytest.mark.parametrize("replicated", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_mixin_load_delegation_and_dcp_context_restore(replicated, fail):
    from functools import lru_cache

    target = SimpleNamespace(dcp=8)
    draft = SimpleNamespace(dcp=1)
    current = [target]
    events = []
    layer = SimpleNamespace()
    target_layer = SimpleNamespace()

    @contextmanager
    def set_config(config):
        previous = current[0]
        current[0] = config
        try:
            yield
        finally:
            current[0] = previous

    @lru_cache(None)
    def enable_dcp():
        return current[0].dcp > 1

    class Parent:
        def load_draft_model(self, model, names):
            events.append((model, names, enable_dcp()))
            if fail:
                raise RuntimeError("loader failure")
            return model

    cls = mixin_node()
    scope: dict[str, Any] = dict(
        cast=cast,
        contextmanager=contextmanager,
        enable_dcp=enable_dcp,
        set_current_vllm_config=set_config,
        AttentionLayerBase=object,
        ReplicatedDraftAttentionBackend=object,
        get_layers_from_vllm_config=lambda *args: {"target": target_layer, "draft": layer},
    )
    exec(compile("from __future__ import annotations\n" + ast.unparse(cls), "mixin", "exec"), scope)
    host = type("Host", (scope[cls.name], Parent), {})()
    host.target_vllm_config = target
    host.attn_vllm_config = host.vllm_config = draft
    assert not hasattr(host, "draft_attn_layer_names")
    host.replicated_draft_kv = replicated
    assert enable_dcp()
    model = object()
    if fail:
        with pytest.raises(RuntimeError, match="loader failure"):
            host.load_draft_model(model, {"target"})
    else:
        assert host.load_draft_model(model, {"target"}) is model
    assert events == [(model, {"target"}, not replicated)]
    assert current[0] is target and enable_dcp()
    assert getattr(layer, "_ascend_dcp_replicated_draft", False) == (replicated and not fail)
    assert not hasattr(target_layer, "_ascend_dcp_replicated_draft")
    assert not hasattr(target_layer, "attn_backend")
    if replicated and not fail:
        assert layer.attn_backend is object


@pytest.mark.parametrize("use_v2", [False, True])
def test_platform_replicated_draft_exception_requires_v2(use_v2):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/platform.py"
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_draft_decode_context_parallel_config"
    )
    scope: dict[str, Any] = {}
    exec(compile("from __future__ import annotations\n" + ast.unparse(function), str(source), "exec"), scope)
    target = SimpleNamespace(hf_config=SimpleNamespace(model_type="kimi_k3"))
    draft = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3", architectures=["Qwen3DSparkModel"]),
        use_mla=False,
        model_arch_config=SimpleNamespace(total_num_attention_heads=96),
        get_total_num_kv_heads=lambda: 16,
    )
    config = SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_config=target,
        parallel_config=SimpleNamespace(tensor_parallel_size=8, decode_context_parallel_size=8),
        speculative_config=SimpleNamespace(
            num_speculative_tokens_per_batch_size=None,
            use_dspark=lambda: True,
            draft_model_config=draft,
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=8),
        ),
    )
    if use_v2:
        scope[function.name](config)
    else:
        with pytest.raises(ValueError, match="must be greater than total num kv heads"):
            scope[function.name](config)


@pytest.mark.parametrize("scheduler", [None, {}, {"recompute_scheduler_enable": True, "other": 3}])
@pytest.mark.parametrize("legacy", [False, True])
def test_draft_recompute_options_are_isolated(scheduler, legacy):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dcp_utils.py"
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "draft_additional_config"
    )
    scope: dict[str, Any] = dict(copy=copy)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)
    target = dict(recompute_scheduler_enable=legacy, scheduler_config=scheduler, multistream_overlap_shared_expert=True)
    original = copy.deepcopy(target)
    draft = scope[function.name](target)
    assert target == original
    assert not draft["recompute_scheduler_enable"]
    assert not draft["scheduler_config"]["recompute_scheduler_enable"]
    assert draft["multistream_overlap_shared_expert"]
    if scheduler and "other" in scheduler:
        assert draft["scheduler_config"]["other"] == 3


@pytest.mark.parametrize("layer_count,layer_stride,valid", [(1, 0, True), (2, 0, False), (2, 16, True)])
def test_replicated_planner_single_layer_descriptors(layer_count, layer_stride, valid):
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    source = root / "worker/v2/attn_utils.py"
    function = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_allocate_kv_cache"
    )
    attention_type = type("AttentionSpec", (), {"page_size_bytes": 8})
    mamba_type = type("MambaSpec", (), {"page_size_bytes": 8})
    names = [f"target{i}" for i in range(layer_count)]
    specs = {name: attention_type() for name in names}
    specs["state"] = mamba_type()
    descriptors = [
        SimpleNamespace(size=64, layers=names, offset=0, layer_stride=layer_stride, block_stride=8),
        SimpleNamespace(size=64, layers=["state"], offset=32, layer_stride=0, block_stride=8),
    ]
    config = SimpleNamespace(
        num_blocks=2,
        kv_cache_tensors=descriptors,
        kv_cache_groups=[SimpleNamespace(layer_names=list(specs))],
    )
    scope: dict[str, Any] = dict(
        torch=torch,
        KVPPConfig=SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=1)),
        get_current_vllm_config=lambda: SimpleNamespace(kv_transfer_config=None),
        _is_dsv4_model=lambda _: False,
        _get_layer_kv_cache_specs=lambda _: specs,
        AttentionSpec=attention_type,
        MambaSpec=mamba_type,
        AscendIndexerKPoolTailSpec=type("OtherSpec", (), {}),
        vllm_version_is=lambda _: False,
        get_kv_cache_tensor_layers=lambda descriptor: descriptor.layers,
        is_hidden_state_cache_spec=lambda _: False,
    )
    exec(compile("from __future__ import annotations\n" + ast.unparse(function), str(source), "exec"), scope)
    allocate = scope[function.name]
    if not valid:
        with pytest.raises(ValueError, match="contiguous per-layer"):
            allocate(config, {}, torch.device("cpu"))
        return
    caches = allocate(config, {}, torch.device("cpu"))
    assert all(cache.is_contiguous() and cache.numel() == 16 for cache in caches.values())
    assert caches["state"].storage_offset() == 32
    caches["target0"].fill_(7)
    assert not caches["state"].any()
    if layer_count > 1:
        assert caches["target1"].storage_offset() == 16
        assert not caches["target1"].any()


@pytest.mark.parametrize("fail", [False, True])
def test_propose_keeps_draft_dcp_context_through_graph_replay(fail):
    root = Path(__file__).resolve().parents[3] / "vllm_ascend"
    tree = ast.parse((root / "worker/v2/spec_decode/dspark/speculator.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "propose"]
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    active = [False]
    calls = []

    class Parent:
        def propose(self, *args, **kwargs):
            assert active[0], "Graph replay selected the target DCP implementation"
            calls.append(args)
            if fail:
                raise RuntimeError("replay failed")
            return "draft"

    @contextmanager
    def draft_context():
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    scope: dict[str, Any] = dict(
        Parent=Parent,
        vllm_version_is=lambda _: False,
        build_attn_metadata_wrapper=nullcontext,
        build_draft_attn_metadata_factory=lambda *args, **kwargs: nullcontext(),
        torch=SimpleNamespace(from_numpy=lambda x: x),
    )
    exec(compile("from __future__ import annotations\n" + ast.unparse(cls), "propose", "exec"), scope)
    host = scope[cls.name]()
    host._draft_dcp_context = draft_context
    host.input_buffers = SimpleNamespace(positions=None)
    host.max_num_tokens = 16
    batch = SimpleNamespace(is_prefilling_np=[False])
    if fail:
        with pytest.raises(RuntimeError, match="replay failed"):
            host.propose(batch, *([None] * 11))
    else:
        assert host.propose(batch, *([None] * 11)) == "draft"
    assert calls and not active[0]


def test_replicated_backend_stays_local_under_target_dcp():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dcp_utils.py"
    cls = next(
        n for n in ast.parse(source.read_text()).body if getattr(n, "name", None) == "ReplicatedDraftAttentionBackend"
    )
    local_builder = type("LocalBuilder", (), {})
    local_impl = type("LocalImpl", (), {})

    class TargetBackend:
        @staticmethod
        def get_builder_cls():
            raise AssertionError("Target DCP builder must not be used for replicated draft")

        @staticmethod
        def get_impl_cls():
            raise AssertionError("Target DCP implementation must not be used for replicated draft")

    scope = dict(
        AscendAttentionBackend=TargetBackend,
        AscendAttentionMetadataBuilder=local_builder,
        AscendAttentionBackendImpl=local_impl,
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), scope)
    backend = scope[cls.name]
    assert backend.get_builder_cls() is local_builder
    assert backend.get_impl_cls() is local_impl
