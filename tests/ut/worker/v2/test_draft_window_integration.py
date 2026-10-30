# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Source-contract integration across the paired upstream and Ascend seams.

Execute unchanged complete Ascend speculator class bodies, the upstream common
metadata method, and Ascend common metadata/factory functions. Only device
constructors, the proposal native body and final backend builder are isolated.
This is CPU interface/ownership evidence, not an engine or NPU execution test.
"""

import ast
import importlib.util
import inspect
import sys
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def _upstream_root():
    spec = importlib.util.find_spec("vllm")
    if spec is not None and spec.submodule_search_locations:
        installed = Path(next(iter(spec.submodule_search_locations)))
        if (installed / "v1/attention/backend.py").is_file():
            return installed
    sibling = ROOT.parent / "vllm/vllm"
    if sibling.is_dir():
        return sibling
    pytest.skip("paired vLLM source is required for real signature and metadata contracts")


def _node(path, name, parent=None) -> Any:
    tree = ast.parse(path.read_text())
    if parent is None:
        nodes = tree.body
    else:
        owner = next(node for node in tree.body if getattr(node, "name", None) == parent)
        assert isinstance(owner, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        nodes = owner.body
    return deepcopy(next(node for node in nodes if getattr(node, "name", None) == name))


def _execute(nodes, scope, filename):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, str(filename), "exec"), scope)


def _upstream_ar_attention_method(name):
    path = _upstream_root() / "v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py"
    klass = _node(path, "TargetDependentARSpeculator")
    if any(getattr(node, "name", None) == name for node in klass.body):
        return path, _node(path, name, "TargetDependentARSpeculator")
    # Main shares attention initialization and capability selection with
    # DraftModelSpeculator. Extract the actual inherited method, not a stub.
    assert any(isinstance(base, ast.Name) and base.id == "DraftModelSpeculator" for base in klass.bases)
    path = _upstream_root() / "v1/worker/gpu/spec_decode/speculator.py"
    if name == "_configure_fused_multi_step_decode":
        name = "_configure_decode_metadata_update"
    return path, _node(path, name, "DraftModelSpeculator")


def _read_leaf(monkeypatch):
    name = "_draft_window_read_source_contract"
    path = ROOT / "vllm_ascend/worker/v2/spec_decode/draft_window_read.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["dflash", "dspark", "autoregressive"])
def contract(request, monkeypatch):
    """Execute paired metadata and Ascend adapters with device imports isolated.

    Complete adapter method bodies stay intact. Only the base proposal body
    and device-only dependencies are replaced at their explicit boundaries.
    """
    upstream = _upstream_root()
    module = ModuleType("_draft_window_source_contract")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    scope = vars(module)
    CUDAGraphMode = Enum("CUDAGraphMode", ["NONE", "FULL"])
    scope.update(
        torch=torch,
        np=np,
        Mapping=Mapping,
        dataclass=dataclass,
        field=field,
        contextmanager=contextmanager,
        nullcontext=nullcontext,
        CUDAGraphMode=CUDAGraphMode,
        AscendAttentionState=SimpleNamespace(ChunkedPrefill=object(), DecodeOnly=object()),
        BatchExecutionDescriptor=SimpleNamespace,
        disable_target_pcp_for_replicated_draft=lambda _: nullcontext(),
        torch_gather_wrapper=nullcontext,
        ring_state_update_skipped=lambda: False,
        uses_turboquant_groups=lambda _: False,
        vllm_version_is=lambda _: False,
        LmheadTPDraftSamplingMixin=type("LmheadTPDraftSamplingMixin", (), {}),
    )
    for name in (
        "AscendDSAMetadataBuilder",
        "AscendDSACPMetadataBuilder",
        "AscendDSAV41MetadataBuilder",
        "AscendSFAMetadataBuilder",
        "QSAMetadataBuilder",
        "GDNAttentionMetadataBuilder",
    ):
        scope[name] = type(name, (), {})
    common_path = upstream / "v1/attention/backend.py"
    common_ascend_path = ROOT / "vllm_ascend/attention/utils.py"
    attn_path = ROOT / "vllm_ascend/worker/v2/attn_utils.py"
    _execute([_node(common_path, "CommonAttentionMetadata")], scope, common_path)
    _execute([_node(common_ascend_path, "AscendCommonAttentionMetadata")], scope, common_ascend_path)
    build_module = SimpleNamespace(build_attn_metadata=None)
    scope["_BUILD_ATTN_METADATA_MODULE"] = build_module
    _execute(
        [
            _node(attn_path, name)
            for name in ("build_attn_metadata", "build_attn_metadata_factory", "build_attn_metadata_wrapper")
        ],
        scope,
        attn_path,
    )
    build_module.build_attn_metadata = scope["build_attn_metadata"]
    base_path = upstream / "v1/worker/gpu/spec_decode/speculator.py"
    upstream_scope = dict(scope)
    upstream_scope["build_attn_metadata"] = lambda **kwargs: build_module.build_attn_metadata(**kwargs)
    _execute([_node(base_path, "_build_attn_metadata", "DraftModelSpeculator")], upstream_scope, base_path)
    is_ar = request.param == "autoregressive"
    proposal_path = upstream / (
        "v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py"
        if is_ar
        else "v1/worker/gpu/spec_decode/dflash/speculator.py"
    )
    upstream_class = "TargetDependentARSpeculator" if is_ar else "DFlashSpeculator"
    proposal = _node(proposal_path, "propose", upstream_class)
    proposal.decorator_list = []
    proposal.body = ast.parse("return _proposal_hook(self, locals())").body
    proposal_hook = Mock(return_value=object())
    upstream_scope["_proposal_hook"] = proposal_hook
    _execute([proposal], upstream_scope, proposal_path)
    base: Any = type(
        "SignatureCheckedUpstream",
        (),
        {name: upstream_scope[name] for name in ("_build_attn_metadata", "propose")},
    )
    prefill_hook = Mock()
    if is_ar:
        prefill = _node(proposal_path, "_prefill", upstream_class)
        prefill.decorator_list = []
        prefill.body = ast.parse("return _prefill_hook(self, locals())").body
        upstream_scope["_prefill_hook"] = prefill_hook
        _execute([prefill], upstream_scope, proposal_path)
        base._prefill = upstream_scope["_prefill"]
        scope["_UPSTREAM_PREFILL_HAS_SKIP_SAMPLE"] = "skip_sample" in inspect.signature(base._prefill).parameters
    scope["DFlashSpeculator"] = scope["DSparkSpeculator"] = scope["TargetDependentARSpeculator"] = base
    source_route = "target_dependent_ar" if is_ar else request.param
    source_path = ROOT / f"vllm_ascend/worker/v2/spec_decode/{source_route}/speculator.py"
    class_name = {
        "dflash": "AscendDFlashSpeculator",
        "dspark": "AscendDSparkSpeculator",
        "autoregressive": "AscendTargetDependentARSpeculator",
    }[request.param]
    _execute([_node(source_path, class_name)], scope, source_path)
    cls = scope[class_name]
    obj = object.__new__(cls)
    full_table = torch.arange(8, dtype=torch.int32).view(1, 8)
    slots = torch.arange(3, dtype=torch.int64).view(1, 3)
    positions = torch.tensor([18, 19, 20])
    obj.input_buffers = SimpleNamespace(
        positions=positions, seq_lens=torch.tensor([21], dtype=torch.int32), query_start_loc=torch.tensor([0, 3])
    )
    obj.block_tables = SimpleNamespace(input_block_tables=[full_table], slot_mappings=slots)
    obj.draft_is_prefilling = torch.tensor([False])
    obj.max_model_len = 1024
    obj.draft_max_seq_len = 25
    obj.dcp_size = 1
    if not is_ar:
        obj.num_query_per_req = 3
    obj.max_num_tokens = 3
    obj.max_num_reqs = 1
    obj.attn_architecture = "GQA"
    obj.use_dcp = False
    obj.replicated_pcp = False
    obj.enable_adaptive_verification = False
    obj.attn_vllm_config = SimpleNamespace(parallel_config=object())
    obj.draft_vllm_config = obj.attn_vllm_config
    obj.model_state = object()
    obj._sparse_kv_offload_metadata = SimpleNamespace(build_kwargs=Mock(return_value=None))
    obj.input_batch = SimpleNamespace(
        is_prefilling_np=np.array([False]),
        query_start_loc_np=np.array([0, 3]),
        seq_lens_cpu_upper_bound=torch.tensor([22]),
    )
    obj._get_seq_lens_cpu = Mock(return_value=torch.tensor([22]))
    obj.draft_attn_layer_names = {"draft.a", "draft.b"}
    obj._prepare_replicated_prefill_attn = lambda metadata, slots, *args, **kwargs: (metadata, slots)
    obj._prepare_draft_dcp_metadata_inputs = Mock(return_value=(None, torch.tensor([False])))
    backend = Mock()
    backend.consumes_pcp_context = False
    backend.build.side_effect = lambda **kwargs: kwargs["common_attn_metadata"]
    obj.attn_groups = [[SimpleNamespace(layer_names=["draft.a", "draft.b"], get_metadata_builder=lambda _: backend)]]
    obj.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    return SimpleNamespace(
        obj=obj,
        mode=CUDAGraphMode,
        scope=scope,
        backend=backend,
        base=base,
        proposal_hook=proposal_hook,
        method=request.param,
        table=full_table,
        slots=slots,
        positions=positions,
        read=_read_leaf(monkeypatch),
        prefill_hook=prefill_hook,
    )


def _build(contract, **extra):
    return contract.obj._build_attn_metadata(
        1,
        SimpleNamespace(num_reqs=1, num_tokens=3, cg_mode=contract.mode.NONE),
        np.array([0, 3], dtype=np.int32),
        torch.tensor([22], dtype=torch.int32),
        3,
        **extra,
    )


def test_explicit_full_write_mapping_preserves_upstream_signature(contract):
    # This parameter was added upstream after the old Ascend override. Call the
    # actual override with it so a missing passthrough produces a real TypeError.
    assert "slot_mappings" in inspect.signature(contract.base._build_attn_metadata).parameters
    explicit_slots = torch.tensor([[70, 71, 72]])
    result = _build(contract, slot_mappings=explicit_slots)
    assert result["draft.a"].slot_mapping.untyped_storage().data_ptr() == explicit_slots.untyped_storage().data_ptr()
    assert result["draft.a"].slot_mapping.tolist() == [70, 71, 72]
    assert contract.slots.tolist() == [[0, 1, 2]]


def test_callback_final_writer_updates_real_common_cpu_mirrors_and_keeps_full_operands(contract):
    retained: list[Any] = []
    called: list[Any] = []
    read = contract.read
    exact = torch.tensor([21], dtype=torch.int32)
    view = read.materialize_direct_prefix(
        read.DirectPrefixSource(contract.table, (0,), (8,), 4),
        exact_cpu=exact,
        exact_device=exact,
        upper_cpu=torch.tensor([25], dtype=torch.int32),
        starts_cpu=torch.tensor([12], dtype=torch.int32),
        padded_rows=1,
    )

    def callback(gid, common):
        called.append(gid)
        assert common.seq_lens_cpu.tolist() == [25]  # Upstream scheduling upper before projection.
        view.apply(common)

    owned_builder = Mock()
    owned_builder.consumes_pcp_context = False
    owned_builder.build.side_effect = lambda **kwargs: kwargs["common_attn_metadata"]
    select_builder = Mock(return_value=owned_builder)
    runtime = SimpleNamespace(
        metadata_callback=Mock(return_value=callback),
        metadata_builder_callback=Mock(return_value=select_builder),
        retain_metadata=retained.append,
        reads_active=True,
    )
    contract.obj._draft_window_runtime = runtime
    with contract.scope["build_attn_metadata_wrapper"]():
        result = _build(contract)
    common = result["draft.a"]
    assert called == [0]
    assert result["draft.b"] is common
    assert common.block_table_tensor is view.block_table
    assert common.seq_lens_cpu is common._seq_lens_cpu is view.seq_lens_cpu
    assert common.seq_lens_cpu.tolist() == [9]
    assert common.seq_lens_cpu_upper_bound.tolist() == [13]
    assert common.seq_lens.tolist() == [9]
    assert common.positions.untyped_storage().data_ptr() == contract.positions.untyped_storage().data_ptr()
    assert common.slot_mapping.untyped_storage().data_ptr() == contract.slots.untyped_storage().data_ptr()
    assert contract.obj.input_buffers.seq_lens.tolist() == [21]
    assert contract.table.tolist() == [list(range(8))]
    assert retained == [result]
    runtime.metadata_callback.assert_called_once_with(3, 1)
    runtime.metadata_builder_callback.assert_called_once_with(3)
    select_builder.assert_called_once_with(0, contract.backend, ["draft.a", "draft.b"])
    contract.backend.build.assert_not_called()
    owned_builder.build.assert_called_once()


def test_off_metadata_retains_full_read_and_write_storage(contract):
    result = _build(contract)
    common = result["draft.a"]
    assert common.block_table_tensor.untyped_storage().data_ptr() == contract.table.untyped_storage().data_ptr()
    assert (
        common.seq_lens.untyped_storage().data_ptr() == contract.obj.input_buffers.seq_lens.untyped_storage().data_ptr()
    )
    assert common.slot_mapping.untyped_storage().data_ptr() == contract.slots.untyped_storage().data_ptr()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("num_speculative_tokens", [0, 3])
def test_propose_uses_real_upstream_signature_scope_and_dp_rules(contract, enabled, profiling, num_speculative_tokens):
    batch = SimpleNamespace(is_prefilling_np=np.array([False]), num_reqs=1)
    arguments = dict(
        input_batch=batch,
        attn_metadata=object(),
        slot_mappings=object(),
        last_hidden_states=object(),
        aux_hidden_states=[object()],
        num_sampled=object(),
        num_rejected=object(),
        last_sampled=object(),
        next_prefill_tokens=object(),
        temperature=object(),
        seeds=object(),
        dp_sync_state=object(),
        dummy_run=profiling,
        skip_attn_for_dummy_run=profiling,
        mm_inputs=object(),
        is_profile=profiling,
        num_speculative_tokens=num_speculative_tokens,
    )
    events = []

    @contextmanager
    def proposal_context(speculator, input_batch, dummy_run, is_profile, **kwargs):
        assert (speculator, input_batch, dummy_run, is_profile) == (contract.obj, batch, profiling, profiling)
        expected = (
            {"num_speculative_tokens": num_speculative_tokens}
            if contract.method == "autoregressive"
            else {"context_kv_only": num_speculative_tokens == 0}
        )
        assert kwargs == expected
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    if enabled:
        contract.obj._draft_window_runtime = SimpleNamespace(proposal_context=proposal_context)
    result = contract.obj.propose(**arguments)
    assert result is contract.proposal_hook.return_value
    passed = contract.proposal_hook.call_args.args[1]
    for name, value in arguments.items():
        reset_dp = name == "dp_sync_state" and profiling and contract.method != "autoregressive"
        assert passed[name] is (None if reset_dp else value)
    assert events == (["enter", "exit"] if enabled else [])


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
def test_eagle_first_pass_builds_fresh_draft_metadata_preserving_target_and_write_identity(contract):
    target_common = object()
    target_metadata = {"draft.a": target_common, "target.only": object()}
    full_write = {"draft.a": torch.tensor([31, 32, 33])}
    frozen_read = SimpleNamespace(seq_lens_cpu=torch.tensor([9]))
    new_metadata = {"draft.a": frozen_read}
    contract.obj._draft_window_runtime = SimpleNamespace(reads_active=True)
    contract.obj._build_attn_metadata = Mock(return_value=new_metadata)
    contract.obj._prefill(1, 3, target_metadata, full_write, None, contract.mode.NONE)
    passed = contract.prefill_hook.call_args.args[1]
    assert passed["attn_metadata"] == new_metadata
    assert passed["attn_metadata"]["draft.a"] is frozen_read
    assert passed["slot_mappings"] is full_write
    assert target_metadata["draft.a"] is target_common
    assert "target.only" in target_metadata
    build = contract.obj._build_attn_metadata.call_args.kwargs
    assert build["step"] == 0
    assert build["query_start_loc_np"] is contract.obj.input_batch.query_start_loc_np


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
def test_eagle_window_update_cannot_overwrite_inflight_read(contract):
    read = torch.tensor([9], dtype=torch.int32)
    metadata = {"draft.a": SimpleNamespace(seq_lens_cpu=read, seq_lens_list=[9])}
    contract.obj._draft_window_runtime = SimpleNamespace(reads_active=True)
    contract.obj._get_seq_lens_cpu.side_effect = AssertionError("Window updater read Full CPU lengths")
    contract.obj._update_decode_attn_metadata(metadata, 2, 1)
    assert metadata["draft.a"].seq_lens_cpu is read
    assert read.tolist() == [9]


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
def test_current_dense_eagle_uses_upstream_unfused_selection(contract):
    upstream = _upstream_root()
    base = _node(upstream / "v1/attention/backend.py", "AttentionMetadataBuilder")
    flag = next(
        node
        for node in base.body
        if isinstance(node, ast.AnnAssign)
        and getattr(node.target, "id", None) == "supports_draft_decode_metadata_update"
    )
    assert flag.value is not None
    assert ast.literal_eval(flag.value) is False
    dense = _node(ROOT / "vllm_ascend/attention/attention_v1.py", "AscendAttentionMetadataBuilder")
    assert not any(
        isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "supports_draft_decode_metadata_update"
        for node in dense.body
    )
    scope = dict(contract.scope, logger=SimpleNamespace(info_once=Mock()))
    path, configure = _upstream_ar_attention_method("_configure_fused_multi_step_decode")
    _execute([configure], scope, path)
    contract.obj.num_speculative_steps = 8
    contract.obj.attn_groups = [
        [
            SimpleNamespace(
                supports_draft_decode_metadata_update=ast.literal_eval(flag.value),
                backend=SimpleNamespace(get_name=lambda: "ASCEND"),
            )
        ]
    ]
    scope[configure.name](contract.obj)
    capability = (
        contract.obj.supports_decode_metadata_update
        if configure.name == "_configure_decode_metadata_update"
        else contract.obj.use_fused_multi_step_decode
    )
    assert capability is False


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
@pytest.mark.parametrize("enabled", [False, True])
def test_eagle_set_attn_installs_runtime_only_after_backend_selection(contract, monkeypatch, enabled):
    path, node = _upstream_ar_attention_method("set_attn")
    node.body = ast.parse("return _set_attn_hook(self, locals())").body
    hook = Mock()
    scope = dict(contract.scope, _set_attn_hook=hook)
    _execute([node], scope, path)
    contract.base.set_attn = scope["set_attn"]
    dense = type("DenseBackend", (), {})
    contract.scope.update(
        set_current_vllm_config=lambda _: nullcontext(),
        _get_graph_update_backend=lambda _: dense,
        AscendAttentionBackend=dense,
    )
    for name in ("AscendDSABackend", "AscendMLABackend", "AscendSFABackend", "AscendSFAIndexerBackend"):
        contract.scope[name] = type(name, (), {})
    module = ModuleType("vllm_ascend.worker.v2.spec_decode.draft_window_runtime")
    runtime = Mock(return_value=object())
    module.__dict__["DraftWindowRuntime"] = runtime
    monkeypatch.setitem(sys.modules, module.__name__, module)
    policy = object()
    if enabled:
        contract.obj.draft_window_policy = policy
    arguments = [object() for _ in range(5)]
    contract.obj.set_attn(*arguments)
    assert list(hook.call_args.args[1].values())[1:] == arguments
    assert contract.obj.attn_backend is dense
    assert contract.obj.attn_architecture == "GQA"
    if enabled:
        runtime.assert_called_once_with(contract.obj, policy)
        assert contract.obj._draft_window_runtime is runtime.return_value
    else:
        runtime.assert_not_called()
        assert not hasattr(contract.obj, "_draft_window_runtime")


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
@pytest.mark.parametrize("consumer", ["sink", "c8"])
def test_eagle_window_first_pass_keeps_multitoken_phase_at_real_consumer(contract, consumer):
    """Execute first-pass assembly and actual consumer dispatch/native arguments.

    Device kernels are recording stubs; the copied function bodies and metadata
    type are unchanged production source. This proves the host query-width
    contract, not NPU arithmetic or cache-write contents.
    """
    from types import MethodType

    path = ROOT / "vllm_ascend/attention/attention_v1.py"
    scope = contract.scope
    scope["Enum"] = Enum
    _execute([_node(path, "AscendAttentionState"), _node(path, "AscendMetadata")], scope, path)
    state, metadata_type = scope["AscendAttentionState"], scope["AscendMetadata"]
    native_calls = []

    def native(query, *args, **kwargs):
        native_calls.append((query, kwargs))
        return torch.ones_like(query), None

    scope.update(
        _EXTRA_CTX=SimpleNamespace(capturing=False),
        torch_npu=SimpleNamespace(npu_fused_infer_attention_score=native, npu_fused_infer_attention_score_v2=native),
        AttentionType=SimpleNamespace(ENCODER_DECODER=object()),
        SWA_INT_MAX=2147483647,
    )
    for method, owner in (
        ("forward_fused_infer_attention", "AscendAttentionBackendImpl"),
        ("forward", "AscendC8AttentionBackendImpl"),
        ("_forward_c8_decode", "AscendC8AttentionBackendImpl"),
    ):
        _execute([_node(path, method, owner)], scope, path)
    contract.obj.input_batch.is_prefilling_np[:] = True
    contract.obj.draft_is_prefilling[:] = True
    contract.obj._draft_window_runtime = SimpleNamespace(
        reads_active=True,
        metadata_callback=lambda *args: lambda gid, common: None,
        metadata_builder_callback=lambda *args: None,
        retain_metadata=lambda metadata: None,
    )

    def build_typed_metadata(**kwargs):
        common = kwargs["common_attn_metadata"]
        return metadata_type(
            attn_state=common.attn_state,
            num_actual_tokens=3,
            num_prefills=1,
            num_decodes=0,
            seq_lens=common.seq_lens,
            seq_lens_cpu=common.seq_lens_cpu,
            seq_lens_list=common.seq_lens_cpu.tolist(),
            actual_seq_lengths_q=common.query_start_loc_cpu[1:].tolist(),
            block_tables=common.block_table_tensor,
            slot_mapping=common.slot_mapping,
            model_runner_type="generate",
        )

    contract.backend.build.side_effect = build_typed_metadata
    with scope["build_attn_metadata_wrapper"]():
        metadata = contract.obj._build_attn_metadata(
            1,
            SimpleNamespace(num_reqs=1, num_tokens=3, cg_mode=contract.mode.NONE),
            np.array([0, 3], dtype=np.int32),
            torch.tensor([22], dtype=torch.int32),
            0,
        )["draft.a"]
    query = torch.zeros(3, 1, 4)
    key = torch.zeros_like(query)
    output = torch.full_like(query, float("nan"))
    cache = torch.zeros(1, 128, 1, 4)
    impl = SimpleNamespace(
        sinks=torch.ones(1),
        sliding_window=None,
        num_heads=1,
        num_kv_heads=1,
        head_size=4,
        scale=0.5,
        attn_type=None,
        key_cache=cache,
        value_cache=cache,
        _get_fia_params=lambda *args: (cache, cache, 128, metadata.block_tables, metadata.seq_lens_list),
    )
    if consumer == "sink":
        scope["forward_fused_infer_attention"](impl, query, key, key, metadata, output)
        assert list(native_calls[0][1]["actual_seq_qlen"]) == [3]
    else:
        impl.vllm_config = SimpleNamespace(kv_transfer_config=None)
        impl._prepare_c8_scales = lambda *args: None
        impl._quantize_kv_to_int8 = lambda k, v, *args: (k, v)
        impl._reshape_and_cache = lambda q, k, v, *args: (q, k, v, None)
        impl._nz_5d_view = lambda value, *args: value
        impl._forward_c8_decode = MethodType(scope["_forward_c8_decode"], impl)
        impl._forward_c8_chunked_prefill = Mock(side_effect=lambda *args: output.fill_(1))
        layer = SimpleNamespace(layer_name="draft.a", _c8_k_aq_scale_nz_bnsd=None, _c8_v_aq_scale_nz_bnsd=None)
        scope["forward"](impl, layer, query, key, key, (cache, cache), metadata, output)
        assert not torch.isnan(output).any(), "DecodeOnly incorrectly consumed only one row of the prefill query"
        impl._forward_c8_chunked_prefill.assert_called_once()
        assert not native_calls
    assert metadata.attn_state is state.ChunkedPrefill
    assert metadata.actual_seq_lengths_q == [3]


@pytest.mark.parametrize("contract", ["autoregressive"], indirect=True)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("prefilling", [False, True])
def test_eagle_replicated_prefill_builds_once_after_full_mapping_restore(contract, enabled, prefilling):
    """Run the real replicated helper and metadata assembly, not its fixture stub."""
    obj = contract.obj
    del obj._prepare_replicated_prefill_attn
    obj.replicated_pcp = True
    contract.scope["AscendInputBatch"] = SimpleNamespace
    slot_helper = _upstream_root() / "v1/worker/gpu/attn_utils.py"
    _execute([_node(slot_helper, "build_slot_mappings_by_layer")], contract.scope, slot_helper)
    obj.kv_cache_config.kv_cache_groups = [SimpleNamespace(layer_names=["draft.a", "draft.b"])]
    obj.input_batch.num_reqs = 1
    obj.input_batch.is_dummy = False
    obj.input_batch.idx_mapping = torch.tensor([0])
    obj.input_batch.query_start_loc = obj.input_buffers.query_start_loc
    obj.input_batch.positions = contract.positions
    obj.input_batch.is_prefilling_np[:] = prefilling
    obj.block_tables.gather_block_tables = Mock()
    restored_slots = torch.tensor([[31, 32, 33]])
    obj.block_tables.slot_mappings = restored_slots
    obj.block_tables.compute_slot_mappings = Mock(return_value=restored_slots)
    if enabled:
        obj._draft_window_runtime = SimpleNamespace(
            reads_active=True,
            metadata_callback=Mock(return_value=None),
            metadata_builder_callback=Mock(return_value=None),
            retain_metadata=Mock(),
        )
    target_value = object()
    target_metadata = {"draft.a": target_value}
    target_slots = {"draft.a": torch.tensor([7, 8, 9])}
    obj._prefill(1, 3, target_metadata, target_slots, None, contract.mode.NONE)
    obj.block_tables.gather_block_tables.assert_called_once_with(obj.input_batch.idx_mapping, num_reqs_padded=1)
    obj.block_tables.compute_slot_mappings.assert_called_once_with(
        obj.input_batch.idx_mapping,
        obj.input_batch.query_start_loc,
        obj.input_batch.positions,
        num_tokens_padded=3,
    )
    assert contract.backend.build.call_count == 1
    passed = contract.prefill_hook.call_args.args[1]
    metadata = passed["attn_metadata"]["draft.a"]
    assert metadata is not target_value
    assert metadata.slot_mapping.untyped_storage().data_ptr() == restored_slots.untyped_storage().data_ptr()
    assert (
        passed["slot_mappings"]["draft.a"].untyped_storage().data_ptr() == restored_slots.untyped_storage().data_ptr()
    )
    assert metadata.query_start_loc_cpu.tolist() == [0, 3]
    assert target_metadata["draft.a"] is target_value
    assert target_slots["draft.a"].tolist() == [7, 8, 9]
    expected = (
        contract.scope["AscendAttentionState"].ChunkedPrefill
        if enabled
        else contract.scope["AscendAttentionState"].DecodeOnly
    )
    assert metadata.attn_state is expected
    if enabled:
        obj._draft_window_runtime.metadata_callback.assert_called_once_with(0, 1)
        obj._draft_window_runtime.retain_metadata.assert_called_once()


@pytest.mark.parametrize("contract", ["dspark"], indirect=True)
@pytest.mark.parametrize("padded_rows", [1, 3])
def test_dspark_mla_reuses_prepared_exact_cpu_without_another_transfer(contract, monkeypatch, padded_rows):
    # Execute the actual DSpark helper, runtime, common metadata and proposal
    # owner. Only backend discovery/completion/native builder are isolated.
    modules = {}
    for suffix in ("lifetime", "plan", "policy", "read", "runtime"):
        name = f"vllm_ascend.worker.v2.spec_decode.draft_window_{suffix}"
        path = ROOT / f"vllm_ascend/worker/v2/spec_decode/draft_window_{suffix}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[suffix] = module
    runtime_module = modules["runtime"]
    runtime_class = runtime_module.DraftWindowRuntime
    monkeypatch.setattr(runtime_class, "_validate_route", staticmethod(lambda spec: (0,)))
    completion = SimpleNamespace(is_complete=lambda: False, wait=lambda: None)
    monkeypatch.setattr(runtime_module, "_NPUCompletion", lambda: completion)
    obj = contract.obj
    del obj._prepare_draft_dcp_metadata_inputs  # Run the real completed CPU copy.
    obj.attn_architecture = "MLA"
    obj.input_buffers.seq_lens = torch.tensor([21] + [0] * (padded_rows - 1), dtype=torch.int32)
    obj.input_buffers.query_start_loc = torch.tensor([0, 3] + [3] * (padded_rows - 1), dtype=torch.int32)
    obj.block_tables.block_tables = [SimpleNamespace(gpu=contract.table)]
    obj.block_tables.num_blocks = SimpleNamespace(np=np.array([[8]], dtype=np.int32))
    obj.block_tables.kernel_block_sizes = [4]
    obj.vllm_config = SimpleNamespace(max_concurrent_batches=2)
    obj.draft_model_config = SimpleNamespace(enforce_eager=True)
    obj.input_batch.num_reqs = 1
    obj.input_batch.req_ids = ["current-request"]
    obj.input_batch.idx_mapping_np = np.array([0])
    obj.input_batch.num_computed_prefill_tokens_np = np.array([16])
    obj.input_batch.prefill_len_np = np.array([16])
    obj.input_batch.num_scheduled_tokens = np.array([3])
    obj.input_batch.num_draft_tokens_per_req = np.array([2])
    runtime = runtime_class(
        obj, modules["policy"].resolve_draft_window_policy(requested_window_size=8, method="dspark")
    )
    obj._draft_window_runtime = runtime
    original_to, original_copy = torch.Tensor.to, torch.Tensor.copy_
    window_cpu_copies, prepared_cpu_copies = [], []

    def observe_to(tensor, *args, **kwargs):
        if kwargs == {"device": "cpu", "dtype": torch.int32, "copy": True}:
            window_cpu_copies.append(tensor)
        return original_to(tensor, *args, **kwargs)

    def observe_copy(tensor, source, *args, **kwargs):
        if source.data_ptr() == obj.input_buffers.seq_lens.data_ptr() and source.shape == (1,):
            prepared_cpu_copies.append(tensor)
        return original_copy(tensor, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observe_to)
    monkeypatch.setattr(torch.Tensor, "copy_", observe_copy)
    with runtime.proposal_context(obj, obj.input_batch):
        owner = runtime._owner
        result = obj._build_attn_metadata(
            1,
            SimpleNamespace(num_reqs=padded_rows, num_tokens=3, cg_mode=contract.mode.NONE),
            np.array([0, 3], dtype=np.int32),
            torch.tensor([30], dtype=torch.int32),  # Deliberately not exact.
            3,
        )
    assert len(prepared_cpu_copies) == 1  # Existing blocking helper copy remains.
    assert len(window_cpu_copies) == 0  # The feature adds no second exact copy.
    prepared = prepared_cpu_copies[0]
    assert any(isinstance(value, torch.Tensor) and value.data_ptr() == prepared.data_ptr() for value in owner.retained)
    assert owner.state is modules["lifetime"].DraftWindowOwnerState.SUBMITTED
    assert owner.completion is completion
    common = result["draft.a"]
    assert common.seq_lens_cpu.tolist() == [9] + [0] * (padded_rows - 1)
    assert common.seq_lens.tolist() == [9] + [0] * (padded_rows - 1)
    assert common.slot_mapping.untyped_storage().data_ptr() == contract.slots.untyped_storage().data_ptr()
    assert obj.input_buffers.seq_lens.tolist() == [21] + [0] * (padded_rows - 1)
    assert contract.table.tolist() == [list(range(8))]
