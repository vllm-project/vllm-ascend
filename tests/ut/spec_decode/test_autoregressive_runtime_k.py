# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU regressions for the Ascend runtime-K adapters.

Execute production methods with strict pre/post-#57053 parent contracts. The
NPU model, kernels, and graph recording are mocked, not validated by this suite.
"""

import ast
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

SOURCE_DIR = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode"
MODES = SimpleNamespace(FULL="full", NONE="none", PIECEWISE="piecewise")


@dataclass(frozen=True)
class BatchDescriptor:
    cg_mode: str
    num_tokens: int
    num_reqs: int | None = None


@dataclass(frozen=True)
class SpeculatorDescriptor(BatchDescriptor):
    num_speculative_tokens: int = 0


def load_class(path, name, methods, parent, namespace):
    """Compile actual method bodies, retaining zero-argument super() binding."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {"Parent": parent, "CUDAGraphMode": MODES, **namespace}
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class RuntimeParent:
    generated_lengths: list[int]

    def propose(self, *args, is_profile=None, num_speculative_tokens=None):
        self.propose_kwargs = (args, is_profile, num_speculative_tokens)
        return num_speculative_tokens

    def _multi_step_decode(self, num_reqs, skip_attn, desc, dp_tokens, seq_lens, num_speculative_steps):
        for _ in range(1, num_speculative_steps):
            self._generate_draft(num_reqs, desc.num_tokens, None, None, dp_tokens, desc.cg_mode, num_speculative_steps)

    def _generate_draft(self, num_reqs, num_tokens, attn, slots, dp_tokens, cg_mode, num_speculative_steps):
        self.generated_lengths.append(num_speculative_steps)


class LegacyParent:
    num_speculative_steps: int
    generated_lengths: list[int]

    def propose(self, *args, is_profile=None):
        self.propose_kwargs = (args, is_profile)
        return "legacy"

    def _multi_step_decode(self, num_reqs, skip_attn, desc, dp_tokens, seq_lens):
        for _ in range(1, self.num_speculative_steps):
            self._generate_draft(num_reqs, desc.num_tokens, None, None, dp_tokens, desc.cg_mode)

    def _generate_draft(self, num_reqs, num_tokens, attn, slots, dp_tokens, cg_mode):
        self.generated_lengths.append(self.num_speculative_steps)


def speculator(runtime=True):
    cls = load_class(
        SOURCE_DIR / "autoregressive/speculator.py",
        "AscendAutoRegressiveSpeculator",
        {
            "propose",
            "capture",
            "_multi_step_decode",
            "_generate_draft",
            "_init_decode_draft_attn_metadatas",
            "build_fia_params",
        },
        RuntimeParent if runtime else LegacyParent,
        {
            "_SUPPORTS_RUNTIME_K": runtime,
            "disable_target_pcp_for_replicated_draft": lambda _: nullcontext(),
            "build_attn_metadata_wrapper": nullcontext,
            "torch_gather_wrapper": nullcontext,
            "copy": copy,
            "AscendAttentionState": SimpleNamespace(DecodeOnly="decode"),
            "BatchExecutionDescriptor": BatchDescriptor,
            "logger": Mock(),
        },
    )
    obj = cls()
    obj.num_speculative_steps = 4
    obj.use_dcp = False
    obj.replicated_pcp = False
    obj.generated_lengths = []
    obj._update_decode_attn_metadata = Mock()
    return obj


@pytest.mark.parametrize("k", [None, 0, 1, 2, 4])
@pytest.mark.parametrize("replicated_pcp", [False, True])
def test_propose_forwards_runtime_k_and_preserves_dp_sync(k, replicated_pcp):
    obj = speculator()
    obj.replicated_pcp = replicated_pcp
    batch, dp_sync = object(), object()
    assert obj.propose(batch, *([None] * 10), dp_sync=dp_sync, num_speculative_tokens=k) == k
    assert obj.input_batch is batch
    assert obj.propose_kwargs[0][11] is (None if replicated_pcp else dp_sync)


@pytest.mark.parametrize("k", [None, 4])
def test_legacy_propose_keeps_old_contract(k):
    obj = speculator(runtime=False)
    assert obj.propose(object(), *([None] * 10), num_speculative_tokens=k) == "legacy"


@pytest.mark.parametrize("k", [0, 1, 2])
def test_legacy_propose_rejects_unsupported_runtime_length(k):
    with pytest.raises(ValueError, match="runtime-K interface"):
        speculator(runtime=False).propose(object(), *([None] * 10), num_speculative_tokens=k)


@pytest.mark.parametrize("mode", [MODES.NONE, MODES.PIECEWISE])
def test_decode_length_changes_do_not_use_maximum_loop(mode):
    obj = speculator()
    desc = BatchDescriptor(mode, 8, 3)
    for k in [4, 2, 0, 1, 4]:
        obj.generated_lengths.clear()
        obj._multi_step_decode(3, False, desc, None, None, k)
        assert obj.generated_lengths == [k] * max(0, k - 1)
    assert obj.num_speculative_steps == 4


@pytest.mark.parametrize("runtime", [False, True])
def test_default_decode_length_remains_maximum(runtime):
    obj = speculator(runtime)
    obj._multi_step_decode(2, False, BatchDescriptor(MODES.NONE, 2, 2), None)
    assert obj.generated_lengths == [4, 4, 4]


def test_full_decode_replays_merged_graph_once():
    obj = speculator()
    obj.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=Mock())
    desc = SpeculatorDescriptor(MODES.FULL, 8, 3, 2)
    obj._multi_step_decode(3, False, desc, None, None, 2)
    obj.decode_cudagraph_manager.run_fullgraph.assert_called_once_with(desc)
    assert obj.generated_lengths == []


def test_generate_preserves_ascend_metadata_update_with_runtime_k():
    obj = speculator()
    metadata = {"draft": object()}
    obj._generate_draft(2, 4, metadata, None, None, MODES.NONE, 2)
    assert obj.generated_lengths == [2]
    obj._update_decode_attn_metadata.assert_called_once_with(metadata, 1, 2)


@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_speculator_requests_k_specialization_for_merged_graphs(runtime, dynamic):
    obj = speculator(runtime)
    obj.last_token_indices = SimpleNamespace(zero_=Mock())
    obj.prefill_cudagraph_manager = SimpleNamespace(use_breakable_cg=False, capture=Mock())
    obj.decode_cudagraph_manager = SimpleNamespace(capture=Mock())
    obj.speculative_config = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: dynamic)
    obj.model_state = obj.target_input_buffers = obj.block_tables = obj.draft_prefill_attn_groups = object()
    obj.input_buffers = obj.attn_groups = obj.kv_cache_config = object()
    obj._prefill = Mock()
    obj.capture()
    obj.prefill_cudagraph_manager.capture.assert_called_once()
    assert obj.decode_cudagraph_manager.capture.call_args.kwargs["specialize_spec_tokens"] is (runtime and dynamic)


@pytest.mark.parametrize("architecture,use_dcp", [("GQA", False), ("GQA", True), ("MLA", False)])
@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_decode_metadata_has_only_runtime_steps(architecture, use_dcp, k):
    obj = speculator()
    obj.attn_architecture = architecture
    obj.use_dcp = use_dcp
    obj.input_batch = SimpleNamespace(num_reqs=2, seq_lens_cpu_upper_bound=[9, 10])
    obj.input_buffers = SimpleNamespace(draft_seq_lens_cpus=[[0] * 8 for _ in range(3)])
    metadata = SimpleNamespace(decode=SimpleNamespace())
    obj._build_uniform_attn_metadata = lambda **_: {"draft": metadata}
    result = obj._init_decode_draft_attn_metadatas({"draft": metadata}, 8, k)
    assert len(result) == max(0, k - 1)
    assert len({id(step["draft"]) for step in result}) == len(result)
    if architecture == "MLA" or use_dcp:
        assert len({id(step["draft"].decode) for step in result}) == len(result)
    assert len(obj.input_buffers.draft_seq_lens_cpus) == 3


@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_fia_updates_match_runtime_k_with_padding_and_seq_limit(k):
    obj = speculator()
    obj.input_batch = SimpleNamespace(num_reqs=2, seq_lens_np=[9, 19])
    obj.max_model_len = 20
    obj.draft_attn_layer_names = ["draft.0", "draft.1"]
    block_table = object()
    result = obj.build_fia_params(4, {"draft.0": SimpleNamespace(block_tables=block_table)}, False, k)
    assert len(result) == 2 * max(0, k - 1)
    for step in range(1, k):
        for params in result[2 * (step - 1) : 2 * step]:
            assert params["actual_seq_lengths_kv"] == [9 + step, 20, 0, 0]
            assert params["actual_seq_lengths"] == [1, 2, 3, 4]
            assert params["block_table"] is block_table


@pytest.mark.parametrize("rank", [0, 1])
def test_dcp_fia_updates_follow_runtime_k_and_keep_local_lengths(rank):
    local_lengths = Mock(side_effect=lambda lengths, **_: lengths // 2)
    cls = load_class(
        SOURCE_DIR / "autoregressive/speculator.py",
        "AscendAutoRegressiveSpeculator",
        {"build_fia_params", "build_fia_params_dcp"},
        object,
        {
            "torch": SimpleNamespace(tensor=lambda values, dtype: np.asarray(values, dtype=dtype), int32=np.int32),
            "get_dcp_local_seq_lens": local_lengths,
            "CPKVScope": SimpleNamespace(FULL="full"),
        },
    )
    obj = cls()
    obj.use_dcp = True
    obj.dcp_manager = SimpleNamespace(dcp_world_rank=rank)
    obj.draft_vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=2, cp_kv_cache_interleave_size=1)
    )
    obj.input_batch = SimpleNamespace(num_reqs=2, seq_lens_np=[9, 19])
    obj.num_speculative_steps = 4
    obj.max_model_len = 20
    obj.draft_attn_layer_names = ["draft.0", "draft.1"]
    block_tables = {layer: object() for layer in obj.draft_attn_layer_names}
    metadata = {
        layer: SimpleNamespace(decode=SimpleNamespace(block_tables=block_table))
        for layer, block_table in block_tables.items()
    }
    for k in [4, 2, 0, 1, 4, None]:
        local_lengths.reset_mock()
        result = obj.build_fia_params(4, metadata, False, k)
        num_steps = 4 if k is None else k
        assert len(result) == 2 * max(0, num_steps - 1)
        assert local_lengths.call_count == max(0, num_steps - 1)
        for step in range(1, num_steps):
            call = local_lengths.call_args_list[step - 1]
            np.testing.assert_array_equal(call.args[0], [9 + step, 20, 0, 0])
            assert call.kwargs == {"dcp_size": 2, "dcp_rank": rank, "cp_kv_cache_interleave_size": 1}
            for params, layer in zip(result[2 * (step - 1) : 2 * step], obj.draft_attn_layer_names):
                assert params["layer_name"] == (layer, "full")
                assert params["actual_seq_lengths"] == [1, 2, 3, 4]
                assert params["actual_seq_lengths_kv"] == [(9 + step) // 2, 10, 0, 0]
                assert params["block_table"] is block_tables[layer]
    assert obj.num_speculative_steps == 4


@pytest.mark.parametrize("path", ["dflash/speculator.py", "dspark/speculator.py"])
def test_parallel_drafters_accept_new_runner_keyword(path):
    tree = ast.parse((SOURCE_DIR / path).read_text(encoding="utf-8"))
    propose = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "propose")
    assert propose.args.args[-1].arg == "num_speculative_tokens"
    assert isinstance(propose.args.defaults[-1], ast.Constant) and propose.args.defaults[-1].value is None


class GraphParent:
    graphs: dict[BatchDescriptor, object]

    def run_fullgraph(self, desc):
        return "replayed"

    def specialize_spec_tokens(self, desc, k):
        if desc.cg_mode != MODES.FULL or desc in self.graphs:
            return desc
        specialized = SpeculatorDescriptor(**vars(desc), num_speculative_tokens=k)
        return specialized if specialized in self.graphs else replace(specialized, cg_mode=MODES.NONE)


class CaptureParent:
    @staticmethod
    def capture(manager, create_forward_fn, progress_bar_desc=None):
        for descs in manager._capture_descs.values():
            for desc in descs:
                create_forward_fn(desc, False)(MODES.NONE)
                manager.graphs[desc] = object()


class FakeUpdatableGraph:
    def __init__(self):
        self.resolve_tasks = Mock(return_value="tasks")
        self.update = Mock()


def graph_manager(updatable=True):
    cls = load_class(
        SOURCE_DIR / "autoregressive/aclgraph.py",
        "AutoRegressiveAclGraphManager",
        {"capture", "specialize_spec_tokens", "run_fullgraph", "_updatable_graph_replay"},
        GraphParent,
        {
            "replace": replace,
            "fields": fields,
            "SpeculatorCudaGraphManager": GraphParent,
            "CudaGraphManager": CaptureParent,
            "communicator_switch": nullcontext,
            "model_capture_wrapper": lambda *_: nullcontext(),
            "use_updatable_graph": lambda _: updatable,
            "speculator_graphs": SimpleNamespace(
                build_dynamic_sd_schedule_lookup=lambda *_, **__: [0, 4, 2, 1, 0],
                SpeculatorBatchDescriptor=SpeculatorDescriptor,
            ),
            "prepare_inputs_to_capture": Mock(),
            "logger": Mock(),
            "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: "stream")),
            "UpdatableGraph": FakeUpdatableGraph,
            "SharedSource": lambda params: params,
        },
    )
    obj = cls()
    obj.is_draft_model_prefill = False
    obj.max_num_reqs = 4
    obj.dp_size = 1
    obj.update_stream = object()
    obj.vllm_config = SimpleNamespace(
        num_speculative_tokens=4, speculative_config=SimpleNamespace(num_speculative_tokens_per_batch_size=[])
    )
    obj.speculator = SimpleNamespace(
        attn_backend=object(), draft_vllm_config=object(), build_draft_attn_metadatas=Mock(return_value=[{}])
    )
    obj._capture_descs = {MODES.FULL: [BatchDescriptor(MODES.FULL, 4, 4), BatchDescriptor(MODES.FULL, 8, 4)]}
    obj.graphs = {}
    return obj


@pytest.mark.parametrize("updatable", [False, True])
def test_capture_and_dispatch_use_matching_k(updatable):
    obj = graph_manager(updatable)
    forward = Mock()
    obj.capture(
        forward, object(), SimpleNamespace(seq_lens_cpu=[0] * 4), object(), [], object(), specialize_spec_tokens=True
    )
    assert len(obj.graphs) == (4 if updatable else 2)
    for call in forward.call_args_list:
        desc = call.args[2]
        assert desc.cg_mode == MODES.NONE
        assert call.kwargs["num_speculative_steps"] == getattr(desc, "num_speculative_tokens", 4)
    desc = BatchDescriptor(MODES.FULL, 4, 4)
    for k in [4, 2, 0, 1, 3, 4]:
        selected = obj.specialize_spec_tokens(desc, k)
        expected_full = k in ({2, 4} if updatable else {4})
        assert (selected.cg_mode == MODES.FULL) is expected_full
        if expected_full:
            assert selected in obj.graphs


def test_graph_miss_falls_back_to_eager_and_piecewise_is_preserved():
    obj = graph_manager()
    assert obj.specialize_spec_tokens(BatchDescriptor(MODES.FULL, 16, 4), 2).cg_mode == MODES.NONE
    desc = BatchDescriptor(MODES.PIECEWISE, 4, 4)
    assert obj.specialize_spec_tokens(desc, 2) is desc


def test_specialized_descriptor_can_select_another_captured_k():
    obj = graph_manager()
    desc = SpeculatorDescriptor(MODES.FULL, 4, 4, 2)
    other = replace(desc, num_speculative_tokens=4)
    obj.graphs = {desc: object(), other: object()}
    assert obj.specialize_spec_tokens(desc, 4) == other
    missing = obj.specialize_spec_tokens(desc, 3)
    assert missing.cg_mode == MODES.NONE
    assert missing.num_speculative_tokens == 3


def test_recapture_keeps_specialized_descriptor_count():
    obj = graph_manager()
    args: tuple[Any, ...] = (Mock(), object(), SimpleNamespace(seq_lens_cpu=[0] * 4), object(), [], object())
    obj.capture(*args, specialize_spec_tokens=True)
    descriptors = obj._capture_descs.copy()
    obj.graphs.clear()
    obj.capture(*args, specialize_spec_tokens=True)
    assert obj._capture_descs == descriptors
    assert len(obj.graphs) == 4


@pytest.mark.parametrize("rebuild_returns_none", [False, True])
def test_absent_decode_metadata_returns_iterable(rebuild_returns_none):
    obj = speculator()
    if rebuild_returns_none:
        obj.attn_architecture = "GQA"
        obj.input_batch = SimpleNamespace(num_reqs=2, seq_lens_cpu_upper_bound=[9, 10])
        obj._build_uniform_attn_metadata = lambda **_: None
        metadata = {"draft": object()}
    else:
        metadata = None
    assert obj._init_decode_draft_attn_metadatas(metadata, 4, 2) == []


def test_replay_builds_metadata_for_selected_k():
    obj = graph_manager()
    obj._updatable_graph_replay = Mock(return_value="replayed")
    desc = SpeculatorDescriptor(MODES.FULL, 4, 4, 2)
    assert obj.run_fullgraph(desc) == "replayed"
    obj.speculator.build_draft_attn_metadatas.assert_called_once_with(4, 4, False, num_speculative_steps=2)


def test_updatable_replay_resolves_only_selected_k_tasks():
    obj = graph_manager()
    obj.update_stream = Mock()
    desc = SpeculatorDescriptor(MODES.FULL, 4, 4, 2)
    graph = FakeUpdatableGraph()
    obj.graphs[desc] = graph
    obj.speculator.build_fia_params = Mock(return_value=[{"step": 1}])
    assert obj._updatable_graph_replay(desc, [{"draft": object()}]) == "replayed"
    assert obj.speculator.build_fia_params.call_args.kwargs == {"num_speculative_steps": 2}
    graph.resolve_tasks.assert_called_once_with([{"step": 1}])
    graph.update.assert_called_once_with(obj.update_stream, "tasks")


def test_legacy_graph_capture_does_not_pass_new_keyword():
    class LegacyGraphParent:
        pass

    # Resolve the capability check against a parent without #57053's method.
    cls = load_class(
        SOURCE_DIR / "autoregressive/aclgraph.py",
        "AutoRegressiveAclGraphManager",
        {"capture"},
        LegacyGraphParent,
        {
            "replace": replace,
            "SpeculatorCudaGraphManager": LegacyGraphParent,
            "CudaGraphManager": CaptureParent,
            "communicator_switch": nullcontext,
            "model_capture_wrapper": lambda *_: nullcontext(),
            "prepare_inputs_to_capture": Mock(),
        },
    )
    obj = cls()
    obj.is_draft_model_prefill = False
    obj.max_num_reqs = 4
    obj.dp_size = 1
    obj.speculator = object()
    obj.graphs = {}
    obj._capture_descs = {MODES.FULL: [BatchDescriptor(MODES.FULL, 4, 4)]}
    forward = Mock()
    obj.capture(forward, object(), SimpleNamespace(seq_lens_cpu=[0] * 4), object(), [], object())
    assert forward.call_args.kwargs == {}
