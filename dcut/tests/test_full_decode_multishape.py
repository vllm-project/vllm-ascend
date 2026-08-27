# SPDX-License-Identifier: Apache-2.0

import ast
import sys
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

FULL_GRAPH_PATH = Path(__file__).resolve().parents[1] / "patch_full_graph.py"
RUNNER_PATH = Path(__file__).resolve().parents[1] / "patch_runner.py"
PROFILE_PATH = Path(__file__).resolve().parents[1] / "dcut_profile.py"
GDN_FORWARD_PATH = Path(__file__).resolve().parents[1] / "gdn_forward_v023.py"
GDN_PATCH_PATH = Path(__file__).resolve().parents[1] / "patch_gdn_v023.py"
PIECEWISE_PATH = Path(__file__).resolve().parents[1] / "patch_piecewise.py"


def _load_descriptor_map():
    tree = ast.parse(FULL_GRAPH_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "_DescriptorGraphParamMap"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Iterator": Iterator,
        "MutableMapping": MutableMapping,
    }
    exec(compile(module, str(FULL_GRAPH_PATH), "exec"), namespace)
    return namespace["_DescriptorGraphParamMap"]


def test_graph_params_are_isolated_for_same_q_and_different_layouts() -> None:
    graph_param_map = _load_descriptor_map()
    active_descriptor = {"value": "uniform"}
    graph_param_map._descriptor = staticmethod(
        lambda: active_descriptor["value"]
    )
    values = graph_param_map({128: []}, list_values=True)

    values[128].append("uniform-handle")
    active_descriptor["value"] = "ragged"
    assert values[128] == []
    values[128].append("ragged-handle")

    active_descriptor["value"] = "uniform"
    assert values[128] == ["uniform-handle"]
    active_descriptor["value"] = "ragged"
    assert values[128] == ["ragged-handle"]
    assert list(values) == [128]


def test_full_decode_contract_keeps_prefill_eager_and_draft_uniform() -> None:
    full_graph = FULL_GRAPH_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "CUDAGraphMode.FULL_DECODE_ONLY" in full_graph
    assert "original_initialize" in full_graph
    assert "self.add_cudagraph_key(CUDAGraphMode.FULL, ragged)" in full_graph
    assert "uniform_descriptor = replace(descriptor, uniform=True)" in full_graph

    assert "if not is_all_decode or not is_all_spec_decode:" in runner
    assert "force_eager = True" in runner
    assert "_dcut_nonuniform_full_batch" in runner
    assert "_dcut_is_ragged_full_capture(" in runner
    assert "_dcut_call_dummy_without_drafter(" in runner
    assert "max_fia_rows = int(request_capacity) + 1" in runner
    assert "_dcut_ragged_full_fia_block_tables" in runner
    assert (
        "self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL"
        not in runner
    )
    assert "_dcut_zero_draft_handoffs_for_proposal" in runner
    assert "_patch_live_fia_graph_params()" in full_graph


def test_ragged_full_capture_uses_decode_geometry_and_live_fia_rows() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "base, remainder = divmod(int(num_tokens), int(num_reqs))" in runner
    assert "num_scheduled_tokens_np[:remainder] += 1" in runner
    assert "self.uniform_decode_query_len" in runner
    pad_patch = runner[
        runner.index("    def _pad_query_start_loc_for_fia") :
        runner.index("    def _build_attention_metadata")
    ]
    assert "use_fixed_drafter_fia" in pad_patch
    assert "return request_capacity + 1" in pad_patch
    target_fia_patch = pad_patch[
        pad_patch.index("        # GDN keeps its own fixed-capacity") :
    ]
    assert "fia_rows = _dcut_ragged_full_fia_rows(" in pad_patch
    assert "if fia_rows > num_reqs:" in target_fia_patch
    assert (
        "query_start_loc.np[num_reqs + 1] = num_tokens_padded"
        in target_fia_patch
    )
    assert (
        "query_start_loc.np[request_capacity + 1]"
        not in target_fia_patch
    )
    assert "num_reqs + 1 : request_capacity + 1" not in target_fia_patch
    assert "gdn_query_start_loc" not in target_fia_patch

    metadata_patch = runner[
        runner.index("    def _build_attention_metadata") :
        runner.index("    def _dummy_run")
    ]
    assert "max_fia_rows = int(request_capacity) + 1" in metadata_patch
    assert "stable = storage[:fia_rows]" in metadata_patch
    assert "_dcut_pad_fia_dummy_seq_len(" in metadata_patch
    assert "_dcut_pad_fixed_drafter_fia_seq_lens(" in metadata_patch
    assert "route = \"drafter\"" in metadata_patch
    assert "route = \"target\"" in metadata_patch
    assert "metadata.block_tables = stable" in metadata_patch


def test_ragged_full_fia_uses_only_one_optional_dummy_row() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    names = {
        "_dcut_ragged_full_fia_rows",
        "_dcut_pad_fia_dummy_seq_len",
        "_dcut_pad_fixed_drafter_fia_seq_lens",
    }
    nodes = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    assert {item.name for item in nodes} == names
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"DUMMY_FIA_KV_SEQ_LEN": 1}
    exec(compile(module, str(RUNNER_PATH), "exec"), namespace)
    fia_rows = namespace["_dcut_ragged_full_fia_rows"]
    pad_dummy = namespace["_dcut_pad_fia_dummy_seq_len"]
    pad_fixed_drafter = namespace["_dcut_pad_fixed_drafter_fia_seq_lens"]

    assert fia_rows(96, 128, 32, 96) == 33
    assert fia_rows(128, 128, 32, 96) == 32
    assert fia_rows(96, 128, 96, 96) == 97
    try:
        fia_rows(96, 128, 97, 96)
    except RuntimeError as exc:
        assert "exceeds graph capacity" in str(exc)
    else:
        raise AssertionError("expected request-capacity validation")

    class FakeTensorView:
        def __init__(self, values, selection):
            self.values = values
            self.selection = selection

        def fill_(self, value):
            start, stop, step = self.selection.indices(len(self.values))
            for index in range(start, stop, step):
                self.values[index] = value

    class FakeTensor:
        def __init__(self, values):
            self.values = values
            self.shape = (len(values),)

        def __getitem__(self, selection):
            return FakeTensorView(self.values, selection)

        def tolist(self):
            return self.values.copy()

    seq_lens = FakeTensor([128, 256, 0, 0, 0])
    metadata = SimpleNamespace(
        seq_lens_list=[128, 256, 0, 0, 0],
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
    )
    pad_dummy(metadata, num_reqs=2, fia_rows=3)

    assert metadata.seq_lens_list == [128, 256, 1, 0, 0]
    assert metadata.seq_lens.tolist() == [128, 256, 1, 0, 0]

    fixed_seq_lens = FakeTensor([128, 256, 0, 0, 0])
    fixed_metadata = SimpleNamespace(
        seq_lens_list=[128, 256, 0, 0, 0],
        seq_lens=fixed_seq_lens,
        seq_lens_cpu=fixed_seq_lens,
    )
    pad_fixed_drafter(fixed_metadata, num_reqs=2, request_capacity=4)

    assert fixed_metadata.seq_lens_list == [128, 256, 1, 1, 1]
    assert fixed_metadata.seq_lens.tolist() == [128, 256, 1, 1, 1]


def test_full_gdn_contract_uses_fixed_metadata_and_target_weights() -> None:
    full_graph = FULL_GRAPH_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    gdn_forward = GDN_FORWARD_PATH.read_text(encoding="utf-8")
    gdn_patch = GDN_PATCH_PATH.read_text(encoding="utf-8")

    for weight in ("conv1d.weight", "A_log", "dt_bias"):
        assert weight in full_graph
    assert "target_module_ids" in full_graph
    assert "tensor.data_ptr()" in full_graph
    assert "_dcut_prepare_gdn_piecewise_replay" in runner
    assert "_dcut_prepare_gdn_graph_capture" in runner
    assert "if ragged_full_capture_dummy:" in runner
    assert "expected_capacity," in runner
    assert "self.uniform_decode_query_len" in runner
    assert "graph_request_capacity = getattr(" in runner
    assert "expected_capacity = min(" in runner
    assert "int(num_tokens_padded)" in runner
    assert "_dcut_gdn_full_graph_safe = True" in runner
    assert '"_dcut_gdn_full_graph_safe"' in gdn_forward
    assert "full_graph_spec_bufs" in gdn_forward
    assert "full_graph_safe = getattr(" in gdn_patch
    assert "if piecewise_graph_safe or full_graph_safe:" in gdn_patch
    assert "return graphable_spec_forward(" in gdn_patch
    assert "qwen_gdn_attention_core custom op" in gdn_patch
    assert "captured ragged FULL GDN graph through the " in runner


def test_sink_fia_task_params_rebind_to_live_block_table_view() -> None:
    tree = ast.parse(FULL_GRAPH_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_dcut_rebind_live_fia_block_tables"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(FULL_GRAPH_PATH), "exec"), namespace)
    rebind = namespace["_dcut_rebind_live_fia_block_tables"]

    first = SimpleNamespace(block_tables="live-first")
    second = SimpleNamespace(block_tables="live-second")
    captured = [
        ("q0", "k0", "v0", "captured-0", "tail", "layer.1"),
        ("q1", "k1", "v1", "captured-1", "tail", None),
    ]
    rebind(
        captured,
        {
            "layer.0": first,
            "layer.1": second,
        },
    )

    assert captured[0][3] == "live-second"
    assert captured[1][3] == "live-second"
    assert captured[0][:3] == ("q0", "k0", "v0")
    assert captured[0][4:] == ("tail", "layer.1")


def test_live_fia_patch_wraps_attention_implementation_class() -> None:
    tree = ast.parse(FULL_GRAPH_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_patch_live_fia_graph_params"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)

    observed = []

    class AttentionImpl:
        @staticmethod
        def update_graph_params(*args, **kwargs):
            observed.append((args, kwargs))
            return "native-update"

    class AttentionBackend:
        @staticmethod
        def get_impl_cls():
            return AttentionImpl

    root_module = ModuleType("vllm_ascend")
    root_module.__path__ = []
    attention_package = ModuleType("vllm_ascend.attention")
    attention_package.__path__ = []
    compilation_package = ModuleType("vllm_ascend.compilation")
    compilation_package.__path__ = []
    forward_context_module = ModuleType(
        "vllm_ascend.ascend_forward_context"
    )
    forward_context_module._EXTRA_CTX = SimpleNamespace(sinks=None)
    attention_module = ModuleType("vllm_ascend.attention.attention_v1")
    attention_module.AscendAttentionBackend = AttentionBackend
    attention_utils_module = ModuleType("vllm_ascend.attention.utils")
    attention_utils_module.using_paged_attention = lambda *_args: False
    acl_graph_module = ModuleType("vllm_ascend.compilation.acl_graph")
    acl_graph_module.get_graph_params = lambda: SimpleNamespace()

    namespace = {
        "logger": SimpleNamespace(info=lambda *_args, **_kwargs: None),
        "_dcut_rebind_live_fia_block_tables": lambda *_args: None,
    }
    exec(compile(module, str(FULL_GRAPH_PATH), "exec"), namespace)
    patch_live_fia = namespace["_patch_live_fia_graph_params"]

    modules = {
        "vllm_ascend": root_module,
        "vllm_ascend.attention": attention_package,
        "vllm_ascend.compilation": compilation_package,
        "vllm_ascend.ascend_forward_context": forward_context_module,
        "vllm_ascend.attention.attention_v1": attention_module,
        "vllm_ascend.attention.utils": attention_utils_module,
        "vllm_ascend.compilation.acl_graph": acl_graph_module,
    }
    with patch.dict(sys.modules, modules):
        patch_live_fia()
        patched_update = AttentionImpl.update_graph_params
        patch_live_fia()

    assert not hasattr(AttentionBackend, "_dcut_live_fia_rows_patched")
    assert AttentionImpl._dcut_live_fia_rows_patched is True
    assert patched_update is AttentionImpl.update_graph_params
    forward_context = SimpleNamespace(
        batch_descriptor=None,
        cudagraph_runtime_mode=None,
    )
    assert (
        patched_update(None, forward_context, 128, object())
        == "native-update"
    )
    assert len(observed) == 1


def test_ragged_full_capacity_is_fixed_by_token_bucket() -> None:
    tree = ast.parse(FULL_GRAPH_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_dcut_ragged_full_request_capacity"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {}
    exec(compile(module, str(FULL_GRAPH_PATH), "exec"), namespace)
    capacity = namespace["_dcut_ragged_full_request_capacity"]

    assert capacity(32, 96) == 32
    assert capacity(64, 96) == 64
    assert capacity(128, 96) == 96
    assert capacity(512, 96) == 96


def test_profile_only_marks_the_full_rectangle_uniform() -> None:
    profile = PROFILE_PATH.read_text(encoding="utf-8")

    assert "profile_uniform_decode = (" in profile
    assert "force_uniform_decode=profile_uniform_decode" in profile
    assert "force_uniform_decode=True" not in profile
    assert "self._pad_query_start_loc_for_fia(" in profile
    assert "self.gdn_query_start_loc.np[num_reqs + 1 :].fill(" in profile


def test_full_only_hooks_are_inert_in_piecewise_mode() -> None:
    full_graph = FULL_GRAPH_PATH.read_text(encoding="utf-8")
    gdn_patch = GDN_PATCH_PATH.read_text(encoding="utf-8")
    piecewise = PIECEWISE_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert "self.cudagraph_mode != CUDAGraphMode.PIECEWISE" in piecewise
    assert "configured_mode != \"FULL_DECODE_ONLY\"" in gdn_patch
    assert (
        "_patch_graph_params_by_descriptor()"
        in full_graph[full_graph.index("CUDAGraphMode.FULL_DECODE_ONLY") :]
    )
    assert '== "PIECEWISE"' in runner
    assert "clear_unused_rows=True" in runner


@dataclass(frozen=True)
class _Descriptor:
    uniform: bool


def _load_drafter_setup():
    tree = ast.parse(FULL_GRAPH_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_dcut_setup_full_decode_drafter"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "replace": replace,
        "_dcut_full_decode_multishape_enabled": lambda config: True,
        "_dcut_has_prefill": (
            lambda runner, scheduler_output, zero_draft_handoffs: (
                scheduler_output.has_prefill
                and not zero_draft_handoffs
            )
        ),
        "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
    }
    exec(compile(module, str(FULL_GRAPH_PATH), "exec"), namespace)
    return namespace["_dcut_setup_full_decode_drafter"]


def test_drafter_routes_ragged_decode_to_uniform_and_prefill_to_eager() -> None:
    setup_drafter = _load_drafter_setup()

    class CUDAGraphMode:
        FULL_DECODE_ONLY = "FULL_DECODE_ONLY"
        NONE = "NONE"

    vllm_module = ModuleType("vllm")
    config_module = ModuleType("vllm.config")
    config_module.CUDAGraphMode = CUDAGraphMode
    vllm_module.config = config_module

    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
    )
    observed = []

    class Drafter:
        use_cuda_graph = True
        fail = False

        def _propose(self, *args, **kwargs):
            observed.append(
                (
                    kwargs["target_model_batch_desc"].uniform,
                    self.use_cuda_graph,
                    getattr(
                        runner,
                        "_dcut_drafter_from_nonuniform_decode",
                        False,
                    ),
                )
            )
            if self.fail:
                raise RuntimeError("injected proposer failure")
            return "ok"

    drafter = Drafter()
    with patch.dict(
        sys.modules,
        {"vllm": vllm_module, "vllm.config": config_module},
    ):
        setup_drafter(runner, drafter)
        assert drafter._propose(
            target_model_batch_desc=_Descriptor(uniform=False),
            scheduler_output=SimpleNamespace(has_prefill=False),
        ) == "ok"
        assert not hasattr(
            runner,
            "_dcut_drafter_from_nonuniform_decode",
        )
        assert drafter._propose(
            target_model_batch_desc=_Descriptor(uniform=False),
            scheduler_output=SimpleNamespace(has_prefill=True),
        ) == "ok"

        runner._dcut_zero_draft_handoffs_for_proposal = frozenset(
            {"handoff"}
        )
        assert drafter._propose(
            target_model_batch_desc=_Descriptor(uniform=False),
            scheduler_output=SimpleNamespace(has_prefill=True),
        ) == "ok"
        del runner._dcut_zero_draft_handoffs_for_proposal

        drafter.fail = True
        for has_prefill in (False, True):
            try:
                drafter._propose(
                    target_model_batch_desc=_Descriptor(uniform=False),
                    scheduler_output=SimpleNamespace(
                        has_prefill=has_prefill
                    ),
                )
            except RuntimeError as exc:
                assert str(exc) == "injected proposer failure"
            else:
                raise AssertionError("expected injected proposer failure")
            assert drafter.use_cuda_graph is True
            assert not hasattr(
                runner,
                "_dcut_drafter_from_nonuniform_decode",
            )

    assert observed == [
        (True, True, True),
        (False, False, False),
        (True, True, True),
        (True, True, True),
        (False, False, False),
    ]
    assert drafter.use_cuda_graph is True


def test_runner_suppresses_drafter_for_ragged_full_capture() -> None:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    names = {
        "_dcut_call_dummy_without_drafter",
        "_dcut_is_ragged_full_capture",
    }
    nodes = [
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name in names
    ]
    assert {item.name for item in nodes} == names
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_dcut_full_decode_multishape_enabled": lambda config: True,
    }
    exec(compile(module, str(RUNNER_PATH), "exec"), namespace)

    is_ragged_full = namespace["_dcut_is_ragged_full_capture"]
    call_dummy = namespace["_dcut_call_dummy_without_drafter"]
    full_mode = SimpleNamespace(name="FULL")
    runner = SimpleNamespace(vllm_config=object(), drafter=object())

    assert is_ragged_full(runner, full_mode, False, True) is True
    assert is_ragged_full(runner, full_mode, True, True) is False
    assert is_ragged_full(
        runner,
        SimpleNamespace(name="PIECEWISE"),
        False,
        True,
    ) is False
    assert is_ragged_full(runner, full_mode, False, False) is False

    original_drafter = runner.drafter
    observed = []

    def dummy(active_runner, num_tokens, marker=None):
        observed.append((active_runner.drafter, num_tokens, marker))
        return "target-ran"

    assert call_dummy(
        runner,
        dummy,
        448,
        (),
        {"marker": "ragged"},
        True,
    ) == "target-ran"
    assert observed == [(None, 448, "ragged")]
    assert runner.drafter is original_drafter

    try:
        call_dummy(
            runner,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected failure")
            ),
            448,
            (),
            {},
            True,
        )
    except RuntimeError as exc:
        assert str(exc) == "injected failure"
    else:
        raise AssertionError("expected injected failure")
    assert runner.drafter is original_drafter
