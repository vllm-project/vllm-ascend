# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import torch

DCUT_DIR = Path(__file__).resolve().parents[1]


def _load_functions(path: Path, names: set[str], namespace: dict):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *functions,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _load_utils():
    names = {
        "_env_flag",
        "_dcut_process_probs_stage",
        "_dcut_reuse_argmax_enabled",
        "_dcut_in_graph_capture",
        "_dcut_graph_capture_mode_name",
        "_dcut_should_collect_draft_probs",
        "_dcut_selected_token_probs",
        "_dcut_greedy_sample_with_selected_probs",
        "_dcut_can_reuse_argmax_for_probs",
        "_dcut_selected_probs_from_graph",
        "_dcut_selected_probs_from_reused_logits",
    }
    return _load_functions(
        DCUT_DIR / "utils.py",
        names,
        {
            "os": os,
            "torch": torch,
            "get_tp_group": lambda: SimpleNamespace(world_size=1),
            "ENV_CONFIG": "VLLM_DCUT_CONFIG",
            "ENV_DISABLE": "VLLM_DCUT_DISABLE",
            "ENV_PROCESS_PROBS_STAGE": "VLLM_DCUT_PROCESS_PROBS_STAGE",
            "ENV_REUSE_ARGMAX": "VLLM_DCUT_REUSE_ARGMAX",
        },
    )


def _load_controller_methods(names: set[str]):
    path = DCUT_DIR / "verify_adaptive_controller.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    controller = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VerifyAdaptiveController"
    )
    methods = [
        node
        for node in controller.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    module = ast.Module(body=methods, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def test_deferred_processing_and_argmax_reuse_defaults(monkeypatch) -> None:
    utils = _load_utils()
    monkeypatch.delenv("VLLM_DCUT_PROCESS_PROBS_STAGE", raising=False)
    monkeypatch.delenv("VLLM_DCUT_REUSE_ARGMAX", raising=False)

    assert utils["_dcut_process_probs_stage"]() == "pre_truncate"
    assert utils["_dcut_reuse_argmax_enabled"]() is True

    monkeypatch.setenv("VLLM_DCUT_PROCESS_PROBS_STAGE", "post-sample")
    monkeypatch.setenv("VLLM_DCUT_REUSE_ARGMAX", "0")
    assert utils["_dcut_process_probs_stage"]() == "post_sample"
    assert utils["_dcut_reuse_argmax_enabled"]() is False


def test_selected_probs_share_the_stock_two_tp_gathers() -> None:
    local_logits = torch.tensor(
        [[0.0, 2.0, 1.0], [5.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    remote_logits = torch.tensor(
        [[4.0, 0.0, -1.0], [4.0, 6.0, 1.0]],
        dtype=torch.float32,
    )

    class FakeTpGroup:
        world_size = 2
        rank_in_group = 0

        def __init__(self) -> None:
            self.gathered_shapes = []

        def all_gather(self, tensor, dim=-1):
            assert dim == -1
            self.gathered_shapes.append(tuple(tensor.shape))
            if tensor.is_floating_point():
                remote_max = remote_logits.amax(dim=-1)
                remote_lse = remote_logits.logsumexp(dim=-1)
                remote = torch.stack((remote_max, remote_lse), dim=-1)
            else:
                remote = (
                    remote_logits.argmax(dim=-1)
                    + remote_logits.shape[-1]
                ).unsqueeze(-1)
            return torch.cat((tensor, remote), dim=-1)

    group = FakeTpGroup()
    sample = _load_functions(
        DCUT_DIR / "utils.py",
        {"_dcut_greedy_sample_with_selected_probs"},
        {
            "torch": torch,
            "get_tp_group": lambda: group,
        },
    )["_dcut_greedy_sample_with_selected_probs"]

    token_ids, selected_probs = sample(local_logits)
    global_logits = torch.cat((local_logits, remote_logits), dim=-1)
    expected_ids = global_logits.argmax(dim=-1)
    expected_probs = global_logits.softmax(dim=-1).gather(
        -1,
        expected_ids.unsqueeze(-1),
    ).squeeze(-1)

    assert group.gathered_shapes == [(2, 2), (2, 1)]
    torch.testing.assert_close(token_ids, expected_ids)
    torch.testing.assert_close(selected_probs, expected_probs)


def test_full_graph_capture_forces_probs_when_runtime_flag_is_disabled(
    monkeypatch,
) -> None:
    utils = _load_utils()
    drafter = SimpleNamespace(
        needs_draft_probs=False,
        method="dflash",
        parallel_drafting=True,
    )
    monkeypatch.delenv("VLLM_DCUT_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_DCUT_DISABLE", raising=False)
    utils["_dcut_in_graph_capture"] = lambda: True
    utils["_dcut_graph_capture_mode_name"] = lambda: "FULL"

    assert utils["_dcut_should_collect_draft_probs"](drafter) is False

    monkeypatch.setenv("VLLM_DCUT_CONFIG", "/tmp/dcut-config.json")
    utils["_dcut_graph_capture_mode_name"] = lambda: "PIECEWISE"
    assert utils["_dcut_should_collect_draft_probs"](drafter) is False
    utils["_dcut_graph_capture_mode_name"] = lambda: "FULL"
    assert utils["_dcut_should_collect_draft_probs"](drafter) is True

    monkeypatch.setenv("VLLM_DCUT_DISABLE", "1")
    assert utils["_dcut_should_collect_draft_probs"](drafter) is False
    monkeypatch.delenv("VLLM_DCUT_DISABLE")

    utils["_dcut_in_graph_capture"] = lambda: False
    assert utils["_dcut_should_collect_draft_probs"](drafter) is False

    monkeypatch.delenv("VLLM_DCUT_CONFIG")
    drafter.needs_draft_probs = True
    assert utils["_dcut_should_collect_draft_probs"](drafter) is True

    drafter.method = "unsupported"
    drafter.parallel_drafting = False
    assert utils["_dcut_should_collect_draft_probs"](drafter) is False


def test_draft_probability_paths_use_capture_aware_gate() -> None:
    proposer = (DCUT_DIR / "patch_proposer.py").read_text(encoding="utf-8")
    drafter = (DCUT_DIR / "drafter.py").read_text(encoding="utf-8")

    assert (
        "needs_draft_probs = bool(os.environ.get(ENV_CONFIG))" in proposer
    )
    assert "return _dcut_should_collect_draft_probs(self)" in proposer
    assert "if _dcut_should_collect_draft_probs(drafter)" in drafter
    assert "_dcut_selected_probs_for_output(" in proposer
    assert proposer.index(
        "selected_probs = _dcut_selected_probs_for_output("
    ) < proposer.index("if in_graph_capture:")
    assert "if not _dcut_should_collect_draft_probs(self)" in drafter
    assert "_dcut_attach_graph_owner(drafter)" in drafter
    assert "capture_pending" in proposer
    assert "_dcut_graph_owner_from_runnable(runnable)" in proposer
    assert "ACLGraphWrapper.__init__ = __init__" in proposer
    assert "_dcut_register_graph_selected_probs(" in proposer
    assert "_dcut_prepare_dflash_graph_context(" in proposer
    assert "proposer._dflash_num_context = restore_dflash_context" in proposer


def test_output_aligned_probs_follow_returned_draft_ids() -> None:
    gather = _load_functions(
        DCUT_DIR / "patch_proposer.py",
        {"_dcut_selected_probs_for_output"},
        {
            "_dcut_selected_token_probs": lambda logits, token_ids: (
                logits.softmax(dim=-1)
                .gather(-1, token_ids.long().unsqueeze(-1))
                .squeeze(-1)
            ),
        },
    )["_dcut_selected_probs_for_output"]
    logits = torch.tensor(
        [[4.0, 1.0, -1.0], [0.2, 0.3, 3.0]],
        dtype=torch.float32,
    )
    returned_ids = torch.tensor([[2, 0]], dtype=torch.int64)

    actual = gather(logits, returned_ids)
    expected = logits.softmax(dim=-1).gather(
        -1,
        returned_ids.reshape(-1, 1),
    ).view_as(returned_ids)

    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(
        actual.reshape(-1),
        logits.softmax(dim=-1).amax(dim=-1),
    )


class _Drafter:
    method = "dflash"
    _dcut_run_merged_patched = True


def test_reused_logits_match_selected_softmax_probability(monkeypatch) -> None:
    utils = _load_utils()
    monkeypatch.delenv("VLLM_DCUT_REUSE_ARGMAX", raising=False)
    logits = torch.tensor(
        [[1.0, 3.0, -2.0], [2.0, 0.5, 1.0]],
        dtype=torch.float32,
    )
    token_ids = torch.tensor([[1, 0]])
    drafter = _Drafter()
    drafter._dcut_last_logits_for_probs = logits
    drafter._dcut_last_draft_ran_python = True

    actual = utils["_dcut_selected_probs_from_reused_logits"](
        drafter,
        token_ids,
    )
    expected = logits.softmax(dim=-1).gather(
        -1,
        token_ids.reshape(-1, 1),
    ).reshape_as(token_ids)

    assert actual is not None
    torch.testing.assert_close(actual, expected)


def test_reused_logits_reject_mapped_vocab_and_tp_shards(monkeypatch) -> None:
    utils = _load_utils()
    monkeypatch.delenv("VLLM_DCUT_REUSE_ARGMAX", raising=False)
    logits = torch.tensor([[1.0, 3.0, -2.0]], dtype=torch.float32)
    token_ids = torch.tensor([[1]])
    drafter = _Drafter()
    drafter._dcut_last_logits_for_probs = logits
    drafter._dcut_last_draft_ran_python = True
    drafter.model = SimpleNamespace(
        draft_id_to_target_id=torch.arange(3),
    )

    mapped_vocab_probs = utils[
        "_dcut_selected_probs_from_reused_logits"
    ](drafter, token_ids)

    assert mapped_vocab_probs is None

    drafter.model = SimpleNamespace(draft_id_to_target_id=None)
    utils["get_tp_group"] = lambda: SimpleNamespace(world_size=2)
    sharded_probs = utils[
        "_dcut_selected_probs_from_reused_logits"
    ](drafter, token_ids)

    assert sharded_probs is None


def test_graph_selected_probs_are_selected_by_output_bucket() -> None:
    utils = _load_utils()
    small = torch.tensor([[0.8, 0.6]], dtype=torch.float32)
    large = torch.tensor(
        [[0.9, 0.7], [0.5, 0.3]],
        dtype=torch.float32,
    )
    draft_token_ids = torch.zeros((1, 2), dtype=torch.int64)
    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=False,
        _dcut_graph_selected_probs_ready=True,
        _dcut_graph_selected_probs_by_output_ptr={
            int(draft_token_ids.data_ptr()): small,
        },
        # Deliberately point the same-shape fallback at another bucket. The
        # fixed output address must win when multiple graphs share a shape.
        _dcut_graph_selected_probs_by_shape={
            (1, 2): large,
        },
        _dcut_graph_selected_probs_by_numel={
            2: large,
        },
    )

    actual = utils["_dcut_selected_probs_from_graph"](
        drafter,
        draft_token_ids,
    )

    assert actual is not None
    torch.testing.assert_close(actual, small)


def test_graph_selected_probs_prefer_exact_descriptor() -> None:
    utils = _load_utils()
    exact = torch.tensor([[0.81, 0.61]], dtype=torch.float32)
    wrong = torch.tensor([[0.22, 0.12]], dtype=torch.float32)
    draft_token_ids = torch.zeros((1, 2), dtype=torch.int64)
    descriptor = object()
    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=False,
        _dcut_graph_selected_probs_ready=True,
        _dcut_current_graph_descriptor=descriptor,
        _dcut_graph_selected_probs_by_descriptor={descriptor: exact},
        _dcut_graph_selected_probs_by_output_ptr={
            int(draft_token_ids.data_ptr()): wrong,
        },
        _dcut_graph_selected_probs_by_shape={(1, 2): wrong},
        _dcut_graph_selected_probs_by_numel={2: wrong},
    )

    actual = utils["_dcut_selected_probs_from_graph"](
        drafter,
        draft_token_ids,
    )

    assert actual is not None
    assert drafter._dcut_last_graph_prob_source == "graph_descriptor"
    torch.testing.assert_close(actual, exact)


def test_graph_descriptor_miss_rejects_unsafe_shape_fallback() -> None:
    utils = _load_utils()
    draft_token_ids = torch.zeros((1, 2), dtype=torch.int64)
    captured_descriptor = object()
    wrong = torch.tensor([[0.22, 0.12]], dtype=torch.float32)
    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=False,
        _dcut_graph_selected_probs_ready=True,
        _dcut_current_graph_descriptor=object(),
        _dcut_graph_selected_probs_by_descriptor={captured_descriptor: wrong},
        _dcut_graph_selected_probs_by_output_ptr={
            int(draft_token_ids.data_ptr()): wrong,
        },
        _dcut_graph_selected_probs_by_shape={(1, 2): wrong},
        _dcut_graph_selected_probs_by_numel={2: wrong},
    )

    actual = utils["_dcut_selected_probs_from_graph"](
        drafter,
        draft_token_ids,
    )

    assert actual is None
    assert drafter._dcut_last_graph_prob_source == "missing"


def test_graph_compatibility_key_is_disabled_when_ambiguous() -> None:
    store = _load_functions(
        DCUT_DIR / "patch_proposer.py",
        {"_dcut_store_unique_graph_buffer"},
        {},
    )["_dcut_store_unique_graph_buffer"]
    owner = SimpleNamespace()
    first = torch.zeros((1, 2))
    second = torch.ones((1, 2))

    store(owner, "buffers", "same-shape", first)
    assert owner.buffers["same-shape"] is first

    store(owner, "buffers", "same-shape", second)
    assert owner.buffers["same-shape"] is None


def test_graph_capture_registration_records_exact_descriptor() -> None:
    functions = _load_functions(
        DCUT_DIR / "patch_proposer.py",
        {
            "_dcut_store_unique_graph_buffer",
            "_dcut_register_graph_selected_probs",
        },
        {},
    )
    register = functions["_dcut_register_graph_selected_probs"]
    owner = SimpleNamespace()
    descriptor = object()
    output = torch.zeros((2, 3), dtype=torch.int64)
    probs = torch.full((2, 3), 0.5)

    assert register(owner, descriptor, output, probs) is True
    assert owner._dcut_graph_selected_probs_ready is True
    assert owner._dcut_graph_selected_probs_by_descriptor[descriptor] is probs
    assert (
        owner._dcut_graph_selected_probs_by_output_ptr[int(output.data_ptr())]
        is probs
    )
    assert owner._dcut_graph_selected_probs_by_shape[(2, 3)] is probs
    assert owner._dcut_graph_selected_probs_by_numel[6] is probs


def test_live_drafter_is_attached_to_aclgraph_wrapper() -> None:
    attach = _load_functions(
        DCUT_DIR / "drafter.py",
        {"_dcut_attach_graph_owner"},
        {},
    )["_dcut_attach_graph_owner"]
    runnable = SimpleNamespace(concrete_aclgraph_entries={})
    drafter = SimpleNamespace(_runnable=runnable)

    assert attach(drafter) is True
    assert runnable._dcut_descriptor_owner is drafter
    assert drafter._dcut_graph_owner_attached is True

    eager = SimpleNamespace(_runnable=lambda: None)
    assert attach(eager) is False


def test_dflash_graph_context_uses_standalone_fill_on_capture_and_replay() -> None:
    warnings = []
    prepare = _load_functions(
        DCUT_DIR / "patch_proposer.py",
        {"_dcut_prepare_dflash_graph_context"},
        {
            "PADDING_SLOT_ID": -1,
            "logger": SimpleNamespace(
                warning=lambda *args: warnings.append(args)
            ),
        },
    )["_dcut_prepare_dflash_graph_context"]
    descriptor = ("FULL", 8)
    slot_mapping = torch.arange(8, dtype=torch.int32)
    drafter = SimpleNamespace(
        method="dflash",
        _dflash_num_context=4,
        _context_slot_mapping_buffer=slot_mapping,
    )

    restore_context = prepare(
        drafter,
        descriptor,
        graph_context_tokens=8,
        capture_pending=True,
    )

    assert restore_context == 4
    assert drafter._dflash_num_context == 8
    assert drafter._dcut_dflash_context_tokens_by_descriptor[descriptor] == 8
    torch.testing.assert_close(
        slot_mapping[:4], torch.arange(4, dtype=torch.int32)
    )
    torch.testing.assert_close(
        slot_mapping[4:], torch.full((4,), -1, dtype=torch.int32)
    )

    # Emulate the graph wrapper's finally block, then replay with a new
    # actual context length against the descriptor's fixed captured length.
    drafter._dflash_num_context = restore_context
    slot_mapping.fill_(9)
    drafter._dflash_num_context = 6
    restore_context = prepare(
        drafter,
        descriptor,
        graph_context_tokens=None,
        capture_pending=False,
    )

    assert restore_context is None
    assert drafter._dflash_num_context == 6
    torch.testing.assert_close(
        slot_mapping[:6], torch.full((6,), 9, dtype=torch.int32)
    )
    torch.testing.assert_close(
        slot_mapping[6:], torch.full((2,), -1, dtype=torch.int32)
    )
    assert drafter._dcut_last_dflash_context_actual == 6
    assert drafter._dcut_last_dflash_context_captured == 8
    assert drafter._dcut_last_dflash_context_tail_masked is True
    assert len(warnings) == 1
    assert "standalone fill" in warnings[0][0]


def test_dflash_graph_rejects_context_larger_than_capture() -> None:
    prepare = _load_functions(
        DCUT_DIR / "patch_proposer.py",
        {"_dcut_prepare_dflash_graph_context"},
        {
            "PADDING_SLOT_ID": -1,
            "logger": SimpleNamespace(warning=lambda *args: None),
        },
    )["_dcut_prepare_dflash_graph_context"]
    descriptor = ("FULL", 4)
    drafter = SimpleNamespace(
        method="dflash",
        _dflash_num_context=5,
        _context_slot_mapping_buffer=torch.arange(8, dtype=torch.int32),
        _dcut_dflash_context_tokens_by_descriptor={descriptor: 4},
    )

    try:
        prepare(drafter, descriptor, None, capture_pending=False)
    except RuntimeError as error:
        assert "exceeds its captured length" in str(error)
    else:
        raise AssertionError("oversized DFlash graph context should fail")


def test_bound_draft_runnable_exposes_graph_owner() -> None:
    get_owner = _load_functions(
        DCUT_DIR / "patch_proposer.py",
        {"_dcut_graph_owner_from_runnable"},
        {},
    )["_dcut_graph_owner_from_runnable"]

    class Drafter:
        def compute_draft_token_ids(self):
            pass

        def _run_merged_draft(self):
            pass

    drafter = Drafter()

    assert get_owner(drafter._run_merged_draft) is drafter
    assert get_owner(lambda: None) is None
    assert get_owner(SimpleNamespace(__self__=SimpleNamespace())) is None


def test_pre_warmup_drops_only_graphs_without_probs() -> None:
    drop_stale = _load_functions(
        DCUT_DIR / "patch_worker.py",
        {"_dcut_drop_pre_warmup_draft_graphs"},
        {},
    )["_dcut_drop_pre_warmup_draft_graphs"]

    stale = object()
    retained = object()
    pending = object()
    entries = {
        stale: SimpleNamespace(aclgraph=object()),
        retained: SimpleNamespace(aclgraph=object()),
        pending: SimpleNamespace(aclgraph=None),
    }
    runnable = SimpleNamespace(concrete_aclgraph_entries=entries)
    drafter = SimpleNamespace(
        needs_draft_probs=True,
        _runnable=runnable,
        _dcut_graph_selected_probs_by_descriptor={retained: object()},
    )
    runner = SimpleNamespace(drafter=drafter)

    assert drop_stale(runner) == 1
    assert stale not in entries
    assert retained in entries
    assert pending in entries

    disabled = SimpleNamespace(
        drafter=SimpleNamespace(needs_draft_probs=False),
    )
    assert drop_stale(disabled) == 0


def test_eager_draft_does_not_reuse_graph_selected_probs() -> None:
    utils = _load_utils()
    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=True,
        _dcut_graph_selected_probs_ready=True,
        _dcut_graph_selected_probs_by_output_ptr={},
        _dcut_graph_selected_probs_by_shape={
            (1, 2): torch.ones((1, 2)),
        },
        _dcut_graph_selected_probs_by_numel={},
    )

    actual = utils["_dcut_selected_probs_from_graph"](
        drafter,
        torch.zeros((1, 2), dtype=torch.int64),
    )

    assert actual is None


def test_probability_rows_include_request_that_just_finished_prefill() -> None:
    functions = _load_functions(
        DCUT_DIR / "probs.py",
        {"_dcut_probability_req_ids", "_dcut_queue_probs"},
        {
            "_dcut_enable_drafter_probs": lambda runner: None,
            "_dcut_selected_probs_from_graph": lambda *args: None,
            "_dcut_selected_probs_from_reused_logits": lambda *args: None,
            "logger": SimpleNamespace(
                info=lambda *args: None,
                warning=lambda *args: None,
            ),
        },
    )

    class Event:
        recorded = 0

        def record(self):
            self.recorded += 1

    selected_probs = torch.tensor(
        [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]],
        dtype=torch.float32,
    )
    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=True,
        take_last_selected_probs=lambda: selected_probs,
    )
    event = Event()
    runner = SimpleNamespace(
        drafter=drafter,
        _draft_token_ids=torch.zeros((2, 3), dtype=torch.int64),
        _draft_token_req_ids=["decode", "just-finished-prefill"],
        _adaptive_probs_pending=False,
        _adaptive_probs_pinned=torch.zeros((2, 3)),
        _adaptive_probs_event=event,
        num_spec_tokens=3,
        input_batch=SimpleNamespace(
            num_reqs=2,
            req_ids=["decode", "just-finished-prefill"],
            num_computed_tokens_cpu=[10, 9],
            num_prompt_tokens=[10, 10],
        ),
    )

    functions["_dcut_queue_probs"](runner, zeros_only=False)

    assert runner._adaptive_req_ids == [
        "decode",
        "just-finished-prefill",
    ]
    assert runner._adaptive_active == {
        "decode",
        "just-finished-prefill",
    }
    torch.testing.assert_close(runner._adaptive_probs_pinned, selected_probs)
    assert event.recorded == 1


def test_adaptive_decision_signature_is_atomic() -> None:
    methods = _load_controller_methods({
        "set_adaptive_decision",
        "clear_adaptive_decision",
        "matches_adaptive_request_set",
    })
    controller = SimpleNamespace(
        _adaptive_draft_lens={"stale": 15},
        _adaptive_decision_req_ids=frozenset({"stale"}),
        _adaptive_decision_batch_size=1,
    )

    methods["set_adaptive_decision"](
        controller,
        ["request-0", "request-1"],
        [2, 3],
        2,
    )

    assert controller._adaptive_draft_lens == {
        "request-0": 2,
        "request-1": 3,
    }
    assert methods["matches_adaptive_request_set"](
        controller,
        ["request-1", "request-0"],
    )
    assert not methods["matches_adaptive_request_set"](
        controller,
        ["request-0", "new-request"],
    )

    methods["clear_adaptive_decision"](controller)
    assert controller._adaptive_draft_lens == {}
    assert not methods["matches_adaptive_request_set"](
        controller,
        ["request-0", "request-1"],
    )


def test_pre_truncate_waits_once_and_filters_finished_requests() -> None:
    process = _load_functions(
        DCUT_DIR / "probs.py",
        {"_maybe_process_adaptive_probs"},
        {},
    )["_maybe_process_adaptive_probs"]

    class Event:
        synchronized = 0

        def query(self):
            return False

        def synchronize(self):
            self.synchronized += 1

    class Controller:
        call = None

        def process_draft_output(self, **kwargs):
            self.call = kwargs

    event = Event()
    controller = Controller()
    runner = SimpleNamespace(
        _adaptive_probs_pending=True,
        _adaptive_probs_event=event,
        _dcut_skip_unready_probs=False,
        _dcut_debug_stats_enabled=True,
        _adaptive_probs_source="graph_descriptor",
        _adaptive_probs_generation=7,
        _adaptive_num_reqs=2,
        _adaptive_active={"keep", "finished"},
        _adaptive_req_ids=["keep", "finished"],
        _adaptive_probs_pinned=torch.ones((2, 3)),
        _verify_adaptive_controller=controller,
        input_batch=SimpleNamespace(
            req_ids=["keep", "new"],
            num_reqs=2,
        ),
    )

    process(runner, stage="pre_truncate")

    assert event.synchronized == 1
    assert runner._adaptive_probs_pending is False
    assert controller.call is not None
    assert controller.call["active_draft_req_ids"] == {"keep"}
    assert runner._adaptive_probs_last_consumed_source == "graph_descriptor"
    assert runner._adaptive_probs_last_consumed_generation == 7
    assert runner._adaptive_probs_last_consumed_mean_by_position == [
        1.0, 1.0, 1.0
    ]


def test_skip_unready_probability_expires_without_cross_step_reuse() -> None:
    process = _load_functions(
        DCUT_DIR / "probs.py",
        {"_maybe_process_adaptive_probs"},
        {},
    )["_maybe_process_adaptive_probs"]

    class Event:
        ready = False

        def query(self):
            return self.ready

    class Controller:
        cleared = 0
        processed = 0

        def clear_adaptive_decision(self):
            self.cleared += 1

        def process_draft_output(self, **kwargs):
            self.processed += 1

    event = Event()
    controller = Controller()
    runner = SimpleNamespace(
        _adaptive_probs_pending=True,
        _adaptive_probs_event=event,
        _dcut_skip_unready_probs=True,
        _adaptive_probs_expired=False,
        _adaptive_probs_source="graph_descriptor",
        _adaptive_probs_generation=3,
        _adaptive_num_reqs=1,
        _adaptive_active={"request-0"},
        _adaptive_req_ids=["request-0"],
        _adaptive_probs_pinned=torch.ones((1, 3)),
        _verify_adaptive_controller=controller,
        input_batch=SimpleNamespace(req_ids=["request-0"], num_reqs=1),
    )

    process(runner, stage="pre_truncate")

    assert runner._adaptive_probs_pending is True
    assert runner._adaptive_probs_expired is True
    assert controller.cleared == 1
    assert controller.processed == 0

    event.ready = True
    process(runner, stage="pre_truncate")

    assert runner._adaptive_probs_pending is False
    assert runner._adaptive_probs_expired is False
    assert runner._adaptive_probs_source == "expired"
    assert controller.cleared == 2
    assert controller.processed == 0


def test_prepare_clears_transient_graph_logits_pointer() -> None:
    prepare = _load_functions(
        DCUT_DIR / "probs.py",
        {"_dcut_prepare_prob_capture"},
        {},
    )["_dcut_prepare_prob_capture"]

    class Controller:
        cleared = 0

        def clear_adaptive_decision(self):
            self.cleared += 1

    drafter = SimpleNamespace(
        _dcut_last_draft_ran_python=True,
        _dcut_last_logits_for_probs=object(),
        _last_selected_probs=object(),
    )
    controller = Controller()
    runner = SimpleNamespace(
        drafter=drafter,
        _verify_adaptive_controller=controller,
    )

    prepare(runner, SimpleNamespace())

    assert drafter._dcut_last_draft_ran_python is False
    assert drafter._dcut_last_logits_for_probs is None
    assert drafter._last_selected_probs is None
    assert drafter._dcut_current_graph_descriptor is None
    assert drafter._dcut_last_graph_prob_source == "none"
    assert controller.cleared == 1


def test_no_low_batch_or_eager_fallback_was_migrated() -> None:
    sources = "\n".join(
        (DCUT_DIR / name).read_text(encoding="utf-8")
        for name in (
            "controller.py",
            "drafter.py",
            "globals.py",
            "patch_proposer.py",
            "patch_runner.py",
            "probs.py",
            "truncate.py",
            "utils.py",
        )
    )
    assert "MIN_SPEC_BATCH" not in sources
    assert "min_spec_batch" not in sources
    assert "use_cuda_graph = False" not in sources


def test_pending_probs_are_processed_before_truncation() -> None:
    source = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    execute_start = source.index("    def execute_model(")
    execute_end = source.index("    _orig_sample_tokens", execute_start)
    execute_source = source[execute_start:execute_end]

    process_at = execute_source.index("_maybe_process_adaptive_probs")
    truncate_at = execute_source.index("scheduler_output = _dcut_truncate")
    assert process_at < truncate_at


def test_scheduler_prefill_route_is_scoped_to_execute_model() -> None:
    route = _load_functions(
        DCUT_DIR / "patch_runner.py",
        {"_dcut_execute_with_gdn_prefill_route"},
        {},
    )["_dcut_execute_with_gdn_prefill_route"]
    runner = SimpleNamespace()
    observed = []

    def execute(active_runner, scheduler_output, intermediate_tensors):
        observed.append(active_runner._dcut_gdn_scheduler_has_prefill)
        return scheduler_output, intermediate_tensors

    result = route(runner, execute, "schedule", "intermediate", False)

    assert result == ("schedule", "intermediate")
    assert observed == [False]
    assert not hasattr(runner, "_dcut_gdn_scheduler_has_prefill")


def test_force_drafter_eager_keeps_target_graph_mode() -> None:
    forced = {"enabled": True}
    warnings = []
    patched = []
    setup = []
    enable = _load_functions(
        DCUT_DIR / "controller.py",
        {"_dcut_enable_drafter_probs"},
        {
            "ENV_FORCE_DRAFTER_EAGER": "VLLM_DCUT_FORCE_DRAFTER_EAGER",
            "_env_flag": lambda _name: forced["enabled"],
            "_dcut_patch_drafter_instance": patched.append,
            "_dcut_setup_full_decode_drafter": (
                lambda runner, drafter: setup.append((runner, drafter))
            ),
            "logger": SimpleNamespace(
                warning=lambda *args, **kwargs: warnings.append((args, kwargs))
            ),
        },
    )["_dcut_enable_drafter_probs"]

    drafter = SimpleNamespace(
        needs_draft_probs=False,
        use_cuda_graph=True,
        method="dflash",
        parallel_drafting=True,
    )
    runner = SimpleNamespace(
        _verify_adaptive_controller=object(),
        drafter=drafter,
        compilation_config=SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
        _dcut_logged_drafter_probs=True,
    )

    enable(runner)
    enable(runner)

    assert drafter.use_cuda_graph is False
    assert drafter.needs_draft_probs is True
    assert patched == [drafter, drafter]
    assert setup == [(runner, drafter), (runner, drafter)]
    assert len(warnings) == 1
    assert warnings[0][0][1] == "VLLM_DCUT_FORCE_DRAFTER_EAGER"
    assert warnings[0][0][2] == "FULL_DECODE_ONLY"

    forced["enabled"] = False
    drafter.use_cuda_graph = True
    enable(runner)
    assert drafter.use_cuda_graph is True

    globals_source = (DCUT_DIR / "globals.py").read_text(encoding="utf-8")
    assert (
        'ENV_FORCE_DRAFTER_EAGER = "VLLM_DCUT_FORCE_DRAFTER_EAGER"'
        in globals_source
    )


def test_no_cut_keeps_gdn_patch_but_skips_control_plane() -> None:
    controller = (DCUT_DIR / "controller.py").read_text(encoding="utf-8")
    runner = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    install = (DCUT_DIR / "install.py").read_text(encoding="utf-8")

    assert "if _env_flag(ENV_DISABLE):" in controller
    assert "D-Cut GDN operator patches remain active" in controller
    assert "not _env_flag(ENV_DISABLE)" in runner
    fast_path = runner[
        runner.index("        if not debug_stats:"):
        runner.index("        # Optional slow-path debug timing.")
    ]
    assert "npu.synchronize" not in fast_path
    assert "if os.environ.get(ENV_CONFIG) and not _patch_gdn_dcut()" in install
