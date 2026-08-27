# SPDX-License-Identifier: Apache-2.0

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

TRUNCATE_PATH = Path(__file__).resolve().parents[1] / "truncate.py"
REPO_ROOT = TRUNCATE_PATH.parents[1]


@dataclass
class _SchedulerOutput:
    scheduled_spec_decode_tokens: dict[str, list[int]]
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_new_reqs: list[object] = field(default_factory=list)


def _load_truncate_module(target_draft_lens: list[int]):
    tree = ast.parse(TRUNCATE_PATH.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_dcut_add_zero_draft_handoffs",
            "_dcut_apply_zero_prob_recompute_caps",
            "_dcut_can_reuse_decision_for_surviving_requests",
            "_dcut_can_reuse_decision_for_zero_draft_handoffs",
            "_dcut_has_prefill",
            "_dcut_is_recompute_handoff",
            "_dcut_recompute_placeholder_req_ids",
            "_dcut_truncate",
            "_dcut_zero_draft_kv_handoff_req_ids",
        }
    ]
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)

    trim_records = []
    namespace = {
        "replace": replace,
        "_dcut_get_target_draft_lens": (
            lambda controller, original: target_draft_lens
        ),
        "_dcut_record_trim": lambda *args: trim_records.append(args[1:]),
        "PLACEHOLDER_TOKEN_ID": -1,
    }
    exec(compile(module, str(TRUNCATE_PATH), "exec"), namespace)
    return namespace, trim_records


def _load_truncate(target_draft_lens: list[int]):
    namespace, trim_records = _load_truncate_module(target_draft_lens)
    return namespace["_dcut_truncate"], trim_records


def test_recompute_handoff_placeholder_is_detected() -> None:
    namespace, _ = _load_truncate_module([])
    is_handoff = namespace["_dcut_is_recompute_handoff"]
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "handoff": [-1] * 15,
            "existing-decode": [10, 11, 12],
        },
        num_scheduled_tokens={"handoff": 16, "existing-decode": 4},
        total_num_scheduled_tokens=20,
        scheduled_new_reqs=[SimpleNamespace(req_id="handoff")],
    )

    assert is_handoff(output) is True
    assert namespace["_dcut_recompute_placeholder_req_ids"](output) == (
        frozenset({"handoff"})
    )


def test_mixed_real_and_placeholder_row_uses_native_fallback() -> None:
    namespace, _ = _load_truncate_module([])
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={"handoff": [-1, 42, -1]},
        num_scheduled_tokens={"handoff": 4},
        total_num_scheduled_tokens=4,
    )

    assert namespace["_dcut_is_recompute_handoff"](output) is True
    assert (
        namespace["_dcut_recompute_placeholder_req_ids"](output)
        == frozenset()
    )


def test_recompute_placeholders_force_zero_cap_and_keep_real_cap() -> None:
    namespace, _ = _load_truncate_module([])

    class Controller:
        def __init__(self):
            self.decision = None

        def get_adaptive_draft_len(self, req_id):
            return 2 if req_id == "decode" else None

        def set_adaptive_decision(self, req_ids, draft_lens, batch_size):
            self.decision = (req_ids, draft_lens, batch_size)

    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "handoff": [-1] * 15,
            "decode": [10, 11, 12],
        },
        num_scheduled_tokens={"handoff": 16, "decode": 4},
        total_num_scheduled_tokens=20,
    )
    controller = Controller()

    assert namespace["_dcut_apply_zero_prob_recompute_caps"](
        controller,
        output,
        frozenset({"handoff"}),
    )
    assert controller.decision == (
        ["handoff", "decode"],
        [0, 2],
        2,
    )


def test_recompute_placeholders_trim_to_anchor_only_full_row() -> None:
    truncate, trim_records = _load_truncate([0, 2])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "handoff": [-1] * 15,
            "decode": [10, 11, 12],
        },
        num_scheduled_tokens={"handoff": 16, "decode": 4},
        total_num_scheduled_tokens=20,
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=object()),
        original,
        has_prefill=False,
    )

    assert result.scheduled_spec_decode_tokens == {
        "handoff": [],
        "decode": [10, 11],
    }
    assert result.num_scheduled_tokens == {
        "handoff": 1,
        "decode": 3,
    }
    assert result.total_num_scheduled_tokens == 4
    assert trim_records == [(18, 16, 2)]


def test_real_draft_tokens_on_new_request_are_not_recompute_handoff() -> None:
    namespace, _ = _load_truncate_module([])
    is_handoff = namespace["_dcut_is_recompute_handoff"]
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={"new": [10, 11, 12]},
        num_scheduled_tokens={"new": 4},
        total_num_scheduled_tokens=4,
        scheduled_new_reqs=[SimpleNamespace(req_id="new")],
    )

    assert is_handoff(output) is False
    assert namespace["_dcut_has_prefill"](
        SimpleNamespace(is_kv_consumer=True),
        output,
    ) is True


def test_kv_consumer_first_token_handoff_becomes_zero_draft_decode() -> None:
    namespace, _ = _load_truncate_module([])
    detect_handoffs = namespace["_dcut_zero_draft_kv_handoff_req_ids"]
    add_handoffs = namespace["_dcut_add_zero_draft_handoffs"]
    has_prefill = namespace["_dcut_has_prefill"]
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={"existing": [10, 11]},
        num_scheduled_tokens={"existing": 3, "handoff": 1},
        total_num_scheduled_tokens=4,
        scheduled_new_reqs=[
            SimpleNamespace(req_id="handoff", num_computed_tokens=128)
        ],
    )
    runner = SimpleNamespace(is_kv_consumer=True)

    handoff_req_ids = detect_handoffs(runner, original)
    result = add_handoffs(original, handoff_req_ids)

    assert handoff_req_ids == frozenset({"handoff"})
    assert has_prefill(runner, original, handoff_req_ids) is False
    assert result is not original
    assert result.scheduled_spec_decode_tokens == {
        "existing": [10, 11],
        "handoff": [],
    }
    assert result.num_scheduled_tokens == original.num_scheduled_tokens
    assert result.total_num_scheduled_tokens == 4


def test_real_new_prefill_is_not_promoted_to_zero_draft_decode() -> None:
    namespace, _ = _load_truncate_module([])
    detect_handoffs = namespace["_dcut_zero_draft_kv_handoff_req_ids"]
    has_prefill = namespace["_dcut_has_prefill"]
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={"existing": [10, 11]},
        num_scheduled_tokens={"existing": 3, "prefill": 8},
        total_num_scheduled_tokens=11,
        scheduled_new_reqs=[
            SimpleNamespace(req_id="prefill", num_computed_tokens=0)
        ],
    )
    runner = SimpleNamespace(is_kv_consumer=True)

    handoff_req_ids = detect_handoffs(runner, output)

    assert handoff_req_ids == frozenset()
    assert has_prefill(runner, output, handoff_req_ids) is True


def test_placeholder_on_resumed_request_is_recompute_handoff() -> None:
    namespace, _ = _load_truncate_module([])
    is_handoff = namespace["_dcut_is_recompute_handoff"]
    output = _SchedulerOutput(
        scheduled_spec_decode_tokens={"existing": [-1] * 15},
        num_scheduled_tokens={"existing": 16},
        total_num_scheduled_tokens=16,
    )

    assert is_handoff(output) is True


def test_mixed_prefill_decode_batch_is_not_truncated() -> None:
    truncate, trim_records = _load_truncate([1])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={"decode": [10, 11, 12]},
        num_scheduled_tokens={"decode": 4, "prefill": 8},
        total_num_scheduled_tokens=12,
        scheduled_new_reqs=[SimpleNamespace(req_id="prefill")],
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=object()),
        original,
    )

    assert result is original
    assert result.scheduled_spec_decode_tokens["decode"] == [10, 11, 12]
    assert trim_records == []


def test_single_token_prefill_tail_is_not_truncated() -> None:
    truncate, trim_records = _load_truncate([1])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={"decode": [10, 11, 12]},
        num_scheduled_tokens={"decode": 4, "prefill": 1},
        total_num_scheduled_tokens=5,
    )
    input_batch = SimpleNamespace(
        req_id_to_index={"decode": 0, "prefill": 1},
        num_computed_tokens_cpu=[32, 7],
        num_prompt_tokens=[32, 8],
    )

    result = truncate(
        SimpleNamespace(
            _verify_adaptive_controller=object(),
            input_batch=input_batch,
        ),
        original,
    )

    assert result is original
    assert result.scheduled_spec_decode_tokens["decode"] == [10, 11, 12]
    assert trim_records == []


def test_request_set_change_skips_stale_partial_truncation() -> None:
    truncate, trim_records = _load_truncate([1, 1])

    class Controller:
        def matches_adaptive_request_set(self, req_ids):
            return frozenset(req_ids) == frozenset({"old-0", "old-1"})

    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "old-0": [10, 11, 12],
            "new-from-prefill": [20, 21, 22],
        },
        num_scheduled_tokens={"old-0": 4, "new-from-prefill": 4},
        total_num_scheduled_tokens=8,
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=Controller()),
        original,
    )

    assert result is original
    assert result.scheduled_spec_decode_tokens == {
        "old-0": [10, 11, 12],
        "new-from-prefill": [20, 21, 22],
    }
    assert trim_records == []


def test_finished_requests_reuse_caps_for_surviving_subset() -> None:
    truncate, trim_records = _load_truncate([1, 2])

    class Controller:
        _adaptive_decision_req_ids = frozenset(
            {"survivor-0", "survivor-1", "finished"}
        )
        _adaptive_decision_batch_size = 3
        _sorted_bs = (4,)

        def matches_adaptive_request_set(self, req_ids):
            return False

        def get_adaptive_draft_len(self, req_id):
            return {"survivor-0": 1, "survivor-1": 2}.get(req_id)

    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "survivor-0": [10, 11, 12],
            "survivor-1": [20, 21, 22],
        },
        num_scheduled_tokens={"survivor-0": 4, "survivor-1": 4},
        total_num_scheduled_tokens=8,
    )
    runner = SimpleNamespace(
        _verify_adaptive_controller=Controller(),
    )

    result = truncate(runner, original, has_prefill=False)

    assert result.scheduled_spec_decode_tokens == {
        "survivor-0": [10],
        "survivor-1": [20, 21],
    }
    assert result.total_num_scheduled_tokens == 5
    assert runner._dcut_last_reused_survivor_decision is True
    assert trim_records == [(6, 3, 2)]


def test_missing_survivor_cap_does_not_reuse_partial_decision() -> None:
    truncate, trim_records = _load_truncate([1, 3])

    class Controller:
        _adaptive_decision_req_ids = frozenset(
            {"survivor-0", "survivor-1", "finished"}
        )
        _adaptive_decision_batch_size = 3
        _sorted_bs = (4,)

        def matches_adaptive_request_set(self, req_ids):
            return False

        def get_adaptive_draft_len(self, req_id):
            return 1 if req_id == "survivor-0" else None

    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "survivor-0": [10, 11, 12],
            "survivor-1": [20, 21, 22],
        },
        num_scheduled_tokens={"survivor-0": 4, "survivor-1": 4},
        total_num_scheduled_tokens=8,
    )
    runner = SimpleNamespace(
        _verify_adaptive_controller=Controller(),
    )

    result = truncate(runner, original, has_prefill=False)

    assert result is original
    assert runner._dcut_last_reused_survivor_decision is False
    assert trim_records == []


def test_survivor_caps_are_not_reused_across_batch_bucket() -> None:
    truncate, trim_records = _load_truncate([1, 1])

    class Controller:
        _adaptive_decision_req_ids = frozenset(
            {"survivor-0", "survivor-1", "finished"}
        )
        _adaptive_decision_batch_size = 3
        _sorted_bs = (2, 4)

        def matches_adaptive_request_set(self, req_ids):
            return False

        def get_adaptive_draft_len(self, req_id):
            return 1

    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "survivor-0": [10, 11, 12],
            "survivor-1": [20, 21, 22],
        },
        num_scheduled_tokens={"survivor-0": 4, "survivor-1": 4},
        total_num_scheduled_tokens=8,
    )
    runner = SimpleNamespace(
        _verify_adaptive_controller=Controller(),
    )

    result = truncate(runner, original, has_prefill=False)

    assert result is original
    assert runner._dcut_last_reused_survivor_decision is False
    assert trim_records == []


def test_zero_draft_handoff_reuses_surviving_cached_decisions() -> None:
    truncate, trim_records = _load_truncate([1])

    class Controller:
        _adaptive_decision_req_ids = frozenset({"old-0", "old-1"})
        _adaptive_decision_batch_size = 2

        def matches_adaptive_request_set(self, req_ids):
            return False

    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "old-0": [10, 11, 12],
        },
        num_scheduled_tokens={"old-0": 4, "handoff": 1},
        total_num_scheduled_tokens=5,
        scheduled_new_reqs=[
            SimpleNamespace(req_id="handoff", num_computed_tokens=128)
        ],
    )
    runner = SimpleNamespace(
        _verify_adaptive_controller=Controller(),
    )

    truncated = truncate(
        runner,
        original,
        has_prefill=False,
        zero_draft_handoff_req_ids=frozenset({"handoff"}),
    )

    assert truncated.scheduled_spec_decode_tokens == {
        "old-0": [10],
    }
    assert truncated.num_scheduled_tokens == {
        "old-0": 2,
        "handoff": 1,
    }
    assert truncated.total_num_scheduled_tokens == 3
    assert runner._dcut_last_reused_handoff_decision is True
    assert trim_records == [(3, 2, 1)]


def test_zero_draft_handoff_does_not_cover_unknown_spec_request() -> None:
    truncate, trim_records = _load_truncate([1, 0])

    class Controller:
        _adaptive_decision_req_ids = frozenset({"old-0", "old-1"})
        _adaptive_decision_batch_size = 3

        def matches_adaptive_request_set(self, req_ids):
            return False

    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "old-0": [10, 11, 12],
            "unknown": [20, 21, 22],
        },
        num_scheduled_tokens={
            "old-0": 4,
            "unknown": 4,
            "handoff": 1,
        },
        total_num_scheduled_tokens=9,
    )
    runner = SimpleNamespace(
        _verify_adaptive_controller=Controller(),
    )

    result = truncate(
        runner,
        original,
        has_prefill=False,
        zero_draft_handoff_req_ids=frozenset({"handoff"}),
    )

    assert result is original
    assert runner._dcut_last_reused_handoff_decision is False
    assert trim_records == []


def test_matching_request_set_applies_one_coherent_decision() -> None:
    truncate, trim_records = _load_truncate([1, 2])

    class Controller:
        def matches_adaptive_request_set(self, req_ids):
            return frozenset(req_ids) == frozenset({"request-0", "request-1"})

    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "request-0": [10, 11, 12],
            "request-1": [20, 21, 22],
        },
        num_scheduled_tokens={"request-0": 4, "request-1": 4},
        total_num_scheduled_tokens=8,
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=Controller()),
        original,
    )

    assert result.scheduled_spec_decode_tokens == {
        "request-0": [10],
        "request-1": [20, 21],
    }
    assert result.num_scheduled_tokens == {"request-0": 2, "request-1": 3}
    assert result.total_num_scheduled_tokens == 5
    assert trim_records == [(6, 3, 2)]


def test_zero_draft_decision_stays_on_spec_path_without_adding_tokens() -> None:
    truncate, trim_records = _load_truncate([0, 2])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={
            "zero": [10, 11, 12],
            "kept": [20, 21, 22],
        },
        num_scheduled_tokens={"zero": 4, "kept": 4, "ordinary": 1},
        total_num_scheduled_tokens=9,
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=object()),
        original,
    )

    assert result is not original
    assert result.scheduled_spec_decode_tokens == {
        "zero": [],
        "kept": [20, 21],
    }
    assert result.num_scheduled_tokens == {
        "zero": 1,
        "kept": 3,
        "ordinary": 1,
    }
    assert result.total_num_scheduled_tokens == 5
    assert "ordinary" not in result.scheduled_spec_decode_tokens
    assert original.scheduled_spec_decode_tokens["zero"] == [10, 11, 12]
    assert trim_records == [(6, 4, 2)]


def test_all_zero_draft_decisions_keep_a_truthy_spec_mapping() -> None:
    truncate, _ = _load_truncate([0, 0])
    original = _SchedulerOutput(
        scheduled_spec_decode_tokens={"first": [1, 2], "second": [3]},
        num_scheduled_tokens={"first": 3, "second": 2},
        total_num_scheduled_tokens=5,
    )

    result = truncate(
        SimpleNamespace(_verify_adaptive_controller=object()),
        original,
    )

    assert result.scheduled_spec_decode_tokens == {
        "first": [],
        "second": [],
    }
    assert result.scheduled_spec_decode_tokens
    assert result.num_scheduled_tokens == {"first": 1, "second": 1}
    assert result.total_num_scheduled_tokens == 2


def test_gdn_uses_zero_as_spec_and_minus_one_as_non_spec() -> None:
    runner = (
        REPO_ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
    ).read_text(encoding="utf-8")
    builder = (
        REPO_ROOT / "vllm_ascend" / "ops" / "gdn_attn_builder.py"
    ).read_text(encoding="utf-8")

    assert "num_decode_draft_tokens = np.full(num_reqs, -1" in runner
    assert "num_decode_draft_tokens[req_idx] = draft_len" in runner
    assert (
        "torch.ge(\n"
        "                num_decode_draft_tokens_cpu,\n"
        "                0,"
    ) in builder
