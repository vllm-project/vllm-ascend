# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


DCUT_DIR = Path(__file__).resolve().parents[1]


def _load_bypass():
    path = DCUT_DIR / "probs.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dcut_bypass_prob_capture_for_prefill"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_dcut_bypass_prob_capture_for_prefill"]


def _load_drafter_enable():
    path = DCUT_DIR / "controller.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dcut_enable_drafter_probs"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_dcut_patch_drafter_instance": lambda drafter: None,
        "_dcut_setup_full_decode_drafter": lambda runner, drafter: None,
        "_env_flag": lambda name: False,
        "ENV_FORCE_DRAFTER_EAGER": "VLLM_DCUT_FORCE_DRAFTER_EAGER",
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_dcut_enable_drafter_probs"]


def _load_runner_helper(name: str, **namespace):
    path = DCUT_DIR / "patch_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def test_prefill_bypass_disables_drafter_and_drops_stale_decision() -> None:
    class Controller:
        cleared = 0

        def clear_adaptive_decision(self):
            self.cleared += 1

    controller = Controller()
    drafter = SimpleNamespace(
        needs_draft_probs=True,
        _dcut_last_draft_ran_python=True,
        _dcut_last_logits_for_probs=object(),
        _last_selected_probs=object(),
    )
    runner = SimpleNamespace(
        drafter=drafter,
        _adaptive_probs_pending=True,
        _adaptive_num_reqs=2,
        _adaptive_req_ids=["old-0", "old-1"],
        _adaptive_active={"old-0", "old-1"},
        _verify_adaptive_controller=controller,
    )

    _load_bypass()(runner)

    assert drafter.needs_draft_probs is False
    assert drafter._dcut_last_draft_ran_python is False
    assert drafter._dcut_last_logits_for_probs is None
    assert drafter._last_selected_probs is None
    assert runner._adaptive_probs_pending is False
    assert runner._adaptive_probs_expired is False
    assert runner._adaptive_probs_source == "prefill_bypass"
    assert runner._adaptive_num_reqs == 0
    assert runner._adaptive_req_ids == []
    assert runner._adaptive_active == set()
    assert controller.cleared == 1


def test_decode_after_prefill_reenables_drafter_probabilities() -> None:
    controller = SimpleNamespace(clear_adaptive_decision=lambda: None)
    drafter = SimpleNamespace(
        needs_draft_probs=True,
        _dcut_last_draft_ran_python=True,
        _dcut_last_logits_for_probs=object(),
        _last_selected_probs=object(),
        method="dflash",
        parallel_drafting=False,
    )
    runner = SimpleNamespace(
        drafter=drafter,
        _adaptive_probs_pending=False,
        _adaptive_num_reqs=0,
        _adaptive_req_ids=[],
        _adaptive_active=set(),
        _verify_adaptive_controller=controller,
        _dcut_logged_drafter_probs=True,
    )

    _load_bypass()(runner)
    _load_drafter_enable()(runner)

    assert drafter.needs_draft_probs is True


def test_prefill_execute_branch_skips_all_probability_work() -> None:
    source = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    execute_start = source.index("    def execute_model(")
    execute_end = source.index("    _orig_sample_tokens", execute_start)
    execute_source = source[execute_start:execute_end]

    prefill_start = execute_source.index(
        "        if (\n"
        "            _ctrl is not None\n"
        "            and _has_prefill\n"
        "            and not _native_recompute_handoff\n"
        "        ):"
    )
    decode_start = execute_source.index(
        "        if (\n"
        "            _ctrl is not None\n"
        "            and not _has_prefill\n"
        "            and not _native_recompute_handoff\n"
        "        ):",
        prefill_start,
    )
    prefill_branch = execute_source[prefill_start:decode_start]

    assert "_dcut_bypass_prob_capture_for_prefill(self)" in prefill_branch
    assert "_maybe_process_adaptive_probs" not in prefill_branch
    assert "_dcut_enable_drafter_probs" not in prefill_branch
    assert "_dcut_truncate" not in prefill_branch
    assert "_dcut_prepare_prob_capture" not in prefill_branch

    decode_end = execute_source.index(
        "        if not debug_stats:", decode_start
    )
    decode_branch = execute_source[decode_start:decode_end]
    assert "_dcut_enable_drafter_probs(self)" in decode_branch
    assert "_dcut_prepare_prob_capture(self, scheduler_output)" in decode_branch

    assert '"drafter_needs_draft_probs"' in execute_source
    assert '"adaptive_probs_pending_after_step"' in execute_source
    assert '"prob_capture_skipped_for_prefill"' in execute_source
    assert '"mixed_prob_capture_planned"' in execute_source
    assert '"draft_ran_python"' in execute_source


def test_mixed_draft_capture_is_enabled_before_proposal() -> None:
    source = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    sample_start = source.index("    def sample_tokens(")
    copy_start = source.index(
        "    def _copy_draft_token_ids_to_cpu(",
        sample_start,
    )
    update_start = source.index("    def _update_states(", copy_start)

    sample_source = source[sample_start:copy_start]
    copy_source = source[copy_start:update_start]
    guard = 'getattr(self, "_dcut_skip_current_prob_capture", False)'
    mixed_flag = 'getattr(self, "_dcut_capture_mixed_probs", False)'

    assert mixed_flag in sample_source
    assert sample_source.index(mixed_flag) < sample_source.index(
        "out = _orig_sample_tokens"
    )
    assert sample_source.index("_dcut_enable_drafter_probs(self)") < (
        sample_source.index("out = _orig_sample_tokens")
    )
    assert guard in sample_source
    assert sample_source.index(guard) < sample_source.index(
        "_maybe_process_adaptive_probs"
    )
    assert guard in copy_source
    assert copy_source.index(guard) < copy_source.index("_dcut_queue_probs")


def test_only_pure_prefill_skips_next_proposal_probabilities() -> None:
    source = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    execute_start = source.index("    def execute_model(")
    execute_end = source.index("    _orig_sample_tokens", execute_start)
    execute_source = source[execute_start:execute_end]

    assert (
        "_ctrl is not None and _has_prefill and _has_spec"
        in execute_source
    )
    assert (
        "_ctrl is not None and _has_prefill and not _has_spec"
        in execute_source
    )


def test_recompute_handoff_uses_dcut_for_piecewise_and_supported_full() -> None:
    graph_enabled = _load_runner_helper(
        "_dcut_adaptive_handoff_graph_enabled",
        _dcut_full_decode_multishape_enabled=lambda config: config.full,
    )
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="PIECEWISE")
        ),
        vllm_config=SimpleNamespace(full=False),
    )
    assert graph_enabled(runner) is True

    runner.compilation_config.cudagraph_mode = SimpleNamespace(
        name="FULL_DECODE_ONLY"
    )
    assert graph_enabled(runner) is False
    runner.vllm_config.full = True
    assert graph_enabled(runner) is True

    runner.compilation_config.cudagraph_mode = SimpleNamespace(name="NONE")
    assert graph_enabled(runner) is False

    source = (DCUT_DIR / "patch_runner.py").read_text(encoding="utf-8")
    execute_start = source.index("    def execute_model(")
    execute_end = source.index("    _orig_sample_tokens", execute_start)
    execute_source = source[execute_start:execute_end]

    assert "_adaptive_recompute_handoff = bool(" in execute_source
    assert "and _recompute_placeholder_req_ids" in execute_source
    assert "and _adaptive_handoff_graph" in execute_source
    assert "_dcut_adaptive_handoff_graph_enabled(self)" in execute_source
    assert "_native_recompute_handoff = bool(" in execute_source

    native_start = execute_source.index(
        "        if _native_recompute_handoff:"
    )
    regular_start = execute_source.index(
        "        _zero_draft_handoffs = frozenset()",
        native_start,
    )
    handoff_branch = execute_source[native_start:regular_start]

    assert "self._dcut_capture_mixed_probs = False" in handoff_branch
    assert "self._dcut_skip_current_prob_capture = True" in handoff_branch
    assert "_dcut_bypass_prob_capture_for_prefill(self)" in handoff_branch
    assert "return _dcut_execute_native_recompute_handoff" in handoff_branch
    assert "elif _adaptive_recompute_handoff:" in handoff_branch
    assert "self._dcut_skip_current_prob_capture = False" in handoff_branch

    assert "_dcut_apply_zero_prob_recompute_caps(" in execute_source
    assert "_dcut_truncate(" in execute_source
    assert "and not _native_recompute_handoff" in execute_source
    assert (
        "self._dcut_zero_draft_handoffs_for_proposal = (\n"
        "            _prefill_exempt_req_ids\n"
        "        )"
        in execute_source
    )
    assert "| _recompute_placeholder_req_ids" in execute_source

    determine_start = source.index(
        "    def _determine_batch_execution_and_padding("
    )
    determine_end = source.index(
        "    def _pad_query_start_loc_for_fia(", determine_start
    )
    determine_source = source[determine_start:determine_end]
    assert '"_dcut_recompute_handoff_active"' in determine_source
    assert "stock_uniform_decode" in determine_source
    assert "is_all_decode" in determine_source
    assert "num_tokens == expected_query_len * num_reqs" in determine_source
    assert "force_eager = True" in determine_source
    assert "_dcut_set_full_gdn_metadata_route(" in determine_source
    assert "runtime_mode == CUDAGraphMode.FULL and descriptor.uniform" in (
        determine_source
    )

    model_forward_start = source.index("    def _model_forward(")
    model_forward_end = source.index(
        "    def execute_model(", model_forward_start
    )
    model_forward_source = source[model_forward_start:model_forward_end]
    assert "stock_uniform_full = bool(" in model_forward_source
    assert "native_gdn_batch = stock_uniform_full or (" in model_forward_source
    assert "initial_spec_rows = _dcut_initial_handoff_spec_rows(self)" in (
        model_forward_source
    )
    piecewise_replay = model_forward_source[
        model_forward_source.index("self._dcut_gdn_piecewise_enabled") :
    ]
    assert "initial_spec_rows=initial_spec_rows" in piecewise_replay


def test_initial_handoff_rows_follow_compact_spec_order() -> None:
    initial_rows = _load_runner_helper("_dcut_initial_handoff_spec_rows")
    runner = SimpleNamespace(
        _dcut_zero_draft_handoffs_for_proposal=frozenset(
            {"handoff-0", "handoff-1"}
        ),
        num_decode_draft_tokens=SimpleNamespace(np=[3, -1, 0, 0]),
        input_batch=SimpleNamespace(
            num_reqs=4,
            req_ids=["decode", "ordinary", "handoff-0", "handoff-1"],
        ),
    )

    assert initial_rows(runner) == (1, 2)
