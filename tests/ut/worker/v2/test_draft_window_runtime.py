# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests at the proposal and metadata callback seams."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import DraftWindowOwner, DraftWindowOwnerState
from vllm_ascend.worker.v2.spec_decode.draft_window_policy import resolve_draft_window_policy
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import DraftWindowRuntime, continuing_prefill_rows

runtime_module = sys.modules[DraftWindowRuntime.__module__]


class Completion:
    complete = False

    def is_complete(self):
        return self.complete

    def wait(self):
        self.complete = True


@pytest.fixture
def setup(monkeypatch):
    # Isolate backend discovery; its actual class/layout checks have their own
    # inventory tests. The read provider and owner pool below are real code.
    monkeypatch.setattr(DraftWindowRuntime, "_validate_route", staticmethod(lambda spec: (0,)))
    completions = []

    def record():
        completion = Completion()
        completions.append(completion)
        return completion

    monkeypatch.setattr(runtime_module, "_NPUCompletion", record)
    canonical = torch.arange(48, dtype=torch.int32).reshape(6, 8)
    spec = SimpleNamespace(
        draft_model_config=SimpleNamespace(enforce_eager=True),
        speculative_config=SimpleNamespace(enforce_eager=None),
        vllm_config=SimpleNamespace(max_concurrent_batches=2),
        num_query_per_req=3,
        num_speculative_steps=3,
        use_fused_multi_step_decode=False,
        input_buffers=SimpleNamespace(positions=torch.tensor([11, 26])),
        last_token_indices=torch.tensor([0, 1]),
        block_tables=SimpleNamespace(
            block_tables=[SimpleNamespace(gpu=canonical)],
            num_blocks=SimpleNamespace(np=np.full((1, 6), 8, dtype=np.int32)),
            kernel_block_sizes=[4],
        ),
    )
    policy = resolve_draft_window_policy(requested_window_size=8, method="dflash")
    runtime = DraftWindowRuntime(spec, policy)
    return runtime, spec, canonical, completions


@pytest.mark.parametrize("method", ["eagle3", "dflash", "dspark"])
@pytest.mark.parametrize("inherited_eager", [False, True])
@pytest.mark.parametrize("draft_override", [None, False, True])
def test_effective_draft_eager_preserves_target_graph(setup, method, inherited_eager, draft_override):
    _, spec, _, _ = setup
    spec.draft_model_config.enforce_eager = inherited_eager
    spec.speculative_config.enforce_eager = draft_override
    policy = resolve_draft_window_policy(requested_window_size=1024, method=method)
    # Match actual graph-manager initialization: an eager Target supplies NONE,
    # and the explicit Draft eager flag additionally forces NONE. False does
    # not enable graphs when the Target supplied NONE.
    if inherited_eager or draft_override:
        assert DraftWindowRuntime(spec, policy).method == method
    else:
        with pytest.raises(ValueError, match="eager Draft"):
            DraftWindowRuntime(spec, policy)


@pytest.mark.parametrize(
    "native_windows,window,expected",
    [
        ({}, 1024, ()),
        ({0: 2048}, 1024, (2048,)),
        ({0: 2048}, 2048, ()),
        ({0: 2048}, 4096, ()),
        ({0: 2048, 1: 4096}, 2048, (4096,)),
        ({0: 4096, 1: 4096}, 1024, (4096,)),
    ],
)
def test_native_swa_warning_uses_only_certified_narrowed_draft_groups(
    setup, monkeypatch, native_windows, window, expected
):
    _, spec, canonical, _ = setup
    before = canonical.clone()
    warnings = []

    def validate(runtime, actual_spec):
        assert actual_spec is spec
        runtime.native_swa_intersections = native_windows
        return (0,)

    monkeypatch.setattr(DraftWindowRuntime, "_validate_route", validate)
    monkeypatch.setattr(runtime_module.logger, "warning_once", lambda *args, **kwargs: warnings.append((args, kwargs)))
    runtime = DraftWindowRuntime(spec, resolve_draft_window_policy(requested_window_size=window, method="dflash"))
    assert runtime.window_size == window
    assert runtime.native_swa_intersections == native_windows
    assert torch.equal(canonical, before)
    assert len(warnings) == bool(expected)
    if expected:
        args, kwargs = warnings[0]
        assert args[1:] == (expected, window)
        assert kwargs == {"scope": "global"}
        assert "may reduce" in args[0]


@pytest.mark.parametrize("current_capability,legacy_flag", [(True, False), (False, True)])
def test_eagle_window_checks_current_decode_metadata_update_capability(setup, current_capability, legacy_flag):
    _, spec, _, _ = setup
    spec.supports_decode_metadata_update = current_capability
    spec.use_fused_multi_step_decode = legacy_flag
    policy = resolve_draft_window_policy(requested_window_size=1024, method="eagle3")
    if current_capability:
        with pytest.raises(NotImplementedError, match="per-step metadata"):
            DraftWindowRuntime(spec, policy)
    else:
        assert DraftWindowRuntime(spec, policy).method == "eagle3"


def batch(*, rows=(1, 3), computed=(8, 24), prompt=(16, 16), scheduled=(7, 3), drafts=(3, 2)):
    return SimpleNamespace(
        num_reqs=len(rows),
        req_ids=[f"request-{row}" for row in rows],
        idx_mapping_np=np.array(rows),
        num_computed_prefill_tokens_np=np.array(computed),
        prefill_len_np=np.array(prompt),
        num_scheduled_tokens=np.array(scheduled),
        num_draft_tokens_per_req=None if drafts is None else np.array(drafts),
        is_prefilling_np=np.array(computed) < np.array(prompt),
    )


def common(*, exact=(12, 27), upper=(14, 30)):
    return SimpleNamespace(
        num_reqs=3,
        seq_lens=torch.tensor([*exact, 0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([*upper, 0], dtype=torch.int32),
        block_table_tensor=torch.full((3, 8), -99, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 3, 6, 6]),
        positions=torch.arange(6),
        slot_mapping=torch.arange(6) + 20,
    )


def test_mixed_phase_excludes_draft_placeholders_and_recognizes_terminal():
    state = batch()
    assert continuing_prefill_rows(state, 2).tolist() == [True, False]
    state.num_computed_prefill_tokens_np[0] = 12
    assert continuing_prefill_rows(state, 2).tolist() == [False, False]


def test_no_draft_placeholders_and_invalid_scheduled_counts():
    assert continuing_prefill_rows(batch(drafts=None), 2).tolist() == [True, False]
    with pytest.raises(ValueError, match="exceed scheduled"):
        continuing_prefill_rows(batch(drafts=(8, 2)), 2)


def test_callback_reads_canonical_source_and_preserves_write_query_objects(setup):
    runtime, spec, canonical, completions = setup
    metadata = common()
    original_read = metadata.seq_lens
    write, positions, query = metadata.slot_mapping, metadata.positions, metadata.query_start_loc
    full = canonical.clone()
    with runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(3, 2)(0, metadata)
        runtime.retain_metadata({"layer": metadata})
    # Continuing prefill is Full; decode E=27 W=8 B=4 gives S=16 V=11.
    assert metadata.seq_lens.tolist() == [12, 11, 0]
    assert metadata.seq_lens_cpu_upper_bound.tolist() == [14, 14, 0]
    assert metadata.block_table_tensor[0, :3].tolist() == canonical[1, :3].tolist()
    assert metadata.block_table_tensor[1, :3].tolist() == canonical[3, 4:7].tolist()
    assert metadata.block_table_tensor.data_ptr() != canonical.data_ptr()
    assert metadata.slot_mapping is write
    assert metadata.positions is positions
    assert metadata.query_start_loc is query
    assert original_read.tolist() == [12, 27, 0]
    assert torch.equal(canonical, full)
    assert len(completions) == 1


def test_exact_after_rejection_controls_start_not_cpu_upper(setup):
    runtime, spec, canonical, _ = setup
    metadata = common(exact=(12, 15), upper=(14, 30))
    with runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(3, 2)(0, metadata)
    assert metadata.seq_lens.tolist() == [12, 11, 0]
    assert metadata.block_table_tensor[1, :3].tolist() == canonical[3, 1:4].tolist()


def test_unprojected_native_group_metadata_is_untouched_while_full_group_windows(setup):
    runtime, spec, canonical, completions = setup
    other_table = canonical.clone() + 100
    spec.block_tables.block_tables.append(SimpleNamespace(gpu=other_table))
    spec.block_tables.num_blocks.np = np.full((2, 6), 8, dtype=np.int32)
    spec.block_tables.kernel_block_sizes = [4, 4]
    # Backend selection's static native-SWA subset certificate has already
    # selected gid1 alone; the actual callback must leave gid0 completely
    # outside the private read projection, rather than sharing/rebinding it.
    runtime.gids = (1,)
    native_metadata, full_metadata = common(), common()
    original_native = vars(native_metadata).copy()
    native_values = {name: value.clone() for name, value in original_native.items() if isinstance(value, torch.Tensor)}
    original_slots = full_metadata.slot_mapping
    with runtime.proposal_context(spec, batch()):
        callback = runtime.metadata_callback(3, 2)
        callback(0, native_metadata)
        callback(1, full_metadata)
    assert vars(native_metadata).keys() == original_native.keys()
    for name, value in original_native.items():
        assert getattr(native_metadata, name) is value
        if name in native_values:
            assert torch.equal(value, native_values[name])
    assert full_metadata.seq_lens.tolist() == [12, 11, 0]
    assert full_metadata.block_table_tensor[1, :3].tolist() == other_table[3, 4:7].tolist()
    assert full_metadata.slot_mapping is original_slots
    assert torch.equal(other_table, canonical + 100)
    assert len(completions) == 1


def test_eagle_uniform_first_pass_reads_current_device_end_without_host_snapshot(setup, monkeypatch):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime.window_size = "eagle3", 3, 1035
    runtime.anchor_block_span = 128
    runtime._dense_device_reads = True
    spec.block_tables.kernel_block_sizes = [128]
    spec.block_tables.block_tables[0].gpu = torch.arange(120, dtype=torch.int32).reshape(6, 20)
    spec.block_tables.num_blocks.np[:] = 20
    spec.input_buffers.positions = torch.tensor([1155, 1156, 1157, 1158, 1156, 1157, 1158, 1159])
    spec.last_token_indices = torch.tensor([3, 6])
    state = batch(computed=(1150, 1150), prompt=(1, 1))
    state.idx_mapping = torch.tensor(state.idx_mapping_np, dtype=torch.int64)
    monkeypatch.setattr(
        runtime, "_eagle_first_pass_lengths", lambda *args: pytest.fail("uniform decode copied Host exact")
    )
    with runtime.proposal_context(spec, state):
        for step, ends, width in [(0, (1159, 1160), 4), (1, (1160, 1160), 1), (2, (1161, 1161), 1)]:
            metadata = common(exact=ends, upper=(1162, 1163))
            metadata.causal = True
            metadata.query_start_loc_cpu = torch.tensor([0, width, 2 * width, 2 * width])
            runtime.metadata_callback(step, 2)(0, metadata)
            read = metadata._draft_window_device_read
            # Rejected native padding must not move H: live E1159+K3 gives
            # H1162, S0 at W1035/B128, although native extent1160 would give S128.
            assert read.starts.tolist() == [0, 0]
            assert read.seq_lens.tolist() == ([1159, 1159] if step == 0 else list(ends))
            assert read.query_width == width


def test_eagle_host_fallback_reuses_the_same_device_first_pass_anchor(setup):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime.window_size = "eagle3", 3, 1035
    runtime.anchor_block_span = 128
    runtime._dense_device_reads = True
    spec.block_tables.kernel_block_sizes = [128]
    spec.block_tables.block_tables[0].gpu = torch.arange(120, dtype=torch.int32).reshape(6, 20)
    spec.block_tables.num_blocks.np[:] = 20
    spec.input_buffers.positions = torch.tensor([1155, 1156, 1157, 1158, 1156, 1157, 1158, 1159])
    spec.last_token_indices = torch.tensor([3, 6])
    state = batch(computed=(1150, 1150), prompt=(1, 1))
    state.idx_mapping = torch.tensor(state.idx_mapping_np, dtype=torch.int64)
    with runtime.proposal_context(spec, state):
        for step, ends, width in [(0, (1159, 1160), 4), (1, (1160, 1160), 1), (2, (1161, 1161), 1)]:
            # A later loose workspace bound exceeds initialized columns. The
            # legacy fallback must use retained current first-pass E, not E-W.
            upper = (1162, 1163) if step == 0 else (3000, 3000)
            metadata = common(exact=ends, upper=upper)
            metadata.causal = True
            metadata.query_start_loc_cpu = torch.tensor([0, width, 2 * width, 2 * width])
            runtime.metadata_callback(step, 2)(0, metadata)
            assert metadata.seq_lens[:2].tolist() == ([1159, 1159] if step == 0 else list(ends))
        assert runtime._eagle_plans is not None


def test_reorder_row_replacement_and_in_place_mapping_refresh_each_proposal(setup):
    runtime, spec, canonical, completions = setup
    state = batch()
    first, second = common(), common()
    with runtime.proposal_context(spec, state):
        runtime.metadata_callback(3, 2)(0, first)
    saved_first = first.block_table_tensor.clone()
    state.idx_mapping_np[:] = [4, 2]
    state.req_ids[:] = ["replacement", "survivor"]
    canonical[4] += 100
    with runtime.proposal_context(spec, state):
        runtime.metadata_callback(3, 2)(0, second)
    assert second.block_table_tensor[0, :3].tolist() == canonical[4, :3].tolist()
    assert second.block_table_tensor[1, :3].tolist() == canonical[2, 4:7].tolist()
    assert torch.equal(first.block_table_tensor, saved_first)
    assert len(completions) == 2


@pytest.mark.parametrize("phase", ["dummy_run", "is_profile"])
def test_dummy_and_profile_bypass_projection_and_completion(setup, phase):
    runtime, spec, _, completions = setup
    with runtime.proposal_context(spec, batch(), **{phase: True}):
        assert runtime.metadata_callback(3, 2) is None
        runtime.retain_metadata(object())
    assert not completions
    assert not runtime.pool.poisoned


def test_failed_native_proposal_poison_keeps_read_metadata(setup):
    runtime, spec, _, completions = setup
    metadata = common()
    with pytest.raises(RuntimeError, match="after model enqueue"), runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(3, 2)(0, metadata)
        owner = runtime._owner
        raise RuntimeError("after model enqueue")
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert metadata in owner.retained
    assert not completions
    with pytest.raises(RuntimeError, match="quarantined"), runtime.proposal_context(spec, batch()):
        pytest.fail("must not enter another native proposal")


def test_event_failure_after_successful_build_poison_keeps_read_metadata(setup, monkeypatch):
    runtime, spec, _, _ = setup

    def failed_record():
        raise RuntimeError("record failed")

    monkeypatch.setattr(runtime_module, "_NPUCompletion", failed_record)
    with pytest.raises(RuntimeError, match="record failed"), runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(3, 2)(0, common())
        owner = runtime._owner
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert owner.retained


def test_no_consumption_and_wrong_actual_query_width_are_errors(setup):
    runtime, spec, _, _ = setup
    with pytest.raises(ValueError, match="actual query width"), runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(8, 2)
    assert runtime.pool.poisoned


def test_stale_callback_cannot_modify_next_proposal(setup):
    runtime, spec, _, _ = setup
    with runtime.proposal_context(spec, batch()):
        stale = runtime.metadata_callback(3, 2)
        stale(0, common())
    with pytest.raises(RuntimeError, match="stale proposal"), runtime.proposal_context(spec, batch()):
        stale(0, common())


def test_target_only_group_is_not_projected(setup):
    runtime, spec, _, _ = setup
    target_only = common()
    table = target_only.block_table_tensor
    with runtime.proposal_context(spec, batch()):
        callback = runtime.metadata_callback(3, 2)
        callback(1, target_only)
        callback(0, common())
    assert target_only.block_table_tensor is table


def test_off_policy_cannot_construct_runtime_or_inspect_speculator():
    policy = resolve_draft_window_policy(requested_window_size=None, method="dflash")
    with pytest.raises(ValueError, match="OFF"), pytest.MonkeyPatch.context() as patch:
        patch.setattr(DraftWindowRuntime, "_validate_route", lambda *args: pytest.fail("OFF inspected backend"))
        DraftWindowRuntime(object(), policy)


def test_callback_runs_after_common_construction_and_before_actual_builder():
    root = Path(__file__).resolve().parents[4]
    source = (root / "vllm_ascend/worker/v2/attn_utils.py").read_text()
    tree = ast.parse(source)
    build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_attn_metadata")
    calls = [node for node in ast.walk(build) if isinstance(node, ast.Call)]
    common_line = next(
        node.lineno
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "AscendCommonAttentionMetadata"
    )
    callback_line = next(
        node.lineno for node in calls if isinstance(node.func, ast.Name) and node.func.id == "common_metadata_callback"
    )
    builder_lines = [
        node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "build"
    ]
    assert common_line < callback_line < min(builder_lines)


def test_eagle_freezes_anchor_across_rejection_and_retains_all_step_reads(setup):
    runtime, spec, canonical, completions = setup
    policy = resolve_draft_window_policy(requested_window_size=8, method="eagle3")
    runtime = DraftWindowRuntime(spec, policy)
    state = batch()
    outputs = []
    with runtime.proposal_context(spec, state):
        for step, exact in enumerate(((12, 27), (13, 25), (14, 26))):
            metadata = common(exact=exact, upper=(16, 31))
            runtime.metadata_callback(step, 2)(0, metadata)
            runtime.retain_metadata(metadata)
            outputs.append(metadata)
        owner = runtime._owner
        assert runtime.reads_active
    assert not runtime.reads_active
    # E0=27, H=30, W=8, B=4 => frozen S=20, including after rejection.
    assert [metadata.seq_lens[1].item() for metadata in outputs] == [7, 5, 6]
    assert all(metadata.block_table_tensor[1, 0] == canonical[3, 5] for metadata in outputs)
    assert len({metadata.block_table_tensor.data_ptr() for metadata in outputs}) == 3
    assert all(any(value is metadata for value in owner.retained) for metadata in outputs)
    assert sum(type(value).__name__ == "DraftReadView" for value in owner.retained) == 3
    assert len(completions) == 1


@pytest.mark.parametrize("requested", [0, 1, 2, 3, None])
def test_eagle_scheduled_prefill_and_steps_keep_static_window_geometry(setup, requested):
    _, spec, canonical, completions = setup
    runtime = DraftWindowRuntime(spec, resolve_draft_window_policy(requested_window_size=8, method="eagle3"))
    consumed = 3 if requested is None else max(1, requested)
    outputs = []
    with runtime.proposal_context(spec, batch(), num_speculative_tokens=requested):
        for step, exact in enumerate(((12, 27), (13, 25), (14, 26))[:consumed]):
            metadata = common(exact=exact, upper=(16, 31))
            runtime.metadata_callback(step, 2)(0, metadata)
            outputs.append(metadata)
        owner = runtime._owner
    assert owner is not None and owner.state is DraftWindowOwnerState.SUBMITTED
    assert runtime.static_k == 3
    assert len(completions) == 1 and runtime._next_step == consumed
    # The configured three-step horizon still freezes S=20. A scheduled
    # zero/short proposal must not silently change the window formula.
    assert all(metadata.block_table_tensor[1, 0] == canonical[3, 5] for metadata in outputs)


@pytest.mark.parametrize("requested", [-1, 4, True])
def test_eagle_invalid_scheduled_count_cannot_enter_a_window_scope(setup, requested):
    _, spec, _, completions = setup
    runtime = DraftWindowRuntime(spec, resolve_draft_window_policy(requested_window_size=8, method="eagle3"))
    with (
        pytest.raises(ValueError, match="configured static K"),
        runtime.proposal_context(spec, batch(), num_speculative_tokens=requested),
    ):
        pytest.fail("invalid count must fail before proposal ownership")
    assert runtime._owner is None and not completions


def test_eagle_zero_drafts_still_requires_the_real_prefill_read(setup):
    _, spec, _, _ = setup
    runtime = DraftWindowRuntime(spec, resolve_draft_window_policy(requested_window_size=8, method="eagle3"))
    with (
        pytest.raises(RuntimeError, match="did not consume a read projection"),
        runtime.proposal_context(spec, batch(), num_speculative_tokens=0),
    ):
        pass
    assert runtime.pool.poisoned


def test_eagle_partial_proposal_cannot_omit_a_requested_step(setup):
    _, spec, _, _ = setup
    runtime = DraftWindowRuntime(spec, resolve_draft_window_policy(requested_window_size=8, method="eagle3"))
    with (
        pytest.raises(RuntimeError, match="every configured draft step"),
        runtime.proposal_context(spec, batch(), num_speculative_tokens=2),
    ):
        runtime.metadata_callback(0, 2)(0, common())
    assert runtime.pool.poisoned


def test_eagle_terminal_prefill_first_pass_stays_full_then_uses_window(setup):
    _, spec, _, _ = setup
    spec.input_buffers.positions = torch.tensor([15, 26])
    policy = resolve_draft_window_policy(requested_window_size=8, method="eagle3")
    runtime = DraftWindowRuntime(spec, policy)
    state = batch(computed=(12, 24))
    outputs = []
    with runtime.proposal_context(spec, state):
        for step in range(3):
            metadata = common(exact=(16 + step, 27 + step), upper=(20, 31))
            runtime.metadata_callback(step, 2)(0, metadata)
            outputs.append(metadata)
    assert [metadata.seq_lens[0].item() for metadata in outputs] == [16, 9, 10]


def test_eagle_first_pass_anchor_uses_live_queries_after_rejection(setup):
    _, spec, _, _ = setup
    canonical = torch.arange(72, dtype=torch.int32).reshape(6, 12)
    spec.block_tables.block_tables[0].gpu = canonical
    spec.block_tables.num_blocks.np = np.full((1, 6), 12, dtype=np.int32)
    spec.block_tables.kernel_block_sizes[0] = 128
    # Current real MRV2 preparation preserves native query padding. Row1 has
    # one rejected query: native extent1160, last live position1158 => E1159.
    spec.input_buffers = SimpleNamespace(positions=torch.tensor([1155, 1156, 1157, 1158, 1156, 1157, 1158, 1155]))
    spec.last_token_indices = torch.tensor([3, 6])
    runtime = DraftWindowRuntime(spec, resolve_draft_window_policy(requested_window_size=1035, method="eagle3"))
    state = batch(computed=(1300, 1300), prompt=(1147, 1148), scheduled=(4, 4), drafts=(3, 3))
    outputs = []
    with runtime.proposal_context(spec, state):
        for step, exact in enumerate(((1159, 1160), (1160, 1160), (1161, 1161))):
            metadata = common(exact=exact, upper=(1162, 1163))
            runtime.metadata_callback(step, 2)(0, metadata)
            outputs.append(metadata)
    # H=1159+3=1162; H-W=127 gives S0. Native padded1160 would
    # incorrectly cross the block boundary to S128 for row1.
    assert [metadata.seq_lens[:2].tolist() for metadata in outputs] == [[1159, 1160], [1160, 1160], [1161, 1161]]
    assert all(metadata.block_table_tensor[1, 0] == canonical[3, 0] for metadata in outputs)


def test_eagle_rejects_unsafe_window_at_startup(setup):
    _, spec, _, _ = setup
    policy = resolve_draft_window_policy(requested_window_size=6, method="eagle3")
    with pytest.raises(ValueError, match="first-pass query width"):
        DraftWindowRuntime(spec, policy)


def test_eagle_later_ar_uses_current_device_exact_with_frozen_anchor(setup, monkeypatch):
    _, spec, canonical, _ = setup
    spec.input_buffers.positions = torch.tensor([6, 7, 8, 9, 23, 24, 25, 26])
    spec.last_token_indices = torch.tensor([3, 7])
    runtime = DraftWindowRuntime(spec, resolve_draft_window_policy(requested_window_size=8, method="eagle3"))
    runtime._dense_device_reads = True  # Provider eligibility is tested separately.
    original = runtime._host_metadata_callback

    def legacy(step, *args, **kwargs):
        callback = original(step, *args, **kwargs)
        if step:

            def forbidden(*args):
                raise AssertionError("Later eager AR step fell back to a Host exact snapshot")

            return forbidden
        return callback

    monkeypatch.setattr(runtime, "_host_metadata_callback", legacy)
    state = batch(computed=(20, 30), prompt=(8, 8))
    state.idx_mapping = torch.from_numpy(state.idx_mapping_np).long()
    full = canonical.clone()
    with runtime.proposal_context(spec, state):
        first = common(exact=(10, 27), upper=(14, 30))
        first.query_start_loc_cpu = torch.tensor([0, 4, 8, 8])
        first.causal = True
        runtime.metadata_callback(0, 2)(0, first)
        for step, ends in enumerate(((11, 28), (12, 29)), 1):
            metadata = common(exact=ends, upper=(14, 30))
            metadata.query_start_loc_cpu = torch.tensor([0, 1, 2, 2])
            metadata.causal = True
            slots, positions, table = metadata.slot_mapping, metadata.positions, metadata.block_table_tensor
            runtime.metadata_callback(step, 2)(0, metadata)
            read = metadata._draft_window_device_read
            assert read.query_width == 1 and read.starts.tolist() == [4, 20]
            assert read.seq_lens.tolist() == [ends[0] - 4, ends[1] - 20]
            assert metadata.slot_mapping is slots and metadata.positions is positions
            assert metadata.block_table_tensor is table
    torch.testing.assert_close(canonical, full, atol=0, rtol=0)


@pytest.mark.parametrize("all_continuing", [False, True])
def test_eagle_continuing_prefill_later_ar_preserves_full_read(setup, all_continuing):
    _, spec, _, _ = setup
    spec.input_buffers.positions = torch.tensor([9, 10, 11, 24, 25, 26])
    spec.last_token_indices = torch.tensor([2, 5])
    runtime = DraftWindowRuntime(spec, resolve_draft_window_policy(requested_window_size=8, method="eagle3"))
    runtime._dense_device_reads = True
    state = batch(computed=(8, 24), prompt=(100, 100 if all_continuing else 16))
    state.idx_mapping = torch.from_numpy(state.idx_mapping_np).long()
    with runtime.proposal_context(spec, state):
        for step, ends in enumerate(((12, 27), (13, 28), (14, 29))):
            metadata = common(exact=ends, upper=(16, 31))
            width = 3 if step == 0 else 1
            metadata.query_start_loc_cpu = torch.tensor([0, width, 2 * width, 2 * width])
            metadata.causal = True
            runtime.metadata_callback(step, 2)(0, metadata)
            # Every AR read for an unfinished prompt is Full. The second
            # row also verifies the frozen decode anchor in a mixed batch.
            expected = [ends[0], ends[1] if all_continuing else ends[1] - 20]
            assert metadata.seq_lens[:2].tolist() == expected
            assert not hasattr(metadata, "_draft_window_device_read")


@pytest.mark.parametrize("groups", [(0,), (0, 1), (1, 0)])
@pytest.mark.parametrize("drafts", [None, (0, 0)])
def test_eagle_full_continuing_keeps_full_addressing_without_window_snapshot(setup, monkeypatch, groups, drafts):
    runtime, spec, canonical, completions = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    runtime.gids = groups
    state = batch(computed=(8, 24), prompt=(100, 100), scheduled=(4, 3), drafts=drafts)
    spec.max_model_len, spec.advance_draft_positions = 4096, True
    spec.block_tables.block_tables.append(SimpleNamespace(gpu=canonical.clone()))
    spec.block_tables.num_blocks.np = np.full((2, 6), 8, dtype=np.int32)
    spec.block_tables.kernel_block_sizes = [4, 4]

    def unexpected(*args, **kwargs):
        pytest.fail("Full continuing prefill requested a Window exact snapshot or read projection")

    monkeypatch.setattr(runtime, "_eagle_first_pass_lengths", unexpected)
    monkeypatch.setattr(runtime_module, "materialize_direct_prefix", unexpected)
    retained_common = []
    with runtime.proposal_context(spec, state):
        owner = runtime._owner
        for step in range(3):
            callback = runtime.metadata_callback(step, 2)
            exact = torch.tensor([12 + step, 27 + step, 0], dtype=torch.int32)
            for gid in groups:
                metadata = common()
                metadata.seq_lens = exact
                metadata.seq_lens_cpu = metadata._seq_lens_cpu = torch.tensor([14, 30, 0], dtype=torch.int32)
                before = dict(vars(metadata))
                callback(gid, metadata)
                assert vars(metadata).keys() == before.keys()
                assert all(
                    getattr(metadata, key) is value
                    for key, value in before.items()
                    if key not in ("seq_lens_cpu", "_seq_lens_cpu")
                )
                assert metadata.seq_lens_cpu.tolist() == [12 + step, 27 + step, 0]
                assert metadata._seq_lens_cpu is metadata.seq_lens_cpu
                retained_common.append((step, metadata))
        assert runtime._next_step == 3 and runtime._build_count == 3
        assert runtime._eagle_plans is None and runtime._eagle_first_pass_pair is None
        for step, metadata in retained_common:
            assert any(item is metadata for item in owner.retained)
            assert metadata.seq_lens_cpu.tolist() == [12 + step, 27 + step, 0]
    assert len(completions) == 1
    assert runtime.pool.reclaim_completed() == 0
    completions[0].complete = True
    assert runtime.pool.reclaim_completed() == 1


@pytest.mark.parametrize("changed", ["duplicate", "mapping", "lengths", "projected", "extent"])
def test_eagle_full_continuing_rejects_changed_preparation(setup, changed):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    runtime.gids = (0, 1)
    state = batch(computed=(8, 24), prompt=(100, 100), scheduled=(4, 3), drafts=None)
    spec.max_model_len, spec.advance_draft_positions = 4096, True
    with pytest.raises((RuntimeError, ValueError), match="Full prefill"), runtime.proposal_context(spec, state):
        callback = runtime.metadata_callback(0, 2)
        first, second = common(), common()
        second.seq_lens = first.seq_lens
        callback(0, first)
        if changed == "mapping":
            state.req_ids.reverse()
        elif changed == "lengths":
            second.seq_lens = second.seq_lens.clone()
        elif changed == "projected":
            second._draft_window_read = object()
        elif changed == "extent":
            second.num_reqs = 1
        callback(0 if changed == "duplicate" else 1, second)
    assert runtime.pool.poisoned


def test_eagle_full_continuing_with_placeholders_keeps_exact_fallback(setup, monkeypatch):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    spec.max_model_len, spec.advance_draft_positions = 4096, True
    state = batch(computed=(8, 24), prompt=(100, 100))
    monkeypatch.setattr(
        runtime,
        "_full_prefill_metadata_callback",
        lambda *args: pytest.fail("placeholder prompt took exact Host shortcut"),
    )
    with runtime.proposal_context(spec, state):
        for step in range(3):
            metadata = common(exact=(12 + step, 27 + step))
            runtime.metadata_callback(step, 2)(0, metadata)
            assert metadata.seq_lens[:2].tolist() == [12 + step, 27 + step]
            assert hasattr(metadata, "_draft_window_read")


def test_eagle_full_continuing_prompt_snapshot_clamps_at_model_limit(setup):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    spec.max_model_len, spec.advance_draft_positions = 32, True
    state = batch(computed=(8, 24), prompt=(32, 32), scheduled=(4, 7), drafts=None)
    with runtime.proposal_context(spec, state):
        for step in range(3):
            expected = (12 + step, min(31 + step, 32))
            metadata = common(exact=expected, upper=(16, 32))
            runtime.metadata_callback(step, 2)(0, metadata)
            assert metadata.seq_lens_cpu.tolist() == [*expected, 0]


def test_eagle_full_continuing_rejects_prompt_progress_change_between_ar_steps(setup):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    spec.max_model_len, spec.advance_draft_positions = 4096, True
    state = batch(computed=(8, 24), prompt=(100, 100), scheduled=(4, 3), drafts=None)
    with pytest.raises(RuntimeError, match="prompt preparation changed"), runtime.proposal_context(spec, state):
        runtime.metadata_callback(0, 2)(0, common())
        state.num_computed_prefill_tokens_np[0] += 1
        runtime.metadata_callback(1, 2)
    assert runtime._eagle_full_prefill_base is None
    assert runtime.pool.poisoned


def test_multiple_groups_project_same_logical_start_with_independent_backing(setup, monkeypatch):
    _, spec, canonical, _ = setup
    other_table = torch.arange(96, dtype=torch.int32).reshape(6, 16) + 100
    spec.block_tables.block_tables.append(SimpleNamespace(gpu=other_table))
    spec.block_tables.num_blocks.np = np.array([[8] * 6, [16] * 6], dtype=np.int32)
    spec.block_tables.kernel_block_sizes = [4, 2]
    monkeypatch.setattr(DraftWindowRuntime, "_validate_route", staticmethod(lambda spec: (0, 1)))
    policy = resolve_draft_window_policy(requested_window_size=8, method="dflash")
    runtime = DraftWindowRuntime(spec, policy)
    first, second = common(), common()
    # build_attn_metadata slices the same prepared seq_lens for every group.
    second.seq_lens = first.seq_lens
    with runtime.proposal_context(spec, batch()):
        callback = runtime.metadata_callback(3, 2)
        callback(0, first)
        callback(1, second)
    assert first.seq_lens.tolist() == second.seq_lens.tolist() == [12, 11, 0]
    # Same S=16 maps to column 4 in B=4 and column 8 in B=2.
    assert first.block_table_tensor[1, :3].tolist() == canonical[3, 4:7].tolist()
    assert second.block_table_tensor[1, :6].tolist() == other_table[3, 8:14].tolist()
    assert first.block_table_tensor.data_ptr() != second.block_table_tensor.data_ptr()


def test_missing_group_cannot_be_silently_dropped(setup, monkeypatch):
    _, spec, _, _ = setup
    monkeypatch.setattr(DraftWindowRuntime, "_validate_route", staticmethod(lambda spec: (0, 1)))
    policy = resolve_draft_window_policy(requested_window_size=8, method="dflash")
    runtime = DraftWindowRuntime(spec, policy)
    with pytest.raises(RuntimeError, match="did not consume"), runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(3, 2)(0, common())
    assert runtime.pool.poisoned


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_context_only_kv_write_keeps_completion_without_inventing_attention(setup, method):
    _, spec, canonical, completions = setup
    policy = resolve_draft_window_policy(requested_window_size=8, method=method)
    runtime = DraftWindowRuntime(spec, policy)
    before = canonical.clone()
    payload = torch.ones(8)
    with runtime.proposal_context(spec, batch(), context_kv_only=True):
        owner = runtime._owner
        assert owner is not None and owner.state is DraftWindowOwnerState.ACTIVE
        runtime.pool.retain(owner, payload)
    assert owner.state is DraftWindowOwnerState.SUBMITTED
    assert owner.completion is completions[-1]
    assert any(value is payload for value in owner.retained)
    assert runtime.pool.reclaim_completed() == 0
    torch.testing.assert_close(canonical, before, rtol=0, atol=0)
    completions[-1].complete = True
    assert runtime.pool.reclaim_completed() == 1
    assert owner.state is DraftWindowOwnerState.RELEASED


def test_context_only_scope_cannot_hide_an_actual_draft_read(setup):
    runtime, spec, _, _ = setup
    with (
        pytest.raises(RuntimeError, match="unexpectedly consumed"),
        runtime.proposal_context(spec, batch(), context_kv_only=True),
    ):
        runtime.metadata_callback(spec.num_query_per_req, 2)(0, common())
    assert runtime.pool.poisoned


def test_eagle_cannot_select_parallel_context_only_scope(setup):
    _, spec, _, _ = setup
    policy = resolve_draft_window_policy(requested_window_size=8, method="eagle3")
    runtime = DraftWindowRuntime(spec, policy)
    with (
        pytest.raises(ValueError, match="parallel drafter"),
        runtime.proposal_context(spec, batch(), context_kv_only=True),
    ):
        pytest.fail("Eagle3 must not enter a context-only scope")


def test_eagle_fused_mode_cannot_bypass_per_step_reads(setup):
    _, spec, _, _ = setup
    spec.use_fused_multi_step_decode = True
    policy = resolve_draft_window_policy(requested_window_size=8, method="eagle3")
    with pytest.raises(NotImplementedError, match="per-step metadata"):
        DraftWindowRuntime(spec, policy)


def test_actual_dense_builder_preserves_sp_padding_after_window_projection(setup, monkeypatch):
    """Execute the checkout's real builder body with CPU-only dependency seams.

    Pinning and phase classification are isolated; the production padding,
    tensor references and metadata assembly execute unmodified. This proves
    the Host builder contract, not distributed/NPU execution.
    """
    runtime, spec, _, _ = setup
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root / "vllm_ascend/attention/attention_v1.py").read_text())
    builder_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionMetadataBuilder"
    )
    build = next(node for node in builder_class.body if isinstance(node, ast.FunctionDef) and node.name == "build")
    utils_tree = ast.parse((root / "vllm_ascend/attention/utils.py").read_text())
    utility_functions = [
        node
        for node in utils_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_select_seq_lens", "uses_pcp_sharded_layout"}
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *utility_functions,
            build,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "CrossAttentionSpec": type("CrossAttentionSpec", (), {})}
    exec(compile(module, "actual_dense_builder_host_extract", "exec"), namespace)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    builder = SimpleNamespace(
        pcp_enabled=False,
        use_pcp=False,
        device=torch.device("cpu"),
        kv_cache_spec=object(),
        speculative_config=None,
        model_config=SimpleNamespace(runner_type="generate"),
        vllm_config=SimpleNamespace(use_v2_model_runner=True, model_config=object()),
        attn_mask_builder=SimpleNamespace(get_attention_mask=lambda *args: None),
        _split_decodes_and_prefills=lambda common: (2, 0, 6, 0),
        _build_backend_metadata=lambda *args, **kwargs: {},
        metadata_cls=SimpleNamespace,
    )
    metadata = common()
    metadata.num_reqs = 2
    metadata.num_actual_tokens = 6
    metadata.num_input_tokens = 8
    metadata.query_start_loc = metadata.query_start_loc[:3]
    metadata.query_start_loc_cpu = metadata.query_start_loc
    metadata.max_query_len = 3
    metadata.attn_state = None
    metadata.is_prefilling = torch.tensor([False, False])
    metadata.causal = False
    slots = metadata.slot_mapping
    with runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(3, 2)(0, metadata)
        result = namespace["build"](builder, 0, metadata)
        runtime.retain_metadata(result)
    assert result.actual_seq_lengths_q == [3, 6, 8]
    assert result.seq_lens_list == [12, 11, 1]
    assert result.block_tables.shape[0] == 3
    assert not result.block_tables[-1].any()
    assert result.slot_mapping.data_ptr() == slots.data_ptr()
    assert result.slot_mapping.numel() == result.num_actual_tokens == 6
    assert result.seq_lens_gpu is metadata.seq_lens
    assert result.query_start_loc_gpu is metadata.query_start_loc


def test_eagle_has_no_parallel_num_query_per_req_attribute(setup):
    runtime, spec, _, _ = setup
    runtime.method = "eagle3"
    runtime.static_k = 1
    del spec.num_query_per_req
    state = batch(computed=(24, 24))
    with runtime.proposal_context(spec, state):
        runtime.metadata_callback(0, 2)(0, common())


def test_dsa_builder_callback_retains_every_step_and_quarantines_error(setup):
    from vllm_ascend.worker.v2.spec_decode.draft_window_dsa import DSABuilderOwners

    runtime, spec, _, completions = setup

    class Builder:
        def __init__(self, spec, layers, config, device, metadata_cls=None):
            self.kv_cache_spec, self.vllm_config, self.device = spec, config, device
            self.metadata_cls, self.compressor_ratio = metadata_cls, 1
            self._device_metadata_enabled = False
            self.buffer = torch.zeros(4)

    original = Builder(None, ["draft"], None, "cpu")
    runtime._dsa_builders = DSABuilderOwners()
    runtime.dsa_layer_names = ("draft",)
    with runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(3, 2)(0, common())
        runtime.dsa_native_windows = {0: 4}
        owner = runtime._owner
        first = runtime.metadata_builder_callback(0)(0, original, ["draft"])
        second = runtime.metadata_builder_callback(1)(0, original, ["draft"])
        assert first is not second and first is not original
        assert any(value is first for value in owner.retained)
        assert any(value is second for value in owner.retained)
    assert not completions[0].is_complete()
    with pytest.raises(RuntimeError, match="after native"), runtime.proposal_context(spec, batch()):
        quarantined = runtime._owner
        later = runtime.metadata_builder_callback(0)(0, original, ["draft"])
        assert later is not first
        raise RuntimeError("after native")
    assert quarantined.state is DraftWindowOwnerState.QUARANTINED
    assert any(value is later for value in quarantined.retained)


def test_dsa_callback_uses_actual_native_phase_split_and_preserves_query(setup, monkeypatch):
    from torch.nn import functional as F

    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root / "vllm_ascend/attention/utils.py").read_text())
    node = next(item for item in tree.body if getattr(item, "name", None) == "split_decodes_and_prefills")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    ns = {"torch": torch, "F": F, "is_pd_decode_recompute_scheduler_enabled": lambda: False}
    exec(compile(module, "actual_phase_split", "exec"), ns)
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.utils", SimpleNamespace(**ns))
    runtime, spec, _, _ = setup
    runtime.dsa_native_windows = {0: 4}
    state = batch(computed=(24, 24))
    metadata = common()
    metadata.query_start_loc_cpu = metadata.query_start_loc
    metadata.context_parallel_metadata = None
    metadata.is_prefilling = torch.tensor([False, False, False])
    metadata.causal = False
    metadata.max_query_len, metadata.num_actual_tokens = 3, 6
    original_query, original_slots = metadata.query_start_loc, metadata.slot_mapping
    with runtime.proposal_context(spec, state):
        runtime.metadata_callback(3, 2)(0, metadata)
        assert metadata.query_start_loc is original_query
        assert metadata.slot_mapping is original_slots
        assert metadata.seq_lens.tolist() == [8, 11, 0]
        certificates = [value for value in runtime._owner.retained if isinstance(value, tuple)]
        assert certificates[0][0].first_token == 5
        assert certificates[0][1].first_token == 20


@pytest.mark.parametrize("with_prepared_exact", [False, True])
def test_prepared_exact_pair_is_explicit_and_retained_until_completion(setup, monkeypatch, with_prepared_exact):
    runtime, spec, _, completions = setup
    metadata = common()
    # An available CPU mirror is not a certificate; only the explicit pair is.
    metadata.seq_lens_cpu = metadata.seq_lens_cpu_upper_bound
    snapshot = metadata.seq_lens[:2].clone()
    original_to = torch.Tensor.to
    exact_copies = []

    def observe_to(tensor, *args, **kwargs):
        if kwargs == {"device": "cpu", "dtype": torch.int32, "copy": True}:
            exact_copies.append(tensor)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observe_to)
    pair = {"prepared_exact_cpu": snapshot, "prepared_exact_device": metadata.seq_lens} if with_prepared_exact else {}
    with runtime.proposal_context(spec, batch()):
        owner = runtime._owner
        runtime.metadata_callback(3, 2, **pair)(0, metadata)
    assert len(exact_copies) == (0 if with_prepared_exact else 1)
    assert metadata.seq_lens_cpu.tolist() == [12, 11, 0]
    if with_prepared_exact:
        assert any(value is snapshot for value in owner.retained)
        assert snapshot.tolist() == [12, 27]
    assert owner.state is DraftWindowOwnerState.SUBMITTED
    runtime.pool.reclaim_completed()
    assert owner.state is DraftWindowOwnerState.SUBMITTED
    completions[0].complete = True
    runtime.pool.reclaim_completed()
    assert owner.state is DraftWindowOwnerState.RELEASED
    assert owner.retained == []


@pytest.mark.parametrize(
    "invalid",
    ["unpaired", "float", "int64", "matrix", "short", "padded", "strided", "noncpu", "device-dtype", "device-short"],
)
def test_prepared_exact_pair_rejects_invalid_type_shape_or_device_prefix(setup, invalid):
    runtime, spec, _, _ = setup
    metadata = common()
    cpu, device = metadata.seq_lens[:2].clone(), metadata.seq_lens
    if invalid == "unpaired":
        device = None
    elif invalid == "float":
        cpu = cpu.float()
    elif invalid == "int64":
        cpu = cpu.long()
    elif invalid == "matrix":
        cpu = cpu.reshape(1, 2)
    elif invalid == "short":
        cpu = cpu[:1]
    elif invalid == "padded":
        cpu = metadata.seq_lens.clone()
    elif invalid == "strided":
        cpu = torch.tensor([12, 0, 27, 0], dtype=torch.int32)[::2]
    elif invalid == "noncpu":
        cpu = torch.empty(2, dtype=torch.int32, device="meta")
    elif invalid == "device-dtype":
        device = device.long()
    elif invalid == "device-short":
        device = device[:1]
    with pytest.raises(ValueError, match="Prepared exact"), runtime.proposal_context(spec, batch()):
        runtime.metadata_callback(3, 2, prepared_exact_cpu=cpu, prepared_exact_device=device)


@pytest.mark.parametrize("change", ["mapping", "device"])
def test_prepared_exact_pair_cannot_be_applied_to_another_input(setup, change):
    runtime, spec, _, _ = setup
    state, metadata = batch(), common()
    snapshot = metadata.seq_lens[:2].clone()
    expected = "request mapping" if change == "mapping" else "device input"
    with pytest.raises((RuntimeError, ValueError), match=expected), runtime.proposal_context(spec, state):
        owner = runtime._owner
        callback = runtime.metadata_callback(3, 2, prepared_exact_cpu=snapshot, prepared_exact_device=metadata.seq_lens)
        if change == "mapping":
            state.req_ids.reverse()
            state.idx_mapping_np[:] = state.idx_mapping_np[::-1]
        else:
            metadata.seq_lens = metadata.seq_lens.clone()
        callback(0, metadata)
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert any(value is snapshot for value in owner.retained)


def test_prepared_exact_pair_keeps_owner_and_step_guards(setup):
    runtime, spec, _, _ = setup
    metadata = common()
    snapshot = metadata.seq_lens[:2].clone()
    pair = {"prepared_exact_cpu": snapshot, "prepared_exact_device": metadata.seq_lens}
    with pytest.raises(RuntimeError, match="no active proposal"):
        runtime.metadata_callback(3, 2, **pair)
    with runtime.proposal_context(spec, batch()):
        with pytest.raises(ValueError, match="actual query width"):
            runtime.metadata_callback(8, 2, **pair)
        callback = runtime.metadata_callback(3, 2, **pair)
        callback(0, metadata)
    with pytest.raises(RuntimeError, match="stale proposal"):
        callback(0, common())
    with pytest.raises(RuntimeError, match="stale proposal"), runtime.proposal_context(spec, batch()):
        callback(0, common())


@pytest.mark.parametrize("upper", [(21, 30), (33, 33)])
def test_dense_device_callback_or_exact_fallback_keeps_the_same_window(setup, upper):
    runtime, spec, canonical, completions = setup
    runtime._dense_device_reads = True
    current_batch = batch(computed=(16, 24))
    current_batch.idx_mapping = torch.tensor(current_batch.idx_mapping_np)
    metadata = common(exact=(20, 27), upper=upper)
    metadata.query_start_loc_cpu = metadata.query_start_loc.clone()
    metadata.causal = False
    full_table, slots, positions = metadata.block_table_tensor, metadata.slot_mapping, metadata.positions
    original = canonical.clone()
    with runtime.proposal_context(spec, current_batch):
        callback = runtime.metadata_callback(3, 2)
        callback(0, metadata)
        runtime.retain_metadata(metadata)
        assert metadata.seq_lens.tolist() == [8, 11, 0]
        if max(upper) <= 32:
            read = metadata._draft_window_device_read
            assert read.starts.tolist() == [12, 16]
            assert metadata.seq_lens_cpu is None
            assert metadata.block_table_tensor is full_table
            assert any(value is read for value in runtime._owner.retained)
        else:
            assert not hasattr(metadata, "_draft_window_device_read")
            assert metadata.seq_lens_cpu.tolist() == [8, 11, 0]
        assert metadata.slot_mapping is slots and metadata.positions is positions
    torch.testing.assert_close(canonical, original, rtol=0, atol=0)
    assert len(completions) == 1
    with pytest.raises(RuntimeError, match="stale"):
        callback(0, common())


@pytest.mark.parametrize("drafts", [None, (0, 0)])
@pytest.mark.parametrize("all_terminal", [False, True])
@pytest.mark.parametrize("max_len", [4096, 2178])
def test_eagle_prompt_only_terminal_rows_reuse_exact_without_d2h(setup, monkeypatch, drafts, all_terminal, max_len):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    runtime.window_size, runtime.anchor_block_span = 1035, 128
    spec.max_model_len, spec.advance_draft_positions = max_len, True
    table = torch.arange(6 * 32, dtype=torch.int32).reshape(6, 32)
    spec.block_tables.block_tables = [SimpleNamespace(gpu=table)]
    spec.block_tables.kernel_block_sizes = [128]
    spec.block_tables.num_blocks.np = np.full((1, 6), 32, dtype=np.int32)
    state = batch(
        computed=(1792, 1920), prompt=(2047, 2177 if all_terminal else 2303), scheduled=(255, 257), drafts=drafts
    )
    state.idx_mapping = torch.tensor(state.idx_mapping_np, dtype=torch.int64)
    state.query_start_loc_np = np.array([0, 255, 512])
    state.cu_num_logits_np = np.array([0, 1, 2 if all_terminal else 1])
    spec.input_buffers.positions = torch.cat((torch.arange(1792, 2047), torch.arange(1920, 2177)))
    spec.last_token_indices = torch.tensor([254, 511])
    original_to = torch.Tensor.to

    def no_exact_cpu_copy(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("copy") is True:
            pytest.fail("Known prompt producer still copied current device exact lengths")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", no_exact_cpu_copy)
    before = table.clone()
    with runtime.proposal_context(spec, state):
        for step in range(3):
            ends = (min(2047 + step, max_len), min(2177 + step, max_len))
            metadata = common(exact=ends, upper=(2050 + step, 2180 + step))
            metadata.query_start_loc_cpu = torch.tensor([0, 255, 512, 512] if step == 0 else [0, 1, 2, 2])
            metadata.causal = True
            slots, positions = metadata.slot_mapping, metadata.positions
            apply = runtime.metadata_callback(step, 2)
            apply(0, metadata)
            expected_starts = [0, 0] if step == 0 else [896, 1024 if all_terminal else 0]
            if all_terminal and step > 0:
                assert metadata._draft_window_device_read.starts.tolist() == expected_starts
            else:
                assert metadata._draft_window_read.starts_cpu.tolist() == expected_starts
            assert metadata.seq_lens[:2].tolist() == [
                ends[0] - expected_starts[0],
                ends[1] - expected_starts[1],
            ]
            assert metadata.slot_mapping is slots and metadata.positions is positions
            assert torch.equal(table, before)
    assert runtime._eagle_prompt_exact_base is None


@pytest.mark.parametrize(
    "invalid",
    ["placeholder", "query_width", "terminal_logits", "past_prompt", "decode_row", "past_limit", "replicated_pcp"],
)
def test_eagle_prompt_only_ambiguous_producer_keeps_exact_fallback(setup, invalid):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    spec.max_model_len, spec.advance_draft_positions = 100, True
    state = batch(computed=(8, 24), prompt=(12, 100), scheduled=(4, 3), drafts=None)
    state.query_start_loc_np = np.array([0, 4, 7])
    state.cu_num_logits_np = np.array([0, 1, 1])
    if invalid == "placeholder":
        state.num_draft_tokens_per_req = np.array([1, 0])
    elif invalid == "query_width":
        state.query_start_loc_np[1] = 3
    elif invalid == "terminal_logits":
        state.cu_num_logits_np[1:] = 2
    elif invalid == "past_prompt":
        state.prefill_len_np[0] = 11
    elif invalid == "decode_row":
        state.is_prefilling_np[0] = False
    elif invalid == "past_limit":
        spec.max_model_len = 11
    else:
        spec.replicated_pcp = True
    with pytest.raises(RuntimeError, match="did not consume a read projection"), runtime.proposal_context(spec, state):
        assert runtime._prompt_only_exact_lengths(0, 2) is None
        assert runtime._eagle_prompt_exact_base is None
    assert runtime.pool.poisoned


def test_eagle_prompt_only_snapshot_cpu_and_generation_scope(setup):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    spec.max_model_len, spec.advance_draft_positions = 100, True
    state = batch(computed=(8, 24), prompt=(12, 100), scheduled=(4, 3), drafts=None)
    state.query_start_loc_np = np.array([0, 4, 7])
    state.cu_num_logits_np = np.array([0, 1, 1])
    with pytest.raises(RuntimeError, match="preparation changed"), runtime.proposal_context(spec, state):
        with torch.device("meta"):
            snapshot = runtime._prompt_only_exact_lengths(0, 2)
        assert snapshot.device.type == "cpu" and snapshot.tolist() == [12, 27]
        state.num_scheduled_tokens[1] += 1
        state.query_start_loc_np[-1] += 1
        runtime._prompt_only_exact_lengths(1, 2)
    assert runtime.pool.poisoned and runtime._eagle_prompt_exact_base is None


def test_eagle_mixed_decode_prompt_without_exact_copy(setup, monkeypatch):
    runtime, spec, _, _ = setup
    runtime.method, runtime.static_k, runtime._dense_device_reads = "eagle3", 3, True
    runtime.window_size, runtime.anchor_block_span = 1035, 128
    spec.max_model_len, spec.advance_draft_positions = 4096, True
    table = torch.arange(6 * 32, dtype=torch.int32).reshape(6, 32)
    spec.block_tables.block_tables = [SimpleNamespace(gpu=table)]
    spec.block_tables.kernel_block_sizes = [128]
    spec.block_tables.num_blocks.np = np.full((1, 6), 32, dtype=np.int32)
    state = batch(computed=(1792, 1920), prompt=(1, 2303), scheduled=(4, 257), drafts=(3, 0))
    state.idx_mapping = torch.tensor(state.idx_mapping_np, dtype=torch.int64)
    spec.input_buffers.positions = torch.cat((torch.arange(1792, 1796), torch.arange(1920, 2177)))
    spec.last_token_indices = torch.tensor([2, 260])  # One rejected decoder query.
    original_to = torch.Tensor.to

    def no_exact_cpu_copy(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("copy") is True:
            pytest.fail("Mixed decode/prompt metadata copied current device exact")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", no_exact_cpu_copy)
    before = table.clone()
    with runtime.proposal_context(spec, state):
        for step, ends in enumerate(((1796, 2177), (1796, 2178), (1797, 2179))):
            metadata = common(exact=ends, upper=(1800, 2180))
            boundaries = [0, 4, 261, 261] if step == 0 else [0, 1, 2, 2]
            metadata.query_start_loc_cpu = torch.tensor(boundaries, dtype=torch.int32)
            metadata.query_start_loc = metadata.query_start_loc_cpu.clone()
            metadata.causal = True
            slots, positions = metadata.slot_mapping, metadata.positions
            runtime.metadata_callback(step, 2)(0, metadata)
            read = metadata._draft_window_device_read
            assert read.starts.tolist() == [640, 0]
            assert read.seq_lens.tolist() == [1155 if step == 0 else ends[0] - 640, ends[1]]
            assert read.num_query_tokens == (261 if step == 0 else 2)
            assert read.capacity == 2208
            assert metadata.slot_mapping is slots and metadata.positions is positions
            assert torch.equal(table, before)


@pytest.mark.parametrize("method", ["dflash", "dspark"])
def test_parallel_continuing_prefill_reuses_certified_prompt_exact_without_full_gather(setup, monkeypatch, method):
    runtime, spec, _, _ = setup
    runtime.method, runtime._dense_device_reads = method, True
    runtime.window_size, runtime.anchor_block_span = 1024, 128
    spec.max_model_len, spec.num_query_per_req = 4096, 8
    table = torch.arange(6 * 32, dtype=torch.int32).reshape(6, 32)
    spec.block_tables.block_tables = [SimpleNamespace(gpu=table)]
    spec.block_tables.kernel_block_sizes = [128]
    spec.block_tables.num_blocks.np = np.full((1, 6), 32, dtype=np.int32)
    state = batch(computed=(1136, 1264), prompt=(4096, 4096), scheduled=(7, 9), drafts=None)
    state.query_start_loc_np = np.array([0, 7, 16])
    state.idx_mapping = torch.tensor(state.idx_mapping_np)
    metadata = common(exact=(1151, 1281), upper=(1152, 1282))
    metadata.query_start_loc_cpu = torch.tensor([0, 8, 16, 16], dtype=torch.int32)
    spec.input_buffers.seq_lens = metadata.seq_lens
    original_to = torch.Tensor.to

    def no_exact_cpu_copy(tensor, *args, **kwargs):
        if kwargs.get("device") == "cpu" and kwargs.get("copy") is True:
            pytest.fail("Certified continuing prompt producer copied device exact")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", no_exact_cpu_copy)
    before = table.clone()
    slots, positions = metadata.slot_mapping, metadata.positions
    with runtime.proposal_context(spec, state):
        runtime.metadata_callback(8, 2)(0, metadata)
        assert not hasattr(metadata, "_draft_window_device_read"), "Full prompt unexpectedly gathered private FP32 KV"
        read = metadata._draft_window_read
        assert read.starts_cpu.tolist() == [0, 0]
        assert read.seq_lens.cpu().tolist() == [1151, 1281, 0]
        assert read.exact_cpu_for(metadata)[:2].tolist() == [1151, 1281]
        assert metadata.slot_mapping is slots and metadata.positions is positions
        assert torch.equal(table, before)


@pytest.mark.parametrize("invalid", ["mixed_decode", "terminal", "placeholder", "query_width", "replicated_pcp"])
def test_parallel_prompt_producer_rejects_uncertified_preparation(setup, invalid):
    runtime, spec, _, _ = setup
    runtime._dense_device_reads = True
    spec.max_model_len, spec.num_query_per_req = 4096, 8
    state = batch(computed=(1136, 1264), prompt=(4096, 4096), scheduled=(7, 9), drafts=None)
    state.query_start_loc_np = np.array([0, 7, 16])
    if invalid == "mixed_decode":
        state.is_prefilling_np[0] = False
    elif invalid == "terminal":
        state.prefill_len_np[0] = 1143
    elif invalid == "placeholder":
        state.num_draft_tokens_per_req = np.array([1, 0])
    elif invalid == "query_width":
        state.query_start_loc_np[-1] += 1
    else:
        spec.replicated_pcp = True
    with pytest.raises(RuntimeError, match="did not consume"), runtime.proposal_context(spec, state):
        assert runtime._parallel_prefill_exact_lengths(2) is None


def test_mla_partial_rotary_copy_failure_keeps_submitted_operands(setup, monkeypatch):
    runtime, spec, canonical, _ = setup
    runtime.mla_layer_names = ("draft",)
    cos, sin = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    metadata = {"draft": SimpleNamespace(external_flashmla=None, decode=SimpleNamespace(cos=cos, sin=sin))}
    original_clone, original_copy = torch.Tensor.clone, torch.Tensor.copy_
    full = canonical.clone()
    owner: DraftWindowOwner | None = None

    def clone(source, *args, **kwargs):
        if source is sin:
            raise RuntimeError("MLA second RoPE copy failed")
        return original_clone(source, *args, **kwargs)

    def copy(destination, source, *args, **kwargs):
        if source is cos or source is sin:
            assert owner is not None
            assert any(value is destination for value in owner.retained)
            assert any(value is source for value in owner.retained)
            if source is sin:
                raise RuntimeError("MLA second RoPE copy failed")
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(RuntimeError, match="MLA second RoPE copy failed"), runtime.proposal_context(spec, batch()):
        owner = runtime._owner
        runtime.retain_metadata(metadata)
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    first_copy = metadata["draft"].decode.cos
    assert first_copy is not cos
    assert any(value is first_copy for value in owner.retained)
    with pytest.raises(RuntimeError, match="quarantined"):
        runtime.pool.acquire()
    torch.testing.assert_close(canonical, full, rtol=0, atol=0)
