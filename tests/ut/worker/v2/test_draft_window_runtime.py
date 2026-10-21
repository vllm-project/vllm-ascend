# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests at the proposal and metadata callback seams."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import DraftWindowOwnerState
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
        vllm_config=SimpleNamespace(max_concurrent_batches=2),
        num_query_per_req=3,
        num_speculative_steps=3,
        use_fused_multi_step_decode=False,
        block_tables=SimpleNamespace(
            block_tables=[SimpleNamespace(gpu=canonical)],
            num_blocks=SimpleNamespace(np=np.full((1, 6), 8, dtype=np.int32)),
            kernel_block_sizes=[4],
        ),
    )
    policy = resolve_draft_window_policy(requested_window_size=8, method="dflash")
    runtime = DraftWindowRuntime(spec, policy)
    return runtime, spec, canonical, completions


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
    assert len(owner.retained) == 9
    assert len(completions) == 1


def test_eagle_terminal_prefill_first_pass_stays_full_then_uses_window(setup):
    _, spec, _, _ = setup
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


def test_eagle_rejects_unsafe_window_at_startup(setup):
    _, spec, _, _ = setup
    policy = resolve_draft_window_policy(requested_window_size=6, method="eagle3")
    with pytest.raises(ValueError, match="first-pass query width"):
        DraftWindowRuntime(spec, policy)


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
    select_lengths = next(
        node for node in utils_tree.body if isinstance(node, ast.FunctionDef) and node.name == "_select_seq_lens"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            select_lengths,
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
