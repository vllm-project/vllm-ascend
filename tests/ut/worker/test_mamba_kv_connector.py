# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.worker.v2.kv_connector import MambaCopyKVConnector


@pytest.mark.parametrize("sync_load", [True, False])
def test_bulk_copy_reads_state_after_pre_forward(monkeypatch, sync_load):
    """A remote load must update the source before it is copied for forward."""
    from vllm_ascend.worker.v2.model_states import mamba_hybrid
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

    memory = {"source": 7, "running": 0}
    events = []

    class DecisionKernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: events.append("decision")

    def copy(*args):
        events.append("copy")
        memory["running"] = memory["source"]

    state = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    state.__dict__.update(
        _align_mode=True,
        _pending_bulk_copy=None,
        _mamba_state_idx_gpu=None,
        _mamba_src_col_gpu=None,
        _mamba_src_off_gpu=None,
        num_accepted_tokens_gpu=None,
        _finish_previous_layerwise_mamba_copy=lambda: None,
        _get_mamba_group_info=lambda cfg: (None, SimpleNamespace(block_size=384)),
        _ensure_align_ctx=lambda *args: SimpleNamespace(run_fused_precopy=copy),
        _prepare_layerwise_mamba_copy=lambda *args: False,
    )
    monkeypatch.setattr(mamba_hybrid, "preprocess_mamba_align_fused_kernel", DecisionKernel())
    monkeypatch.setattr(mamba_hybrid, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(
        mamba_hybrid,
        "get_kv_transfer_group",
        lambda: SimpleNamespace(requires_mamba_state_copy_after_load=True),
    )
    batch = SimpleNamespace(num_reqs=1, idx_mapping=None, query_start_loc=None)
    AscendMambaHybridModelState.preprocess_state(state, batch, (), None, None)
    assert events == ["decision"]

    def load(scheduler_output, **kwargs):
        events.append("pre_forward")
        if sync_load:
            memory["source"] = 42

    inner = SimpleNamespace(pre_forward=load)
    wrapped = MambaCopyKVConnector(inner, lambda: AscendMambaHybridModelState.copy_state_after_kv_load(state))
    wrapped.pre_forward(object(), attn_metadata=object())
    assert events == ["decision", "pre_forward", "copy"]
    assert memory["running"] == (42 if sync_load else 7)
    # The normal forward can now consume the state; a second hook is a no-op.
    AscendMambaHybridModelState.copy_state_after_kv_load(state)
    assert events.count("copy") == 1
    assert state._pending_bulk_copy is None


def test_load_failure_does_not_copy_state():
    inner = SimpleNamespace(pre_forward=Mock(side_effect=ValueError("load failed")))
    copy = Mock()
    wrapped = MambaCopyKVConnector(inner, copy)
    with pytest.raises(ValueError, match="load failed"):
        wrapped.pre_forward(object())
    copy.assert_not_called()


def test_pre_forward_passes_input_batch_and_metadata():
    inner = SimpleNamespace(pre_forward=Mock())
    copy = Mock()
    wrapped = MambaCopyKVConnector(inner, copy)
    scheduler_output, input_batch = object(), object()
    attn_metadata: dict[str, object] = {}
    wrapped.pre_forward(scheduler_output, input_batch, attn_metadata=attn_metadata)
    inner.pre_forward.assert_called_once_with(scheduler_output, input_batch, attn_metadata=attn_metadata)
    copy.assert_called_once_with()


def test_post_forward_does_not_inject_removed_arguments():
    output = object()

    class Inner:
        def post_forward(self, finished_req_ids):
            assert finished_req_ids == {"done"}
            return output

    wrapped = MambaCopyKVConnector(Inner(), Mock())
    assert wrapped.post_forward({"done"}) is output


def test_connector_lifecycle_is_forwarded():
    output = object()
    inner = SimpleNamespace(
        pre_forward=Mock(),
        finish_forward=Mock(),
        post_forward=Mock(return_value=output),
        no_forward=Mock(return_value=output),
        set_disabled=Mock(),
        reset_capture_state=Mock(),
    )
    copy = Mock()
    wrapped = MambaCopyKVConnector(inner, copy)
    wrapped.finish_forward()
    assert wrapped.post_forward({"done"}, wait_for_save=False) is output
    tokenless_step = object()
    assert wrapped.no_forward(tokenless_step) is output
    wrapped.set_disabled(True)
    wrapped.reset_capture_state()
    inner.finish_forward.assert_called_once_with()
    inner.post_forward.assert_called_once_with({"done"}, wait_for_save=False)
    inner.no_forward.assert_called_once_with(tokenless_step)
    inner.set_disabled.assert_called_once_with(True)
    inner.reset_capture_state.assert_called_once_with()
    copy.assert_not_called()


def test_wrappers_keep_runner_state_independent():
    copy_a, copy_b = Mock(), Mock()
    inner = SimpleNamespace(pre_forward=Mock())
    first = MambaCopyKVConnector(inner, copy_a)
    second = MambaCopyKVConnector(inner, copy_b)
    first.pre_forward(None)
    copy_a.assert_called_once_with()
    copy_b.assert_not_called()
    second.pre_forward(None)
    assert copy_a.call_count == 1
    copy_b.assert_called_once_with()


@pytest.mark.parametrize("failed", [True, False])
def test_finish_execution_cleans_failed_copy_and_finishes_metadata(failed):
    from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

    state = AscendMambaHybridModelState.__new__(AscendMambaHybridModelState)
    ctx = Mock()
    pending = (ctx, 1, Mock())
    state._pending_bulk_copy = pending
    state.device_metadata = Mock()

    state.finish_execution(failed=failed)

    assert state._pending_bulk_copy is (None if failed else pending)
    state.device_metadata.finish.assert_called_once_with()
    if failed:
        # A failed load must not leave stale state to copy on a later hook.
        state.copy_state_after_kv_load()
        ctx.run_fused_precopy.assert_not_called()
