# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm_ascend.worker.v1.spec_decode import utils


@pytest.fixture
def parallel_state(monkeypatch):
    state = SimpleNamespace(_TP=object(), _TP_STATE_PATCHED=False)
    state.get_tp_group = lambda: state._TP
    monkeypatch.setattr(utils, "_ps", state)
    return state


@pytest.mark.parametrize("previously_patched", [False, True])
@pytest.mark.parametrize("raise_in_body", [False, True])
def test_tp_group_context_restores_previous_state(parallel_state, previously_patched, raise_in_body):
    original_group = parallel_state._TP
    parallel_state._TP_STATE_PATCHED = previously_patched
    draft_group = object()

    def run_context():
        with utils.patch_tensor_parallel_group(draft_group):
            assert parallel_state._TP is draft_group
            assert parallel_state._TP_STATE_PATCHED is True
            if raise_in_body:
                raise RuntimeError("draft failed")

    if raise_in_body:
        with pytest.raises(RuntimeError, match="draft failed"):
            run_context()
    else:
        run_context()

    assert parallel_state._TP is original_group
    assert parallel_state._TP_STATE_PATCHED is previously_patched


@pytest.mark.parametrize("raise_in_inner", [False, True])
def test_nested_tp_group_context_keeps_outer_patch_active(parallel_state, raise_in_inner):
    original_group = parallel_state._TP
    outer_group, inner_group = object(), object()

    def run_inner():
        with utils.patch_tensor_parallel_group(inner_group):
            assert parallel_state._TP is inner_group
            assert parallel_state._TP_STATE_PATCHED is True
            if raise_in_inner:
                raise RuntimeError("inner failed")

    with utils.patch_tensor_parallel_group(outer_group):
        if raise_in_inner:
            with pytest.raises(RuntimeError, match="inner failed"):
                run_inner()
        else:
            run_inner()
        assert parallel_state._TP is outer_group
        assert parallel_state._TP_STATE_PATCHED is True

    assert parallel_state._TP is original_group
    assert parallel_state._TP_STATE_PATCHED is False


def test_tp_group_context_defaults_missing_flag_to_false(parallel_state):
    original_group = parallel_state._TP
    del parallel_state._TP_STATE_PATCHED

    with utils.patch_tensor_parallel_group(object()):
        assert parallel_state._TP_STATE_PATCHED is True

    assert parallel_state._TP is original_group
    assert parallel_state._TP_STATE_PATCHED is False
