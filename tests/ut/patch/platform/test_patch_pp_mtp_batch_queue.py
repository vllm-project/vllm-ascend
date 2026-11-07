# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the PP + speculation batch-queue serialization patch.

``_patch_engine_core_batch_queue`` wraps ``EngineCore.__init__`` and collapses
the batch queue for MRV1 + PP>1 + speculation where the scheduler would
otherwise advance a request before the previous verify step's output has been
consumed. Only the decision gate is exercised here; the end-to-end behaviour is
covered by ``tests/e2e/pull_request/two_card/test_mrv1_pp_mtp.py``.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.engine.core import EngineCore

from vllm_ascend.patch.platform.patch_pp_mtp import _patch_engine_core_batch_queue


@pytest.fixture
def patched_init():
    """Replace the real (heavy) EngineCore.__init__ with a stub before patching."""
    original = EngineCore.__init__
    EngineCore.__init__ = lambda self, *args, **kwargs: None
    try:
        _patch_engine_core_batch_queue()
        yield
    finally:
        EngineCore.__init__ = original


def _config(*, spec=True, pp=2, async_scheduling=False, v2=False, kv_role=None):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp") if spec else None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        use_v2_model_runner=v2,
        kv_transfer_config=SimpleNamespace(kv_role=kv_role) if kv_role is not None else None,
    )


def _make_core(config, *, queue_size=2):
    core = SimpleNamespace(
        vllm_config=config,
        batch_queue_size=queue_size,
        batch_queue=object() if queue_size > 1 else None,
    )
    core.step = lambda: "step"
    core.step_fn = "step_with_batch_queue"
    return core


def test_mrv1_pp_spec_serializes_batch_queue(patched_init):
    core = _make_core(_config())
    EngineCore.__init__(core, core.vllm_config)

    assert core.batch_queue_size == 1
    assert core.batch_queue is None
    assert core.step_fn == core.step


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"spec": False},
        {"pp": 1},
        {"async_scheduling": True},
        {"v2": True},
        {"kv_role": "kv_producer"},
    ],
)
def test_gated_off_configurations_keep_batch_queue(patched_init, config_kwargs):
    core = _make_core(_config(**config_kwargs))
    EngineCore.__init__(core, core.vllm_config)

    assert core.batch_queue_size == 2
    assert core.batch_queue is not None
    assert core.step_fn == "step_with_batch_queue"


def test_already_serialized_queue_is_untouched(patched_init):
    core = _make_core(_config(), queue_size=1)
    EngineCore.__init__(core, core.vllm_config)

    assert core.batch_queue_size == 1
    assert core.batch_queue is None
    assert core.step_fn == "step_with_batch_queue"


def test_missing_config_is_noop(patched_init):
    core = SimpleNamespace(batch_queue_size=2, batch_queue=object())
    core.step = lambda: "step"
    core.step_fn = "step_with_batch_queue"
    EngineCore.__init__(core)

    assert core.batch_queue_size == 2
    assert core.batch_queue is not None


def test_patch_is_idempotent(patched_init):
    _patch_engine_core_batch_queue()
    once = EngineCore.__init__
    _patch_engine_core_batch_queue()

    assert EngineCore.__init__ is once
