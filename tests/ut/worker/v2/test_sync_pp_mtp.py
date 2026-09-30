# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for synchronous MRV2 speculative PP batch ownership."""

import pickle
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from vllm_ascend.patch.platform import patch_pp_mtp as patch
from vllm_ascend.worker.v2 import model_runner as runner_mod
from vllm_ascend.worker.v2.pp_utils import attach_batch_draft_tokens


@pytest.fixture(autouse=True)
def _release_pp_path(monkeypatch):
    """Exercise the release workaround independently of the installed version."""
    from vllm_ascend.worker.v2 import pp_utils

    monkeypatch.setattr(pp_utils, "use_legacy_spec_pp", lambda: True)


def config(async_scheduling=False, speculative=True, v2=True):
    return SimpleNamespace(
        use_v2_model_runner=v2,
        speculative_config=SimpleNamespace(method="mtp") if speculative else None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        model_config=SimpleNamespace(architecture="Glm5NextForConditionalGeneration"),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        kv_transfer_config=None,
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("use_pp", [False, True])
def test_v2_scope(async_scheduling, speculative, use_pp):
    cfg = config(async_scheduling, speculative)
    assert patch._use_v2_sync_pp_mtp_runtime_patch(cfg, use_pp) == (use_pp and speculative and not async_scheduling)
    # V2 must never enable the V1 IPC transport.
    assert not patch._use_pp_ipc_runtime_patch(cfg, use_pp)
    cfg.kv_transfer_config = SimpleNamespace(kv_role="kv_producer")
    assert not patch._use_v2_sync_pp_mtp_runtime_patch(cfg, use_pp)


def test_snapshot_survives_next_batch_and_ipc():
    rows = [[101], [202]]
    handler = SimpleNamespace(get_draft_tokens=lambda: DraftTokenIds(["b", "a"], rows))
    first = SimpleNamespace(
        model_runner_output=ModelRunnerOutput(
            req_ids=["a", "b"], req_id_to_index={"a": 0, "b": 1}, sampled_token_ids=[[11], [22]]
        )
    )
    attach_batch_draft_tokens(first, handler)
    rows[0][0] = 999
    second = SimpleNamespace(
        model_runner_output=ModelRunnerOutput(req_ids=["b"], req_id_to_index={"b": 0}, sampled_token_ids=[[33]])
    )
    attach_batch_draft_tokens(second, handler)
    restored = pickle.loads(pickle.dumps(first.model_runner_output))
    assert restored.spec_token_ids == [[202], [101]]
    assert second.model_runner_output.spec_token_ids == [[999]]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_runner_attaches_only_sync_batch(monkeypatch, async_scheduling):
    output = SimpleNamespace(
        model_runner_output=ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[[11]])
    )
    monkeypatch.setattr(runner_mod.GPUModelRunner, "sample_tokens", lambda *args: output)
    runner = runner_mod.NPUModelRunner.__new__(runner_mod.NPUModelRunner)
    runner.pcp_manager = None
    runner.is_last_pp_rank = True
    runner.use_spec_pp = True
    runner.vllm_config = config(async_scheduling)
    runner.pp_handler = SimpleNamespace(broadcast_drafts=Mock())
    runner._restore_replicated_draft_target_states = Mock()
    runner.draft_tokens_handler = SimpleNamespace(get_draft_tokens=Mock(return_value=DraftTokenIds(["a"], [[-1]])))
    assert runner.sample_tokens(None) is output
    runner.pp_handler.broadcast_drafts.assert_called_once()
    assert runner.draft_tokens_handler.get_draft_tokens.call_count == (0 if async_scheduling else 1)
    if not async_scheduling:
        assert output.model_runner_output.spec_token_ids == [[-1]]


def test_reject_inconsistent_verification_before_kernel(monkeypatch):
    runner = runner_mod.NPUModelRunner.__new__(runner_mod.NPUModelRunner)
    runner._update_seq_lens_cpu = Mock()
    runner.device = torch.device("cpu")
    runner.vllm_config = config()
    runner.input_buffers = SimpleNamespace(seq_lens_np=np.array([21]))
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    monkeypatch.setattr(runner_mod, "async_copy_to_gpu", lambda a, **kw: torch.from_numpy(a))
    monkeypatch.setattr(runner_mod, "build_attn_state", lambda *args: None)
    batch = SimpleNamespace(
        num_tokens=1, req_ids=["a"], num_scheduled_tokens=np.array([1]), idx_mapping_np=np.array([0])
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"a": [-1]})
    with pytest.raises(ValueError, match=r"scheduled=\[1\], logits=\[2\]"):
        runner.prepare_inputs(scheduled, batch, SimpleNamespace(num_tokens=1))


def test_engine_queue_does_not_schedule_draft_before_sampled_output(monkeypatch):
    # Use the real EngineCore batch queue and patch hooks. Stub only the
    # scheduler's KV/cache accounting and the device executor.
    req = SimpleNamespace(
        num_tokens=20,
        num_computed_tokens=0,
        spec_token_ids=[],
        is_prefill_chunk=True,
        next_decode_eligible_step=0,
        is_finished=lambda: False,
    )

    def after_schedule(self, out):
        for count in out.num_scheduled_tokens.values():
            req.num_computed_tokens += count
            req.is_prefill_chunk = req.num_computed_tokens < req.num_tokens

    def update_from_output(self, out, result):
        assert req.next_decode_eligible_step > self.current_step
        req.num_tokens += len(result.sampled_token_ids[0])
        return {}

    monkeypatch.setattr(Scheduler, "_update_after_schedule", after_schedule)
    monkeypatch.setattr(Scheduler, "update_from_output", update_from_output)
    patch._patch_scheduler_update_after_schedule()
    patch._patch_scheduler_update_from_output()
    scheduler = SimpleNamespace(
        vllm_config=config(),
        use_pp=True,
        requests={"a": req},
        current_step=0,
        structured_output_manager=SimpleNamespace(should_advance=lambda r: False),
    )
    scheduled_counts = []

    def schedule(_):
        scheduler.current_step += 1
        count = req.num_tokens + len(req.spec_token_ids) - req.num_computed_tokens
        if scheduler.current_step < req.next_decode_eligible_step:
            count = 0
        scheduled_counts.append(count)
        out = SimpleNamespace(
            num_scheduled_tokens={"a": count} if count else {},
            total_num_scheduled_tokens=count,
            scheduled_spec_decode_tokens={},
            pending_structured_output_tokens=False,
        )
        Scheduler._update_after_schedule(scheduler, out)
        return out

    scheduler.schedule = schedule
    scheduler.has_requests = lambda: True
    scheduler.get_grammar_bitmask = lambda out: None
    scheduler.update_from_output = MethodType(Scheduler.update_from_output, scheduler)
    result = ModelRunnerOutput(
        req_ids=["a"], req_id_to_index={"a": 0}, sampled_token_ids=[[785]], spec_token_ids=[[-1]]
    )

    def ready(value):
        future = Future()
        future.set_result(value)
        return future

    executor = SimpleNamespace(
        execute_model=lambda *a, **kw: ready(None),
        sample_tokens=lambda *a, **kw: ready(result),
        take_draft_token_ids=Mock(side_effect=AssertionError("unpaired draft fetch")),
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        model_executor=executor,
        batch_queue=deque(),
        batch_queue_size=2,
        is_ec_consumer=True,
        is_pooling_model=False,
        async_scheduling=False,
        use_spec_decode=True,
        check_for_draft_tokens=True,
        _should_throttle_prefills=lambda: False,
        log_error_detail=lambda _: nullcontext(),
        capture_iteration_details=lambda _: nullcontext(),
        _process_aborts_queue=lambda: None,
        _attach_iteration_details=lambda *a: None,
    )
    assert EngineCore.step_with_batch_queue(engine) == (None, True)
    EngineCore.post_step(engine, True)
    assert req.spec_token_ids == []
    EngineCore.step_with_batch_queue(engine)
    assert scheduled_counts == [20, 0]
    assert req.num_tokens == 21
    assert req.spec_token_ids == [-1]
    assert req.next_decode_eligible_step == 0
    # The next decode now has both the sampled token and the draft slot.
    assert schedule(False).num_scheduled_tokens == {"a": 2}
    executor.take_draft_token_ids.assert_not_called()


@pytest.mark.parametrize(
    "async_scheduling,is_prefill_chunk,speculative,blocked",
    [
        (False, False, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (False, False, False, False),
    ],
)
def test_fence_preserves_chunked_prefill_and_other_modes(
    monkeypatch, async_scheduling, is_prefill_chunk, speculative, blocked
):
    monkeypatch.setattr(Scheduler, "_update_after_schedule", lambda *args: None)
    patch._patch_scheduler_update_after_schedule()
    request = SimpleNamespace(is_prefill_chunk=is_prefill_chunk, next_decode_eligible_step=0)
    scheduler = SimpleNamespace(vllm_config=config(async_scheduling, speculative), use_pp=True, requests={"a": request})
    Scheduler._update_after_schedule(scheduler, SimpleNamespace(num_scheduled_tokens={"a": 3}))
    assert (request.next_decode_eligible_step > 0) is blocked


def test_finished_request_is_not_given_new_drafts(monkeypatch):
    monkeypatch.setattr(Scheduler, "update_from_output", lambda *args: {})
    patch._patch_scheduler_update_from_output()
    request = SimpleNamespace(is_finished=lambda: True, spec_token_ids=[])
    scheduler = SimpleNamespace(vllm_config=config(), use_pp=True, requests={"a": request})
    out = SimpleNamespace(num_scheduled_tokens={"a": 2})
    result = SimpleNamespace(spec_token_ids=[[-1]], req_id_to_index={"a": 0}, sampled_token_ids=[[100]])
    Scheduler.update_from_output(scheduler, out, result)
    assert request.spec_token_ids == []


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "method,architecture,supported",
    [
        ("mtp", "Glm5NextForConditionalGeneration", True),
        ("dspark", "GlmMoeDsaForCausalLM", True),
        ("eagle3", "MiniMaxM3SparseForCausalLM", True),
        ("dspark", "Glm5NextForConditionalGeneration", False),
        ("eagle3", "Glm5NextForConditionalGeneration", False),
        ("unknown", "Glm5NextForConditionalGeneration", False),
    ],
)
def test_scheduler_and_worker_share_capability_routing(monkeypatch, legacy, method, architecture, supported):
    from vllm_ascend.worker.v2 import pp_utils

    monkeypatch.setattr(pp_utils, "use_legacy_spec_pp", lambda: legacy)
    cfg = config()
    cfg.speculative_config.method = method
    cfg.model_config.architecture = architecture
    expected = legacy and supported
    assert patch._use_v2_sync_pp_mtp_runtime_patch(cfg, True) is expected
    assert pp_utils.use_sync_spec_pp_output(cfg) is expected
    cfg.parallel_config.pipeline_parallel_size = 1
    assert not pp_utils.use_sync_spec_pp_output(cfg)


def test_native_path_keeps_upstream_draft_fetch(monkeypatch):
    from vllm_ascend.worker.v2 import pp_utils

    monkeypatch.setattr(pp_utils, "use_legacy_spec_pp", lambda: False)
    executor = SimpleNamespace(take_draft_token_ids=Mock(return_value=None))
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(vllm_config=config(), use_pp=True),
        model_executor=executor,
        batch_queue=deque(),
        async_scheduling=False,
        use_spec_decode=True,
        check_for_draft_tokens=True,
    )
    EngineCore.post_step(engine, True)
    executor.take_draft_token_ids.assert_called_once()
