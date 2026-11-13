# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore

from vllm_ascend.patch.platform.patch_pp_mtp import (
    _PP_IN_FLIGHT_STEP,
    _patch_engine_core,
    _patch_scheduler_update_after_schedule,
    _patch_scheduler_update_from_output,
    _use_pp_ipc_runtime_patch,
)


def make_config(pp=2, tp=16, role="kv_producer", speculative=True, v2=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp, tensor_parallel_size=tp),
        kv_transfer_config=SimpleNamespace(kv_role=role),
        speculative_config=object() if speculative else None,
        use_v2_model_runner=v2,
    )


@pytest.mark.parametrize(
    "pp,tp,role,speculative,v2,use_pp,expected",
    [
        (2, 16, "kv_producer", True, False, True, True),
        (2, 8, "kv_producer", True, False, True, False),
        (1, 16, "kv_producer", True, False, False, False),
        (4, 16, "kv_producer", True, False, True, False),
        (2, 16, "kv_producer", False, False, True, False),
        (2, 16, "kv_producer", True, True, True, False),
        (2, 16, "kv_producer", True, False, False, False),
        (2, 16, "kv_consumer", True, False, True, True),
    ],
)
def test_prefill_fence_scope(pp, tp, role, speculative, v2, use_pp, expected):
    assert _use_pp_ipc_runtime_patch(make_config(pp, tp, role, speculative, v2), use_pp) is expected


def test_prefill_final_chunk_waits_for_output_but_intermediate_chunks_fill_pipeline(monkeypatch):
    monkeypatch.setattr(Scheduler, "_update_after_schedule", lambda self, out: None)
    _patch_scheduler_update_after_schedule()
    final = SimpleNamespace(is_prefill_chunk=False, next_decode_eligible_step=0)
    intermediate = SimpleNamespace(is_prefill_chunk=True, next_decode_eligible_step=0)
    scheduler = SimpleNamespace(
        vllm_config=make_config(), use_pp=True, requests={"final": final, "chunk": intermediate}
    )
    output = SimpleNamespace(num_scheduled_tokens={"final": 21, "chunk": 16363})
    Scheduler._update_after_schedule(scheduler, output)
    assert final.next_decode_eligible_step == _PP_IN_FLIGHT_STEP
    assert intermediate.next_decode_eligible_step == 0


def test_prefill_post_step_does_not_publish_drafts_before_confirmed_output(monkeypatch):
    calls = []
    monkeypatch.setattr(EngineCore, "post_step", lambda self, executed: calls.append(executed))
    _patch_engine_core()
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(vllm_config=make_config(), use_pp=True),
        batch_queue=object(),
        async_scheduling=False,
        use_spec_decode=True,
    )
    EngineCore.post_step(engine, True)
    assert calls == []


@pytest.mark.parametrize("sampled,expected_drafts", [([42], [43]), ([], [])])
def test_prefill_output_releases_fence_and_publishes_only_confirmed_drafts(monkeypatch, sampled, expected_drafts):
    monkeypatch.setattr(Scheduler, "update_from_output", lambda self, out, result: "outputs")
    _patch_scheduler_update_from_output()
    request = SimpleNamespace(
        next_decode_eligible_step=_PP_IN_FLIGHT_STEP, is_finished=lambda: False, spec_token_ids=[]
    )
    scheduler = SimpleNamespace(
        vllm_config=make_config(),
        use_pp=True,
        requests={"req": request},
        structured_output_manager=SimpleNamespace(validate_tokens=lambda req, tokens: tokens),
    )
    output = SimpleNamespace(num_scheduled_tokens={"req": 21})
    result = SimpleNamespace(req_id_to_index={"req": 0}, sampled_token_ids=[sampled], spec_token_ids=[[43]])
    assert Scheduler.update_from_output(scheduler, output, result) == "outputs"
    assert request.next_decode_eligible_step == 0
    assert request.spec_token_ids == expected_drafts
