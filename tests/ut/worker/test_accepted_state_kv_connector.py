# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""MTP save ordering against the native V2 runner/connector lifecycle.

Run in the CPU UT environment, whose conftest installs device mocks. No model,
worker constructor, graph, or transport thread is started. The native core
sample_tokens body supplies the order under test; do not replace it with a
handwritten imitation of the Ascend hook sequence.
"""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.worker.gpu import kv_connector as core_connector
from vllm.v1.worker.gpu import model_runner as core_runner
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.fixture(params=[False, True], ids=["async-load", "sync-load"])
def lifecycle(monkeypatch, request):
    """Use real core orchestration with state/transport/device work at the edges."""
    events = []
    backend = Mock()
    backend.get_transfer_results.return_value = SimpleNamespace(
        finished_sending=set(), finished_recving={"loaded"}, failed_recving=set()
    )
    backend.get_block_ids_with_load_errors.return_value = set()
    backend.get_kv_connector_stats.return_value = None
    backend.get_kv_connector_kv_cache_events.return_value = None
    backend.build_connector_worker_meta.return_value = None
    backend.bind_connector_metadata.side_effect = lambda metadata: events.append("bind")
    backend.start_load_kv.side_effect = lambda *args, **kwargs: events.append("load")
    backend.clear_connector_metadata.side_effect = lambda: events.append("clear")

    backend.wait_for_save.side_effect = lambda: events.append("save")
    active = core_connector.ActiveKVConnector.__new__(core_connector.ActiveKVConnector)
    active.kv_connector = backend
    active.vllm_config = object()
    active._pending_load_kwargs = None
    active._disabled = False
    monkeypatch.setattr(core_connector, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(core_connector, "get_forward_context", lambda: object())

    # Keep native sample_tokens, post_forward and finalize_step control flow.
    # Only device/output work and the model proposal are boundary doubles.
    monkeypatch.setattr(core_runner, "AsyncOutput", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(core_runner, "use_workspace_lane", lambda *args: nullcontext())
    monkeypatch.setattr(core_runner, "get_drafter_hidden_states", lambda model, hidden: hidden)

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.postprocess_sampled = Mock(side_effect=lambda *args: events.append("postprocess"))
    runner.is_last_pp_rank = True
    runner.pcp_manager = None
    runner.pp_handler = None
    runner.aux_output_connector = None
    runner.adaptive_verification = None
    runner._draft_workspace_lane = 0
    runner.__dict__["main_stream"] = object()
    runner.output_copy_stream = object()
    runner.check_ep_fault = False
    runner.kv_connector = active
    runner.model = SimpleNamespace(compute_logits=Mock())
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=Mock(return_value=None),
        compute_prompt_token_id_logprobs=Mock(return_value=None),
    )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(temperature=SimpleNamespace(gpu=None), seeds=SimpleNamespace(gpu=None))
    )
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
        last_sampled_tokens=None,
        next_prefill_tokens=None,
        draft_tokens=torch.zeros((1, 3), dtype=torch.int64, device="cpu"),
    )
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=Mock())
    runner.sample = Mock(return_value=(SimpleNamespace(sampled_token_ids=None), 1, 0))
    batch = SimpleNamespace(
        req_ids=["request"], idx_mapping=torch.tensor([0], device="cpu"), query_start_loc=None, num_tokens=1
    )
    schedule = SimpleNamespace(kv_connector_metadata=object(), has_sync_kv_loads=request.param, finished_req_ids=set())

    def draft(*args, **kwargs):
        # K=0 must still reach propose before save; this checks orchestration,
        # not the model's first-pass kernels or state bytes.
        backend.wait_for_save.assert_not_called()
        backend.clear_connector_metadata.assert_not_called()
        events.append("propose")
        return torch.zeros((1, 3), dtype=torch.int64, device="cpu")

    proposer = SimpleNamespace(supports_mm_inputs=False, propose=Mock(side_effect=draft))
    runner.speculator = proposer
    runner.num_speculative_steps = 3

    def run(num_drafts):
        runner.execute_model_state = SimpleNamespace(
            input_batch=batch,
            attn_metadata={},
            slot_mappings_by_layer={},
            hidden_states=object(),
            aux_hidden_states=None,
            dp_sync_state=None,
            finished_req_ids=set(),
            ec_connector_output=None,
            cudagraph_stats=None,
            num_spec_tokens_to_schedule=num_drafts,
        )
        runner.kv_connector.pre_forward(schedule, batch, attn_metadata={})
        runner.kv_connector.finish_forward()
        # Strip inference/EPLB decorators only; execute the installed native
        # implementation, so a changed core order cannot silently pass.
        return inspect.unwrap(GPUModelRunner.sample_tokens)(runner, None)

    return SimpleNamespace(
        runner=runner, backend=backend, events=events, proposer=proposer, run=run, has_sync_kv_loads=request.param
    )


@pytest.mark.parametrize("num_drafts", [0, 3])
def test_mtp_native_finalize_runs_once_after_propose(lifecycle, num_drafts):
    result = lifecycle.run(num_drafts)

    # Async loads may start in native post_forward before or after proposal.
    # Sync loads must precede compute; accepted-state saves must follow MTP.
    assert [event for event in lifecycle.events if event != "load"] == [
        "bind",
        "postprocess",
        "propose",
        "save",
        "clear",
    ]
    lifecycle.backend.start_load_kv.assert_called_once()
    assert lifecycle.events.index("bind") < lifecycle.events.index("load") < lifecycle.events.index("save")
    if lifecycle.has_sync_kv_loads:
        assert lifecycle.events.index("load") < lifecycle.events.index("postprocess")
    lifecycle.backend.wait_for_save.assert_called_once_with()
    lifecycle.backend.clear_connector_metadata.assert_called_once_with()
    lifecycle.proposer.propose.assert_called_once()
    assert lifecycle.proposer.propose.call_args.kwargs["num_speculative_tokens"] == num_drafts
    published = lifecycle.runner.draft_tokens_handler.set_draft_tokens.call_args.args[1]
    assert published.shape == (1, num_drafts)
    assert result.model_runner_output.kv_connector_output.finished_recving == {"loaded"}
