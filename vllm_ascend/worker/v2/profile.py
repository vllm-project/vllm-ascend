# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Startup memory profiling with separate target and draft token extents."""

import gc
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import model_runner as upstream
from vllm.v1.worker.gpu.input_batch import InputBatch

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _profile_target(runner: "NPUModelRunner", num_tokens: int, randomize_inputs: bool):
    """Run only the target; execute_model leaves drafting to its caller."""
    num_reqs = min(num_tokens, runner.max_num_reqs)
    counts = np.full(num_reqs, num_tokens // num_reqs, dtype=np.int32)
    counts[num_reqs - num_tokens % num_reqs :] += 1
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.num_scheduled_tokens = {f"_dummy_req_{i}": int(n) for i, n in enumerate(counts)}
    scheduler_output.total_num_scheduled_tokens = num_tokens
    intermediate = None
    if not runner.is_first_pp_rank:
        assert runner.intermediate_tensors is not None
        intermediate = runner.intermediate_tensors[:num_tokens]
    max_loras = runner.lora_config.max_loras if runner.lora_config is not None else 0
    runner.kv_connector.set_disabled(True)
    try:
        with runner.maybe_dummy_run_with_lora(
            runner.lora_config,
            num_scheduled_tokens=counts,
            num_sampled_tokens=None,
            remove_lora=True,
            num_active_loras=max_loras,
        ):
            runner.execute_model(
                scheduler_output,
                intermediate_tensors=intermediate,
                dummy_run=True,
                skip_attn_for_dummy_run=True,
                is_profile=True,
                randomize_inputs=randomize_inputs,
            )
    finally:
        runner.kv_connector.set_disabled(False)
    state = runner.execute_model_state
    runner.execute_model_state = None
    return state


def _global_profile_hidden(hidden: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Account for restored global features without requiring initialized KV.

    Values are synthetic, as for the startup dummy itself. Keep the local
    output alive while allocating the global output, matching restore overlap.
    """
    return hidden.new_zeros((num_tokens, *hidden.shape[1:]))


def _profile_draft(runner: "NPUModelRunner", num_tokens: int, local_tokens: int, hidden, aux_hidden):
    speculator = runner.speculator
    assert speculator is not None
    num_reqs = min(runner.max_num_reqs, num_tokens // speculator.num_query_per_req)
    sharded = speculator.vllm_config.parallel_config.prefill_context_parallel_size > 1
    # MTP/Eagle prefill is local, while their subsequent decode keeps all
    # requests. DFlash/DSpark consume global context and choose B*Q themselves.
    context_tokens = max(local_tokens, num_reqs) if sharded else num_tokens
    if sharded and aux_hidden:
        # Eagle combines global auxiliary features before selecting PCP rows.
        hidden = speculator.model.combine_hidden_states(torch.cat(aux_hidden, dim=-1))
        aux_hidden = None
    batch = InputBatch.make_dummy(num_reqs, context_tokens, runner.input_buffers, is_padding=False)
    mm_inputs = (
        ([], torch.zeros(context_tokens, dtype=torch.bool, device="cpu")) if speculator.supports_mm_inputs else None
    )
    if isinstance(runner.sampler, upstream.GPUWatermarkSampler):
        speculator.prepare_watermarking(runner.sampler, batch.idx_mapping)
    with upstream.use_workspace_lane(runner._draft_workspace_lane):
        speculator.propose(
            input_batch=batch,
            attn_metadata=None,
            slot_mappings=None,
            last_hidden_states=hidden[:context_tokens],
            aux_hidden_states=[h[:context_tokens] for h in aux_hidden] if aux_hidden else None,
            num_sampled=torch.ones(num_reqs, dtype=torch.int32, device=runner.device),
            num_rejected=torch.zeros(num_reqs, dtype=torch.int32, device=runner.device),
            last_sampled=runner.req_states.last_sampled_tokens,
            next_prefill_tokens=runner.req_states.next_prefill_tokens,
            temperature=runner.sampler.sampling_states.temperature.gpu,
            seeds=runner.sampler.sampling_states.seeds.gpu,
            dp_sync_state=None,
            dummy_run=True,
            skip_attn_for_dummy_run=True,
            mm_inputs=mm_inputs,
            is_profile=True,
        )


def profile_pcp(runner: "NPUModelRunner", randomize_inputs: bool = False) -> None:
    """Keep the global budget, but profile target and draft stages separately."""
    # Same encoder/aux-output setup as GPUModelRunner.profile_run.
    if runner.vllm_config.aux_output_config.enabled:
        runner.aux_output_connector = upstream.get_aux_output_connector(runner.model, runner.vllm_config)
    if runner.supports_mm_inputs and runner.is_first_pp_rank:
        mm_config = runner.model_config.multimodal_config
        if mm_config is not None and not mm_config.skip_mm_profiling:
            budget = upstream.MultiModalBudget(runner.vllm_config, runner.mm_registry, enable_cache=False)
            runner.model_state.encoder_runner.profile_encoder_cache(budget.get_dummy_encoder_profile_inputs(), budget)

    global_tokens = runner.max_num_tokens
    pcp_size = runner.vllm_config.parallel_config.prefill_context_parallel_size
    local_tokens = (global_tokens + pcp_size - 1) // pcp_size
    state = _profile_target(runner, local_tokens, randomize_inputs)
    if runner.is_last_pp_rank:
        assert state is not None
        assert state.hidden_states is not None
        hidden = _global_profile_hidden(state.hidden_states, global_tokens)
        aux_hidden = (
            [_global_profile_hidden(h, global_tokens) for h in state.aux_hidden_states]
            if state.aux_hidden_states
            else None
        )
        if runner.speculator is not None:
            draft_local = upstream.get_drafter_hidden_states(runner.model, state.hidden_states)
            draft_hidden = (
                hidden if draft_local is state.hidden_states else _global_profile_hidden(draft_local, global_tokens)
            )
            _profile_draft(runner, global_tokens, local_tokens, draft_hidden, aux_hidden)
        num_reqs = min(global_tokens, runner.max_num_reqs)
        if runner.pooling_runner is None:
            runner._dummy_sampler_run(hidden[:num_reqs])
        else:
            runner._dummy_pooler_run(hidden)
        # Keep restored global features and local outputs live through drafting.
        torch.accelerator.synchronize()
        del hidden, aux_hidden
    else:
        torch.accelerator.synchronize()
    del state
    runner.reset_encoder_cache()
    runner._finalize_dump_data(dump=False)
    gc.collect()
