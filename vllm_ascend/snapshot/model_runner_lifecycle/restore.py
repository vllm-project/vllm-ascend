# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
import torch.nn as nn

from vllm_ascend.ops.fused_moe.moe_utils import reset_moe_padding_cache_after_snapshot_restore
from vllm_ascend.ops.rotary_embedding import reload_cos_and_sin_after_restore
from vllm_ascend.snapshot.model_runner_lifecycle.checkpoint import model_checkpoint_path
from vllm_ascend.snapshot.model_runner_lifecycle.module_lifecycle import (
    get_drafter_model,
    reset_modules_runtime_state,
    restore_model_non_checkpoint_state,
    restore_state_dict,
)

logger = logging.getLogger(__name__)


def _restore_model_checkpoint(runner, model: nn.Module, model_save_path: str, label: str) -> None:
    restore_state_dict(model, model_save_path, label)
    restore_model_non_checkpoint_state(model, runner.model_config.dtype, label)


def restore_model_runner(runner, path: str = "/mnt") -> None:
    model = runner.get_model()
    _restore_model_checkpoint(
        runner,
        model,
        model_checkpoint_path(runner, path),
        "model",
    )
    drafter_model = get_drafter_model(runner)
    if drafter_model is not None:
        _restore_model_checkpoint(
            runner,
            drafter_model,
            model_checkpoint_path(runner, path, drafter=True),
            "drafter",
        )

    _restore_model_runner_runtime_state(runner)


def _restore_model_runner_runtime_state(runner) -> None:
    """Prepare all non-checkpoint model-runner state for post-restore inference.

    This includes global derived tensors, speculative-decoding state, attention
    metadata, runner input buffers, model-module runtime state, and block tables.
    """
    reload_cos_and_sin_after_restore(runner.get_model())
    reset_moe_padding_cache_after_snapshot_restore()
    _reset_v2_runner_runtime_state(runner)
    _reset_target_and_drafter_modules_after_restore(runner)
    # Zeroers retain device-resident address/stride tables, not just KV views.
    # Rebuild only when cold start constructed one, before any kernel consumes it.
    if runner.kv_block_zeroer is not None:
        runner._init_kv_zero_meta()
    if runner.vllm_config.parallel_config.pipeline_parallel_size > 1:
        from vllm_ascend.snapshot.pipeline import restore_pipeline_runtime

        restore_pipeline_runtime(runner)


def _reset_input_buffers(buffers) -> None:
    for name in (
        "input_ids",
        "positions",
        "is_padding",
        "query_start_loc",
        "seq_lens",
        "dcp_local_seq_lens",
    ):
        getattr(buffers, name).zero_()


def _reset_staged_tensor(buffer) -> None:
    buffer.gpu.zero_()
    buffer.clear_staged_writes()


def _reset_request_state(state) -> None:
    # Snapshot creation is an idle-service operation, not request migration.
    # Preserve the RequestState object: sampler/PCP helpers also reference it.
    if state.num_reqs:
        raise RuntimeError("Snapshot restore requires an empty Model Runner V2 request state")
    for name in ("all_token_ids", "total_len", "num_computed_tokens"):
        _reset_staged_tensor(getattr(state, name))
    for name in ("prompt_len", "prefill_len"):
        buffer = getattr(state, name)
        buffer.cpu.zero_()
        buffer.copy_to_uva()
    state.num_computed_prefill_tokens.fill(0)
    state.num_computed_tokens_np.fill(0)
    state.max_seq_len.fill(0)
    state.last_sampled_tokens.zero_()
    state.draft_tokens.zero_()
    state.next_prefill_tokens.zero_()


def _reset_block_tables(tables) -> None:
    for table in tables.block_tables:
        _reset_staged_tensor(table)
    tables.num_blocks.cpu.zero_()
    tables.num_blocks.copy_to_uva()
    for table in tables.input_block_tables:
        table.zero_()
    tables.slot_mappings.fill_(-1)
    # These tensors contain device addresses and layout constants, not request
    # data. Rebuild them instead of zeroing them like the block IDs above.
    tables.init_block_table_layout_tensors()


def _reset_speculator(speculator, target_model) -> None:
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
    from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import ExtractHiddenStatesSpeculator
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import TargetDependentARSpeculator

    _reset_input_buffers(speculator.input_buffers)
    for name in ("idx_mapping", "temperature", "seeds", "draft_tokens"):
        getattr(speculator, name).zero_()
    speculator.draft_is_prefilling.zero_()
    speculator.draft_token_confidence_probs.zero_()
    if speculator.acceptance_estimator is not None:
        from vllm.v1.worker.gpu.spec_decode.acceptance_estimator import OnlineAcceptanceEstimator

        speculator.acceptance_estimator = OnlineAcceptanceEstimator(
            speculator.max_num_reqs, speculator.num_speculative_steps, speculator.device
        )
    if speculator.draft_logits is not None:
        # Main owns logits plus TP row/owner maps in DraftLogitsCache. Reuse
        # the allocation contract so sharded sampling starts with fresh maps.
        speculator._init_draft_logits(target_model)
    speculator.input_batch = None
    if isinstance(speculator, AutoRegressiveSpeculator):
        speculator.input_buffers.seq_lens_cpu.zero_()
        speculator.current_draft_step.zero_()
        speculator.last_token_indices.zero_()
    if isinstance(speculator, (TargetDependentARSpeculator, DFlashSpeculator, ExtractHiddenStatesSpeculator)):
        speculator.hidden_states.zero_()
    if isinstance(speculator, TargetDependentARSpeculator):
        speculator.sample_src_positions.zero_()
        if speculator.inputs_embeds is not None:
            speculator.inputs_embeds.zero_()
    if isinstance(speculator, DFlashSpeculator):
        if speculator._aux_staging is not None:
            speculator._aux_staging.zero_()
        speculator._num_rejected = None
        speculator.context_positions.zero_()
        speculator.sample_indices.zero_()
        speculator.sample_pos.zero_()
        speculator.sample_idx_mapping.fill_(-1)
        speculator.sample_col.copy_(
            torch.arange(
                speculator.num_speculative_steps,
                dtype=torch.int32,
                device=speculator.device,
            ).repeat(speculator.max_num_reqs)
        )
        speculator._context_slot_mappings.fill_(-1)
    if isinstance(speculator, DFlash2Speculator):
        speculator.candidate_sampler.scores.zero_()
        speculator.candidate_sampler.cached_candidate_ids.zero_()
    if isinstance(speculator, DSparkSpeculator):
        speculator._step_cols.copy_(
            torch.arange(
                speculator.num_speculative_steps,
                dtype=torch.int32,
                device=speculator.device,
            )
        )
        speculator._anchor_idx.copy_(
            torch.arange(
                speculator.max_num_reqs,
                dtype=torch.int64,
                device=speculator.device,
            )
            * speculator.num_query_per_req
        )
        if speculator._d2t_scatter_index is not None:
            d2t = speculator.model.draft_id_to_target_id
            speculator._d2t_scatter_index.copy_(torch.arange(d2t.shape[0], device=d2t.device) + d2t)
            speculator._draft_scatter_buf.fill_(float("-inf"))


@torch.inference_mode()
def _reset_metadata_builders(builders) -> None:
    # Builders can retain inference tensors allocated during warmup/capture.
    builders_and_masks = builders + [
        builder.attn_mask_builder for builder in builders if hasattr(builder, "attn_mask_builder")
    ]
    seen_ids: set[int] = set()
    for builder_or_mask in builders_and_masks:
        if id(builder_or_mask) in seen_ids:
            continue
        seen_ids.add(id(builder_or_mask))
        reset_state = getattr(builder_or_mask, "reset_runtime_state_after_snapshot_restore", None)
        if callable(reset_state):
            reset_state()
        restore_constants = getattr(builder_or_mask, "restore_constants_after_snapshot_restore", None)
        if callable(restore_constants):
            restore_constants()


@torch.inference_mode()
def _reset_v2_runner_runtime_state(runner) -> None:
    from vllm.v1.worker.gpu import pcp_manager
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    from vllm_ascend.worker.v2.kvpp import KVPPRuntime

    runner.execute_model_state = None
    _reset_request_state(runner.req_states)
    _reset_input_buffers(runner.input_buffers)
    runner.input_buffers.seq_lens_cpu.zero_()
    runner.num_computed_tokens_cpu.zero_()
    _reset_block_tables(runner.block_tables)
    runner.reset_encoder_cache()
    _reset_v2_model_state(runner.model_state)
    runner.draft_tokens_handler.req_ids = []
    runner.draft_tokens_handler.draft_tokens_np = None
    runner.draft_tokens_handler.num_draft_tokens = 0

    # PCP stores local block-table pointer arrays and per-batch restore indices.
    # Reuse its cold-start constructor, then reconnect the shared consumers.
    runner.pcp_manager = pcp_manager.maybe_build_pcp_manager(
        runner.vllm_config,
        runner.device,
        runner.supports_mm_inputs,
        runner.block_tables,
        cls=runner.pcp_manager_cls,
    )
    if runner.pcp_manager is not None:
        runner.pcp_manager.set_runner_state(runner.vllm_config, runner.kv_cache_config, runner.input_buffers)
        runner.model_state.pcp_manager = runner.pcp_manager
        if runner.speculator is not None:
            runner.pcp_manager.set_speculator(runner.speculator)
            runner.speculator.pcp_manager = runner.pcp_manager

    groups = list(runner.attn_groups)
    if isinstance(runner.speculator, DraftModelSpeculator):
        _reset_speculator(runner.speculator, runner.get_model())
        if runner.speculator.model_state is not runner.model_state:
            _reset_v2_model_state(runner.speculator.model_state)
        groups.extend(runner.speculator.attn_groups)
    _reset_metadata_builders(
        [builder for kv_groups in groups for group in kv_groups for builder in group.metadata_builders]
    )

    if runner.kvpp.scheduler is not None:
        runner.kvpp.scheduler._prefetch_executor.shutdown(wait=True)
        runner.kvpp = KVPPRuntime.create_from_kv_cache(
            vllm_config=runner.vllm_config,
            kv_cache_config=runner.kv_cache_config,
            static_forward_context=runner.compilation_config.static_forward_context,
        )
        runner.model_state.kvpp_runtime = runner.kvpp


@torch.inference_mode()
def _reset_v2_model_state(model_state) -> None:
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

    if isinstance(model_state, DefaultModelState):
        # Capture reads these buffers without running per-request preparation.
        # They are separate from runner.input_buffers and model RoPE caches.
        if model_state.rope_state is not None:
            rope = model_state.rope_state
            rope.positions.zero_()
            rope.prefill_positions.clear_staged_writes()
            rope.prefill_delta.cpu.zero_()
            rope.prefill_delta.copy_to_uva()
        if model_state.supports_mm_inputs:
            model_state.encoder_runner.inputs_embeds.zero_()
        elif model_state.prompt_embeds_state is not None:
            model_state.inputs_embeds.zero_()
        if model_state.prompt_embeds_state is not None:
            prompt = model_state.prompt_embeds_state
            prompt.gpu_tensors.clear()
            for buffer in (prompt.embeds_ptrs, prompt.mask_ptrs, prompt.embeds_lens):
                buffer.cpu.zero_()
                buffer.copy_to_uva()
    if isinstance(model_state, AscendMambaHybridModelState):
        model_state.reset_runtime_state_after_snapshot_restore()


def _reset_target_and_drafter_modules_after_restore(runner) -> None:
    """Reset reusable runtime state owned by target and drafter modules.

    The module lifecycle dispatcher visits modules and their implementations;
    modules must not forward the same hook to their impl again.
    """
    reset_modules_runtime_state((runner.get_model(), get_drafter_model(runner)))


def reset_graph_managers(runner) -> None:
    """Drop Model Runner V2 graph handles after the ACL Graph pool reset."""
    from vllm_ascend.compilation import acl_graph
    from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

    # Manager constructors register new objects, so drop all three old
    # registrations once, before constructing either target or draft managers.
    acl_graph._graph_params = None
    acl_graph._draft_graph_params = None
    acl_graph._draft_graph_prefill_params = None

    manager = runner.cudagraph_manager
    runner.cudagraph_manager = ModelAclGraphManager(
        runner.vllm_config,
        runner.device,
        manager.cudagraph_mode,
        runner.vllm_config.get_decode_cudagraph_query_lens(runner.decode_query_len),
        runner,
        lora_capture_cases=manager.lora_capture_cases,
        varlen_decode=manager.varlen_decode,
        ubatch_runner=manager.ubatch_runner,
    )
    if runner.fast_prefill is not None:
        # This helper dispatches through the manager, not through the runner.
        runner.fast_prefill.cudagraph_manager = runner.cudagraph_manager
    if runner.speculator is not None:
        runner.speculator.init_cudagraph_manager(manager.cudagraph_mode)
