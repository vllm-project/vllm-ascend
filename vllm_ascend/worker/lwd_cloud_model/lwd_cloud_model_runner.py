# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD model runner subclass (cloud side).

All LWD runner-side logic lives here (moved out of
``model_runner_v1.py``): the per-step cloud-side collection of the
DOWN payload (hidden-only tensor) plus the step metadata (global ranks
/ num_accepted / req_ids) that rides back to the scheduler on
``ModelRunnerOutput.lwd_c2e_meta``.

Selected via ``LwdCloudWorker`` (worker_cls), which swaps the model
runner class at init_device when ``lwd_config`` is set.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.lwd_logger import lwd_logger
from vllm.v1.lwd_control.lwd_comm.lwd_id_adapter import parse_edge_id
from vllm.v1.outputs import ModelRunnerOutput

from vllm_ascend.distributed.lwd_data_plane.lwd_types import ConnectionId
from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import LwdBaseModelRunner
from vllm_ascend.worker.lwd_cloud_model.lwd_mtp_proposer import LwdMTPProposer
from vllm_ascend.worker.lwd_cloud_model.lwd_token_rank import LwdTokenRankEncoder
from vllm_ascend.worker.lwd_hash.lwd_hash_routing import (
    LwdHashRoutingState,
    hash_layer_count,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from typing import Any

    from vllm.config import VllmConfig
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import (
        GrammarOutput,
        NewRequestData,
        SchedulerOutput,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ECConnectorOutput, SamplerOutput
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_worker import LwdCloudWorker
    from vllm_ascend.worker.lwd_cloud_model.lwd_host_stage import LwdHostStager


_LWD_H2D_VERIFY_ELEMENTS = 16
_LWD_WINDOW_MIN_DEPTH = 2


class LwdPromptWindow(NamedTuple):
    """One request's prompt rows in the current native scheduling window."""

    req_id: str
    req_index: int
    start: int
    rows: int


class LwdUpWindow(NamedTuple):
    """One connection's broadcast window in a native mixed batch."""

    connection: ConnectionId
    dev: torch.Tensor
    win_idx: int | None
    plan: list[LwdPromptWindow]


class LwdDownPayload(NamedTuple):
    connection: ConnectionId
    hidden: torch.Tensor
    meta_dev: torch.Tensor
    seg_lens_host: list[int] | None
    ready_event: torch.npu.Event
    req_ids: list[str]


class LwdDownSelection(NamedTuple):
    """Row/rank selection product of one DOWN collect step."""

    hidden_packet: torch.Tensor
    ranks: list[torch.Tensor]
    counts_dev: torch.Tensor | None
    seg_lens_host: list[int] | None


class LwdCloudModelRunner(LwdBaseModelRunner):
    """NPUModelRunner + LWD cloud-side runner logic."""

    # Config-derived constants, computed once at first use; lwd_config and
    # the HF config are immutable at runtime. Class-level defaults keep
    # instances built via __new__ (tests) from tripping on missing attrs.
    _lwd_hash_dims_cached: tuple[int, int] | None = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdCloudWorker | None = None,
    ) -> None:
        super().__init__(vllm_config, device, worker)
        self._lwd_pending_down_payload = None
        self._lwd_request_connections: dict[str, ConnectionId] = {}
        self._lwd_token_rank_encoder = LwdTokenRankEncoder()
        # Row-slot residue cleanup: flush reused rows as soon as a request
        # is admitted (2026-10-21 MTE root-cause closure)
        self._lwd_reset_reused_rows_on_admission()
        # Stage per-request prompt lengths for device-side mask refresh.
        if self.use_async_spec_decode and self.enable_prompt_embeds:
            self._lwd_prompt_lens_gpu = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=self.device)
        else:
            self._lwd_prompt_lens_gpu = None
        self.lwd_hash_state = None
        # None marks "no real batch staged yet"; distinguishes dummy forwards.
        self._lwd_hash_step_tokens = None
        config = self.model_config.hf_config
        num_hash_layers = hash_layer_count(config)
        if num_hash_layers and vllm_config.lwd_config is not None:
            # UP currently broadcasts one edge's batch to one cloud TP group.
            # CP/DP need additional request-to-rank layouts before supporting V4.
            parallel = vllm_config.parallel_config
            if (
                parallel.data_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
            ):
                raise ValueError("LWD V4 Hash routing currently requires DP=PCP=DCP=1 (TP/EP are supported)")
            self.lwd_hash_state = LwdHashRoutingState(
                num_hash_layers,
                config.num_experts_per_tok,
                config.n_routed_experts,
            )
            # Persistent buffers are populated before model execution, so graph
            # capture and replay see the same addresses for prompt and decode.
            self._lwd_hash_experts = torch.zeros(
                self.max_num_tokens,
                num_hash_layers,
                config.num_experts_per_tok,
                dtype=torch.int32,
                device=device,
            )
            self._lwd_hash_prompt_mask = torch.zeros(self.max_num_tokens, dtype=torch.bool, device=device)
            self._lwd_hash_token_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
            # Retain the zero placeholder so graph replay uses the captured storage.
            self._lwd_hash_placeholder_ids = torch.zeros_like(self.input_ids.gpu)

    def _lwd_reset_reused_row(
        self,
        request: CachedRequestState,
        req_index: int,
    ) -> None:
        """Clear tokens left in a reused batch row by the previous request.

        LWD cloud requests carry no token ids (the wire ships embeddings
        only), so the native add_request skips writing the row's prompt
        region for them. Stale cells from the row's previous occupant -
        including -1 placeholders that async scheduling leaves behind and
        nothing repairs for penalty-free requests - survive row reuse and
        ride the prompt-token tensor into the penalty bincount kernel,
        where negative ids become negative-offset writes (the aivec
        fault). Native requests carry real ids and overwrite the region
        on every admission, which is exactly the behavior missing here.

        Fill [0, num_prompt) with vocab_size - the "no token" padding
        value every downstream consumer already ignores - once, at
        admission, before any reader can observe the row.
        """
        if request.prompt_token_ids is not None:
            return  # rows with real ids are already overwritten natively
        batch = self.input_batch
        num_prompt = int(batch.num_prompt_tokens[req_index])
        batch.token_ids_cpu[req_index, :num_prompt] = batch.vocab_size

    def _lwd_reset_reused_rows_on_admission(self) -> None:
        """Hook the row reset onto request admission (instance-level).

        The admission call site lives inside the shared update_states
        with no LWD-owned seam; overriding it would mean copying shared
        logic that breaks on every upstream change. Wrap add_request on
        this process's InputBatch instance instead: native bookkeeping
        first, then _lwd_reset_reused_row. Migrate to an upstream hook
        if one is ever provided.

        Idempotent per InputBatch instance: initialize_kv_cache
        REBUILDS the InputBatch when the final KV geometry differs from
        the pre-load guess (always true for the hybrid groups here,
        model_runner_v1 re-init), so the hook must be re-anchored on
        the live instance - see initialize_kv_cache below. The first
        acceptance run failed exactly this way: the wrapper stayed on
        the discarded instance and the residue poison survived.
        """
        batch = self.input_batch
        if getattr(batch, "_lwd_row_reset_hooked", False):
            return
        batch._lwd_row_reset_hooked = True
        orig_add = batch.add_request

        def add_request_then_reset_row(request: CachedRequestState) -> int:
            req_index = orig_add(request)
            self._lwd_reset_reused_row(request, req_index)
            return req_index

        batch.add_request = add_request_then_reset_row

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        """Re-anchor the row-reset hook after the InputBatch rebuild.

        The base may replace self.input_batch here (hybrid KV geometry);
        the per-instance hook flag on the fresh instance is absent, so
        the installer below (re)hooks the live one."""
        super().initialize_kv_cache(kv_cache_config, kv_cache_allocation_context)
        self._lwd_reset_reused_rows_on_admission()

    def _get_drafter(self) -> SpecDecodeBaseProposer | None:
        """Use the LWD MTP embedding adapter only in LWD mode."""
        if (
            self.vllm_config.lwd_config is not None
            and self.speculative_config.method == "mtp"
            and not self.speculative_config.use_step3p5_mtp()
        ):
            return LwdMTPProposer(self.vllm_config, self.device, self)
        return super()._get_drafter()

    # ------------------------------------------------------------------ #
    # Remote embeds injection (cloud input has NO token ids — the prompt   #
    # embeddings arrive over the UP channel and must drive forward)        #
    # ------------------------------------------------------------------ #

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, int]:
        """Prime full-prompt M-RoPE and harvest arrivals before native preparation."""
        for new_req in scheduler_output.scheduled_new_reqs:
            payload = new_req.lwd_mrope_positions or b""
            if not payload:
                continue
            prompt_len = new_req.prompt_len
            if prompt_len <= 0:
                raise RuntimeError(f"[Lwd][cloud-runner] mrope prime req={new_req.req_id}: prompt length is missing")
            self._lwd_prime_mrope_positions(new_req.req_id, payload, prompt_len)
        windows = self._lwd_harvest_up_batches(num_scheduled_tokens, scheduler_output)
        # Native placeholder assembly contains only zeros. Skip its redundant
        # host copy and H2D; the arrival window below is authoritative.
        embeds_map = self.input_batch.req_prompt_embeds
        self.input_batch.req_prompt_embeds = {}
        try:
            out = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        finally:
            self.input_batch.req_prompt_embeds = embeds_map
        self._lwd_rebuild_is_token_ids_mask(num_scheduled_tokens)
        self._lwd_release_consumed_prompt_embeds()
        for window in windows:
            self._lwd_inject_remote_embeds(
                window.dev,
                window.win_idx,
                window.plan,
                num_scheduled_tokens,
            )
        if self.lwd_hash_state is not None:
            self._lwd_prepare_hash_batch(num_scheduled_tokens)
        return out

    def _lwd_prepare_hash_batch(self, num_scheduled_tokens: np.ndarray) -> None:
        """Skip remote expert staging for decode, preserving graph buffers."""
        num_reqs = self.input_batch.num_reqs
        counts = num_scheduled_tokens[:num_reqs]
        computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
        prompt_lengths = self.input_batch.num_prompt_tokens[:num_reqs]
        self._lwd_hash_step_tokens = sum(int(n) for n in counts)
        has_prompt = any(
            int(count) > 0 and int(start) < int(prompt_len)
            for count, start, prompt_len in zip(counts, computed, prompt_lengths)
        )
        # Clear the full mask, including graph padding and previous larger
        # batches. Keep its address and the captured merge graph unchanged.
        self._lwd_hash_prompt_mask.zero_()
        if not has_prompt:
            # The merge selects local lookup for every row. Stale expert IDs
            # are only unselected values, never indices into an expert table.
            return
        batch, mask = self.lwd_hash_state.build_batch(
            self.input_batch.req_ids,
            counts,
            computed,
            prompt_lengths,
        )
        self._lwd_hash_experts.zero_()
        self._lwd_hash_experts[: batch.shape[0]].copy_(batch)
        self._lwd_hash_prompt_mask[: mask.shape[0]].copy_(mask)

    def _lwd_rebuild_is_token_ids_mask(self, num_scheduled_tokens: np.ndarray) -> None:
        """Rebuild the flattened is_token_ids mask for the prompt-embeds branch.

        Root cause (async spec decode optimism): the runner's
        num_computed_tokens_cpu assumes every previous draft was accepted
        (r tokens past the real commit position), while the native
        flattening (token_indices = positions + idx*M in
        model_runner_v1.py) gathers from the 2D mask with those
        optimistic positions — decode rows shift right by r and the last
        r rows of each segment fall past the True-marked region, reading
        False/stale values. The prompt-embeds branch then treats them as
        embeds rows, skipping local embedding and leaving stale garbage
        in inputs_embeds.gpu; those rows are exactly the sampled rows
        (logits_indices takes the last 1+k rows of each segment), so
        garbage logits pollute committed tokens directly. Native/phase/
        single-request shapes are unaffected (the mask is only consumed
        on the prompt-embeds branch; those shapes take the MM branch).

        Under LWD semantics only injected rows (prompt segments) may be
        embeds rows (False); decode/draft rows are always token ids
        (True). This rebuild is deterministic from scheduler state and
        needs no optimistic-value correction: a decode request's
        optimistic computed only over-estimates, so prefill_rows is
        always 0 — inherently safe. The CPU-view repair is kept for the
        parent path (.np and .cpu share storage); the async path
        regenerates the GPU mask in _preprocess from corrected device
        positions and consumes the device embedding directly.
        """
        if not self.input_batch.req_prompt_embeds:
            # No embeds buffers in the batch (MM/text branch; nobody reads the mask)
            return
        mask = self.is_token_ids.np
        computed = self.input_batch.num_computed_tokens_cpu
        num_prompt = self.input_batch.num_prompt_tokens
        n_reqs = len(num_scheduled_tokens)
        total = int(np.sum(num_scheduled_tokens[:n_reqs]))
        mask[:total] = True
        off = 0
        for i in range(n_reqs):
            n = int(num_scheduled_tokens[i])
            prefill_rows = min(n, max(int(num_prompt[i]) - int(computed[i]), 0))
            if prefill_rows > 0:
                mask[off : off + prefill_rows] = False
            off += n

    def _preprocess(
        self,
        scheduler_output: SchedulerOutput,
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        """Refresh the decode mask after the async speculative-position fix, then prepare model inputs."""
        if (
            self.use_async_spec_decode
            and self.enable_prompt_embeds
            and intermediate_tensors is None
            and (self.vllm_config.parallel_config.prefill_context_parallel_size == 1)
        ):
            self._lwd_refresh_decode_embedding_mask(scheduler_output)
            if (
                get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder
                and (not self.supports_mm_inputs or self.input_batch.req_prompt_embeds)
            ):
                return self._lwd_preprocess_prompt_embeds(scheduler_output, num_input_tokens)
            # Other model paths keep the parent's preprocessing, which may
            # still consume the CPU mask (for example encoder-decoder models).
            count = scheduler_output.total_num_scheduled_tokens
            self.is_token_ids.np[:count] = self.is_token_ids.gpu[:count].cpu().numpy()
        return super()._preprocess(scheduler_output, num_input_tokens, intermediate_tensors)

    # ------------------------------------------------------------------ #
    # Zero-base-weave seams: all LWD behavior surfaces via subclass    #
    # overrides; the base classes carry no seams.                       #
    # ------------------------------------------------------------------ #

    def _lwd_rewind_preempted_requests(self, scheduler_output: SchedulerOutput) -> None:
        """Align host read pointers with scheduler rollbacks and prefix hits."""
        worker = self.worker
        if worker is None or not worker._lwd_is_early_recv_endpoint:
            return
        stager = worker._lwd_stager
        if stager is None:
            return
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_computed in zip(cached.req_ids, cached.num_computed_tokens):
            # Each connection is staged only on its real endpoint rank. Other
            # TP ranks must not rewind a nonexistent local host descriptor.
            if stager.take_req(req_id) is None:
                continue
            consumed = stager.consumed_rows(req_id)
            if consumed > new_computed:
                stager.rewind_consumed(req_id, new_computed)
                lwd_logger.info(
                    "[LWD][cloud-runner] REWIND req=%s: host read pointer "
                    "%d -> %d (preemption recompute; payload retained)",
                    req_id,
                    consumed,
                    new_computed,
                )
            elif new_computed > consumed:
                req = stager.take_req(req_id)
                if req is not None and consumed < req.prompt_rows:
                    stager.skip_consumed(req_id, new_computed)
                    lwd_logger.info(
                        "[LWD][cloud-runner] SKIP req=%s: host read pointer %d -> %d (re-admission hit)",
                        req_id,
                        consumed,
                        new_computed,
                    )

    def _update_states(self, scheduler_output: SchedulerOutput) -> Callable[[], None] | None:
        """Align host reads and materialize meta placeholders.

        The engine attaches prompt embeds as meta tensors (shape only,
        empty storage), so the placeholder rides the MQ inline pickle
        path instead of the overflow slow channel; the base
        CachedRequestState construction passes new_req_data.prompt_embeds
        straight through — swap meta tensors for local all-zeros
        allocations here, native path behavior unchanged.
        NOTE: a meta tensor still reports full nbytes (numel * element_size);
        do not infer transport cost from nbytes."""
        self._lwd_rewind_preempted_requests(scheduler_output)
        for new_req in scheduler_output.scheduled_new_reqs:
            embeds = new_req.prompt_embeds
            if embeds is not None and embeds.is_meta:
                new_req.prompt_embeds = torch.zeros(embeds.shape, dtype=embeds.dtype)
        return super()._update_states(scheduler_output)

    def _update_streaming_request(self, req_id: str, new_req_data: NewRequestData) -> CachedRequestState:
        """LWD: meta placeholder materialization for streaming updates
        (same as _update_states, zero base seams)."""
        embeds = new_req_data.prompt_embeds
        if embeds is not None and embeds.is_meta:
            new_req_data.prompt_embeds = torch.zeros(embeds.shape, dtype=embeds.dtype)
        return super()._update_streaming_request(req_id, new_req_data)

    def _init_mrope_positions(self, req_state: CachedRequestState) -> None:
        """LWD prompt-embeds-only: text-only M-RoPE positions do not depend
        on the ids content (the scan only honors vision markers), so build
        them directly from the prompt length; with ids present, fall back
        to the native scan (zero base seams)."""
        if req_state.prompt_token_ids is None:
            n = req_state.num_prompt_tokens
            req_state.mrope_positions = torch.arange(n).unsqueeze(0).expand(3, n).clone()
            req_state.mrope_position_delta = 0
            return
        super()._init_mrope_positions(req_state)

    @property
    def supports_mm_inputs(self) -> bool:
        """LWD: present injected batches as "no MM this step" so the native
        MM gate skips the encoder naturally (recomputed embeddings would
        wholesale-overwrite the remotely injected values); decode and
        text-only steps return the model's true value. Zero base seams."""
        input_batch = getattr(self, "input_batch", None)
        if input_batch is not None and input_batch.req_prompt_embeds:
            return False
        return getattr(self, "_model_supports_mm_inputs", False)

    @supports_mm_inputs.setter
    def supports_mm_inputs(self, value: bool) -> None:
        # The base __init__ assigns directly; keep the backing for the dynamic getter.
        self._model_supports_mm_inputs = value

    def _lwd_preprocess_prompt_embeds(
        self, scheduler_output: SchedulerOutput, num_input_tokens: int
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        """Consume the corrected device mask on the LWD cloud decoder path."""
        count = scheduler_output.total_num_scheduled_tokens
        is_token_ids = self.is_token_ids.gpu[:count]
        # Remote prompt rows do not carry meaningful token IDs.
        token_ids = self.input_ids.gpu[:count].masked_fill(~is_token_ids, 0)
        tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
        embeds = self.inputs_embeds.gpu[:count]
        embeds.copy_(torch.where(is_token_ids.unsqueeze(-1), tokens_to_embeds, embeds))

        # Match the parent's position layout and clear padding for graph replay.
        # The retired base exposed an xdrope-dimension attribute here; this
        # base splits positions two-way (mrope/plain) exactly like its own
        # dummy run, so xdrope models follow whichever layout the base uses.
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions[:num_input_tokens]
            if num_input_tokens > count:
                self.positions[count:num_input_tokens].zero_()

        # DSV4-only: the upstream model-layer fail-fast requires a non-None ids
        # tensor on hash MoE layers, but LWD keeps real token IDs off the cloud
        # entirely. Nothing true exists here, so the placeholder is
        # information-free by construction (all zeros): the gate below and the
        # remote expert-table publication in _model_forward key off the same
        # lwd_hash_state, so the router always swaps the tensor for row
        # indices before any read -- the values are dead on arrival. Other
        # models keep the native None unchanged.
        model_input_ids: torch.Tensor | None = None
        if self.lwd_hash_state is not None:
            model_input_ids = self._lwd_hash_placeholder_ids[:num_input_tokens]
        return (
            model_input_ids,
            self.inputs_embeds.gpu[:num_input_tokens],
            positions,
            None,
            self._init_model_kwargs(),
            None,
        )

    def _lwd_refresh_decode_embedding_mask(self, scheduler_output: SchedulerOutput) -> None:
        """Rebuild the mask from actual device positions so decode rows never
        reuse stale embeddings.

        Upload per-request prompt lengths asynchronously and expand them
        to token rows via the on-device request indices. The mask is
        written straight into the device buffer and consumed on-device
        by the LWD prompt-embeds path; positions are not round-tripped
        and the CPU mask is not refreshed.
        """
        num_reqs = len(self.input_batch.req_ids)
        count = scheduler_output.total_num_scheduled_tokens
        prompt_lens_gpu = self._lwd_prompt_lens_gpu
        assert prompt_lens_gpu is not None
        prompt_lens_gpu[:num_reqs].copy_(
            self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        prompt_lens_rows = prompt_lens_gpu[self.req_indices.gpu[:count]]
        self.is_token_ids.gpu[:count] = self.positions[:count] >= prompt_lens_rows

    # Clear the real-batch marker before cloud-side dummy forwards, so the
    # previous step's routing is never reused.
    def _dummy_run(self, *args, **kwargs) -> None:
        """Clear the real-batch Hash routing marker, then run the dummy forward."""
        # None marks "no real batch staged yet"; distinguishes dummy forwards.
        self._lwd_hash_step_tokens = None
        return super()._dummy_run(*args, **kwargs)

    def _model_forward(
        self,
        num_tokens_padded: int,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        """Publish the full Hash routing data into the forward context."""
        context = get_forward_context()
        if self.lwd_hash_state is not None:
            # Initialize the routing index cache first so padded rows use the legal zero index.
            self._lwd_hash_token_ids.zero_()
            if self._lwd_hash_step_tokens is None:
                # Synthetic forwards have no remote prompt; route dummy token 0.
                self._lwd_hash_prompt_mask.zero_()
            else:
                count = self._lwd_hash_step_tokens
                if count > num_tokens_padded:
                    raise ValueError("LWD Hash routing rows exceed model input rows")
                # Native input preparation retains real decode IDs even when
                # model input_ids=None because the model consumes embeddings.
                self._lwd_hash_token_ids[:count].copy_(self.input_ids.gpu[:count])
            context.input_ids = self._lwd_hash_token_ids[:num_tokens_padded]
            context.lwd_hash_expert_ids = self._lwd_hash_experts[:num_tokens_padded]
            context.lwd_hash_prompt_mask = self._lwd_hash_prompt_mask[:num_tokens_padded]
            if input_ids is None:
                # Covers the dummy/profile run: with enable_prompt_embeds it
                # also takes the inputs_embeds branch (input_ids=None), and
                # the model-layer hash fail-fast rejects None before the
                # router's dummy-token-0 path can run. Same contract as the
                # request-side placeholder: all zeros, never read (the
                # router swaps in row indices).
                input_ids = self._lwd_hash_placeholder_ids[:num_tokens_padded]
        return super()._model_forward(
            num_tokens_padded,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **model_kwargs,
        )

    def _lwd_hash_dims(self) -> tuple[int, int]:
        """(num_hash_layers, experts_per_tok), cached at first use."""
        dims = self._lwd_hash_dims_cached
        if dims is None:
            config = self.model_config.hf_config
            dims = (
                hash_layer_count(config),
                getattr(config, "num_experts_per_tok", 0),
            )
            self._lwd_hash_dims_cached = dims
        return dims

    def _lwd_release_consumed_prompt_embeds(self) -> None:
        """Release assembly buffers whose prompt was fully consumed.

        A buffer becomes releasable once the request's computed tokens
        cover the whole prompt: the last chunk's rows were consumed by
        the previous step's fill loop, and the draft proposer read its
        window during that step's sampling (which precedes this call).
        Decode steps never read prompt embeds, so dropping the entry
        here cannot corrupt anything.  Entries follow native index
        compaction (condense/swap), so keying by idx stays correct.

        Also drops orphan entries whose slot no longer holds a live
        request (preemption removes the request without finishing it, so
        the flush hook never fires): a stale buffer at a recycled index
        would otherwise be mistaken for the new occupant's embeds.
        """
        if self.lwd_hash_state is not None:
            for req_id in list(self.lwd_hash_state.requests):
                request = self.requests.get(req_id)
                if request is None or request.num_computed_tokens >= request.num_prompt_tokens:
                    self.lwd_hash_state.discard(req_id)
        embeds_map = self.input_batch.req_prompt_embeds
        if not embeds_map:
            return
        num_prompt = self.input_batch.num_prompt_tokens
        computed = self.input_batch.num_computed_tokens_cpu
        slot_req_ids = self.input_batch.req_ids
        for idx in list(embeds_map.keys()):
            if idx >= len(computed):
                embeds_map.pop(idx)
                continue
            if idx >= len(slot_req_ids) or slot_req_ids[idx] is None:
                embeds_map.pop(idx)
                continue
            if computed[idx] >= num_prompt[idx]:
                embeds_map.pop(idx)

    def lwd_release_finished(self, finished_req_ids: list[str]) -> None:
        """Release collection bookkeeping for finished requests (sole owner of
        the release path): the Hash chunk table plus prompt-embeds assembly
        buffers. Called by the worker on the engine's finished_req_ids signal
        (backstop for prefill aborted mid-way; normal prefill completion is
        already released in _lwd_release_consumed_prompt_embeds)."""
        embeds_map = self.input_batch.req_prompt_embeds
        req_id_to_index = self.input_batch.req_id_to_index
        for req_id in finished_req_ids:
            self._lwd_request_connections.pop(req_id, None)
            if self.lwd_hash_state is not None:
                self.lwd_hash_state.discard(req_id)
            idx = req_id_to_index.get(req_id)
            if idx is not None:
                embeds_map.pop(idx, None)

    def _lwd_harvest_up_batches(
        self,
        num_scheduled_tokens: np.ndarray,
        scheduler_output: SchedulerOutput | None = None,
    ) -> list[LwdUpWindow]:
        """Stage and broadcast each connection's native prompt window."""
        plan = self._lwd_prompt_plan(num_scheduled_tokens)
        if not plan:
            return []
        worker = self.worker
        if worker is None:
            raise RuntimeError("[Lwd][cloud-runner] cloud runner has no worker")
        request_connections = getattr(self, "_lwd_request_connections", None)
        if request_connections is None:
            request_connections = self._lwd_request_connections = {}
        connections = getattr(worker, "_lwd_connections", ())

        def request_connection(req_id: str) -> ConnectionId:
            # Canonical cloud request IDs carry their source edge. A native
            # mixed batch can contain several edges, so defaulting to edge 0
            # would select the wrong UP/DOWN FIFO for the other requests.
            edge_id, _ = parse_edge_id(req_id)
            candidates = [c for c in connections if c.edge_id == edge_id]
            if len(candidates) != 1:
                raise RuntimeError(f"[Lwd][cloud-runner] req={req_id} has no unique local connection")
            return candidates[0]

        grouped: dict[ConnectionId, list[LwdPromptWindow]] = {}
        for window in plan:
            connection = request_connection(window.req_id)
            request_connections[window.req_id] = connection
            grouped.setdefault(connection, []).append(window)

        hidden_size = self.model_config.get_hidden_size()
        num_hash_layers, top_k = self._lwd_hash_dims()
        tail_per_row = 4 * num_hash_layers * top_k
        windows: list[LwdUpWindow] = []
        tp = get_tp_group()
        for connection in sorted(grouped):
            connection_plan = grouped[connection]
            numel = sum(window.rows for window in connection_plan) * (hidden_size + tail_per_row)
            connection_endpoint = worker._lwd_up_src_ranks[connection] == dist.get_rank()
            if connection_endpoint:
                stager = worker._lwd_stager
                if stager is None:
                    raise RuntimeError("[Lwd][cloud-runner] endpoint rank without host stager")
                dev, win_idx = self._lwd_window_device(numel, min_depth=len(grouped))
                self._lwd_stage_window(stager, connection_plan, dev, win_idx, hidden_size, tail_per_row)
            else:
                dev = torch.empty(numel, dtype=torch.bfloat16, device=self.device)
                win_idx = None
            if tp.world_size > 1:
                work = dist.broadcast(
                    dev,
                    src=worker._lwd_up_src_ranks[connection],
                    group=tp.device_group,
                    async_op=True,
                )
                work.wait()
            windows.append(LwdUpWindow(connection, dev, win_idx, connection_plan))
        return windows

    def _lwd_request_connection(self, req_id: str) -> ConnectionId:
        """Resolve one request's UP/DOWN connection without a default route.

        Prefers the stamp recorded at harvest time. A fully prefix-cached
        request never enters a prompt window and is therefore never stamped;
        resolve it from its canonical edge identity instead of falling back
        to an arbitrary connection, which would send the hidden packet to
        the wrong edge (or drop it) in a multi-edge batch.
        """
        stamped = getattr(self, "_lwd_request_connections", {}).get(req_id)
        if stamped is not None:
            return stamped
        connections = getattr(self.worker, "_lwd_connections", ()) if self.worker is not None else ()
        edge_id, _ = parse_edge_id(req_id)
        candidates = [c for c in connections if c.edge_id == edge_id]
        if len(candidates) != 1:
            raise RuntimeError(f"[LWD][cloud-runner] req={req_id} has no unique local connection for edge {edge_id}")
        return candidates[0]

    def _lwd_prompt_plan(self, num_scheduled_tokens: np.ndarray) -> list[LwdPromptWindow]:
        """Derive prompt-only rows from native request order and watermarks."""
        plan: list[LwdPromptWindow] = []
        computed = self.input_batch.num_computed_tokens_cpu
        prompt_lens = self.input_batch.num_prompt_tokens
        for idx, req_id in enumerate(self.input_batch.req_ids):
            if idx >= len(num_scheduled_tokens):
                break
            start = int(computed[idx])
            prompt_len = int(prompt_lens[idx])
            scheduled = int(num_scheduled_tokens[idx])
            rows = min(scheduled, prompt_len - start)
            if rows > 0:
                plan.append(LwdPromptWindow(req_id, idx, start, rows))
        return plan

    def _lwd_stage_window(
        self,
        stager: LwdHostStager,
        plan: list[LwdPromptWindow],
        dev: torch.Tensor,
        win_idx: int,
        hidden_size: int,
        tail_per_row: int,
    ) -> None:
        """Reassemble host embeddings and expert tails on the dedicated H2D stream."""
        started = time.monotonic()
        embed_numel = sum(window.rows for window in plan) * hidden_size
        stream = self._lwd_h2d_stream()
        used_event = self._lwd_win_used_ev[win_idx]
        with torch.npu.stream(stream):
            if used_event is not None:
                stream.wait_event(used_event)
            embed_offset = 0
            tail_offset = embed_numel
            for window in plan:
                self._lwd_stage_one_request(stager, window, dev, embed_offset, hidden_size)
                embed_offset += window.rows * hidden_size
                if tail_per_row:
                    req = stager.take_req(window.req_id)
                    assert req is not None
                    tail_start, _ = req.tail_range_at(window.start, window.rows)
                    numel = window.rows * tail_per_row
                    # Same first-copy-loss risk, same helper (expert-ID bytes
                    # may legitimately be zero: retry only, no fail-fast).
                    self._lwd_h2d_copy_verified(
                        dev[tail_offset : tail_offset + numel],
                        req.host[tail_start : tail_start + numel],
                        window.req_id,
                        strict=False,
                    )
                    tail_offset += numel
            ready_event = torch.npu.Event()
            ready_event.record(stream)
        torch.npu.current_stream().wait_event(ready_event)
        for window in plan:
            stager.consume(window.req_id, window.rows)
        lwd_logger.info(
            "[Lwd][cloud-runner] CONSUME rows=%d reqs=%d stage=%.2fms",
            sum(window.rows for window in plan),
            len(plan),
            (time.monotonic() - started) * 1000,
        )

    def _lwd_h2d_copy_verified(self, dst: torch.Tensor, src: torch.Tensor, rid: str, strict: bool = True) -> None:
        """Pinned-to-device-window H2D copy (sync + read-back check + one retry).

        Root-cause fix for the 2E2C cold-start first window (2026-11, ported
        from cr_mi_to_23.0_azy_1029_memc b471c205b): right after a segmented
        D2H lands the edge prompt embeddings in a pinned buffer, the **first**
        ``copy_`` issued in the H2D stream context can be silently dropped by
        torch_npu (first-read transient) — the source was verified non-zero,
        the window memory is writable, and an immediate same-parameter retry
        recovers bit-for-bit. A lost first copy hands the model all-zero
        embeddings, degrading the output to an unconditional continuation
        from BOS ("answers an unrelated question"): only the first few
        requests after startup, only on non-group-leader endpoint ranks,
        self-heals afterwards, no error anywhere.

        The check compares the first 16 destination elements with the same
        source elements, including matching NaNs. All-zero and zero-sum data
        therefore pass the element-wise comparison — but a full-buffer
        absolute-sum guard catches the case where a lost copy leaves dst
        all-zero while src is non-zero and the head-16 sample happens to
        match (both zero).  ``strict=False`` is for the expert-ID tail: retry
        a mismatch once, but never fail; the retry is idempotent. The
        standing cost is one blocking copy plus a bounded sample comparison
        and one reduction on the H2D stream.

        Sync scope: the read-back only needs the *issuing* stream drained,
        which the blocking ``copy_`` already guarantees, so a stream-level
        synchronize suffices. Cross-stream visibility towards the compute
        stream is owned by the ``ready_event`` bridge recorded after the
        window is staged. A device-wide ``torch.npu.synchronize()`` would
        additionally park the engine thread behind every unrelated
        in-flight stream (previous window's forward tail, DOWN sends,
        early-recv HCCL ops) at each staging point with no correctness
        gain, and is deliberately avoided here.
        """
        dst.copy_(src, non_blocking=False)
        torch.npu.current_stream().synchronize()
        dst_sample = dst[:_LWD_H2D_VERIFY_ELEMENTS]
        src_sample = src[:_LWD_H2D_VERIFY_ELEMENTS]

        def samples_match() -> bool:
            # Full-buffer guard: a silently dropped copy_ leaves dst all
            # zero; if src is non-zero this is always a failure even when
            # the head-16 sample happens to match (both zero).
            if float(src.abs().sum()) != 0.0 and float(dst.abs().sum()) == 0.0:
                return False
            try:
                if torch.equal(dst_sample, src_sample):
                    return True
            except RuntimeError:
                # Some backends require both operands on the same device.
                pass
            dst_cpu = dst_sample.cpu()
            src_cpu = src_sample.cpu()
            if torch.equal(dst_cpu, src_cpu):
                return True
            if dst_cpu.is_floating_point() or dst_cpu.is_complex():
                return bool(torch.all((dst_cpu == src_cpu) | (torch.isnan(dst_cpu) & torch.isnan(src_cpu))).item())
            return False

        if samples_match():
            return
        lwd_logger.warning(
            "[LWD][cloud-runner] H2D first-copy mismatch req=%s (sample differs after sync copy; retrying)",
            rid,
        )
        dst.copy_(src, non_blocking=False)
        torch.npu.current_stream().synchronize()
        if strict and not samples_match():
            raise RuntimeError(
                f"[LWD][cloud-runner] H2D copy mismatch twice req={rid} "
                "(source and destination samples differ after retry)"
            )

    def _lwd_h2d_stream(self) -> torch.npu.Stream:
        """Return the dedicated stream used for host-window copies."""
        stream = getattr(self, "_lwd_h2d_copy_stream", None)
        if stream is None:
            stream = torch.npu.Stream()
            self._lwd_h2d_copy_stream = stream
        return stream

    def _lwd_window_device(self, numel: int, min_depth: int = _LWD_WINDOW_MIN_DEPTH) -> tuple[torch.Tensor, int]:
        """Return a rotating device window protected by a prior-use event."""
        buffers = getattr(self, "_lwd_win_dev", None)
        depth = max(_LWD_WINDOW_MIN_DEPTH, min_depth)
        if buffers is not None and buffers[0].numel() >= numel:
            while len(buffers) < depth:
                buffers.append(torch.empty(buffers[0].numel(), dtype=torch.bfloat16, device=self.device))
                self._lwd_win_used_ev.append(None)
        else:
            for event in getattr(self, "_lwd_win_used_ev", ()):
                if event is not None:
                    event.synchronize()
            hidden_size = self.model_config.get_hidden_size()
            num_hash_layers, top_k = self._lwd_hash_dims()
            per_row = hidden_size + 4 * num_hash_layers * top_k
            budget = self.vllm_config.scheduler_config.max_num_batched_tokens
            capacity = max(numel, per_row * budget)
            buffers = [torch.empty(capacity, dtype=torch.bfloat16, device=self.device) for _ in range(depth)]
            self._lwd_win_dev = buffers
            self._lwd_win_flip = 0
            self._lwd_win_used_ev = [None] * depth
        win_idx = self._lwd_win_flip % len(buffers)
        self._lwd_win_flip = (win_idx + 1) % len(buffers)
        return buffers[win_idx][:numel], win_idx

    def _lwd_stage_one_request(
        self,
        stager: LwdHostStager,
        window: LwdPromptWindow,
        dev: torch.Tensor,
        elem_offset: int,
        hidden_size: int,
    ) -> None:
        """Copy one host window after enforcing the native read watermark."""
        computed = self.input_batch.num_computed_tokens_cpu
        expect = int(computed[window.req_index])
        consumed = stager.consumed_rows(window.req_id)
        if consumed != expect or consumed != window.start:
            raise RuntimeError(
                f"[Lwd][cloud-runner] STAGE POSITION MISMATCH req={window.req_id}: "
                f"host read pointer consumed={consumed}, computed={expect}, "
                f"window_start={window.start}"
            )
        host_start, numel, _ = stager.segment_for(window.req_id, window.rows)
        if numel != window.rows * hidden_size:
            raise RuntimeError(f"[Lwd][cloud-runner] invalid host window for {window.req_id}")
        req = stager.take_req(window.req_id)
        assert req is not None
        # Cold-start first-window silent copy loss: verify + retry (see helper).
        self._lwd_h2d_copy_verified(
            dev[elem_offset : elem_offset + numel],
            req.host[host_start : host_start + numel],
            window.req_id,
        )

    def _lwd_inject_remote_embeds(
        self,
        dev: torch.Tensor | None,
        win_idx: int | None,
        plan: list[LwdPromptWindow],
        num_scheduled_tokens: np.ndarray,
    ) -> None:
        """Inject the broadcast windows into native flattened embedding rows."""
        if dev is None:
            return
        hidden_size = self.model_config.get_hidden_size()
        num_hash_layers, top_k = self._lwd_hash_dims()
        tail_per_row = 4 * num_hash_layers * top_k
        embed_numel = sum(window.rows for window in plan) * hidden_size
        embeds = dev[:embed_numel].view(-1, hidden_size)
        offsets: list[int] = []
        offset = 0
        for idx in range(len(self.input_batch.req_ids)):
            offsets.append(offset)
            if idx < len(num_scheduled_tokens):
                offset += int(num_scheduled_tokens[idx])
        row = 0
        for window in plan:
            if self.lwd_hash_state is not None:
                tail_start = embed_numel + row * tail_per_row
                tail_end = embed_numel + (row + window.rows) * tail_per_row
                expert_ids = self._lwd_decode_expert_ids(
                    dev[tail_start:tail_end],
                    window.rows,
                    num_hash_layers,
                    top_k,
                )
                self.lwd_hash_state.add_chunk(
                    window.req_id,
                    int(self.input_batch.num_prompt_tokens[window.req_index]),
                    window.start,
                    expert_ids,
                )
            output = offsets[window.req_index]
            self.inputs_embeds.gpu[output : output + window.rows].copy_(
                embeds[row : row + window.rows],
                non_blocking=True,
            )
            # The direct-GPU MTP first pass reads this same authoritative window.
            self.input_batch.is_token_ids[
                window.req_index,
                window.start : window.start + window.rows,
            ] = False
            row += window.rows
        if win_idx is not None:
            used_event = torch.npu.Event()
            used_event.record()
            self._lwd_win_used_ev[win_idx] = used_event
        else:
            dev.record_stream(torch.npu.current_stream())

    @staticmethod
    def _lwd_decode_expert_ids(
        seg: torch.Tensor,
        rows: int,
        num_layers: int,
        top_k: int,
    ) -> torch.Tensor:
        """Decode numeric byte values into int32 IDs without losing values above 255."""
        return seg.contiguous().to(torch.uint8).view(torch.int32).view(rows, num_layers, top_k)

    def _lwd_prime_mrope_positions(self, req_id: str, payload: bytes, prompt_len: int) -> None:
        """Write full-prompt positions and derive the completion-position delta."""
        req_state = self.requests.get(req_id)
        if req_state is None:
            raise RuntimeError(f"[Lwd][cloud-runner] mrope prime req={req_id}: req_state not found")
        expected_numel = 3 * prompt_len
        if len(payload) != expected_numel * 8:
            raise RuntimeError(
                f"[Lwd][cloud-runner] mrope prime req={req_id}: payload bytes "
                f"{len(payload)} != 3 * prompt_len * 8 ({expected_numel * 8})"
            )
        positions = torch.frombuffer(bytearray(payload), dtype=torch.int64).view(3, prompt_len)
        req_state.mrope_positions[:, :prompt_len] = positions
        req_state.mrope_position_delta = int(positions.max().item()) + 1 - prompt_len
        lwd_logger.info(
            "[Lwd][cloud-runner] mrope primed: req=%s prompt=%d delta=%d",
            req_id,
            prompt_len,
            req_state.mrope_position_delta,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        """Capture the sampler output for the collect probe
        (the base sample_tokens clears execute_model_state on return)."""
        sampler_output = super()._sample(logits, spec_decode_metadata)
        self._lwd_captured_sampler_output = sampler_output
        return sampler_output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput | None) -> ModelRunnerOutput:
        # Every TP rank collects its sampler/hidden state. Each connection's
        # payload is retained only on that connection's real channel endpoint.
        captured = None
        if self.execute_model_state is not None:
            # ExecuteModelState layout (see model_runner_v1.sample_tokens):
            # (scheduler_output, logits, spec_decode_metadata,
            #  spec_decode_common_attn_metadata, hidden_states,
            #  sample_hidden_states, ...)
            state = self.execute_model_state
            # Snapshot the batch order/mask as well: with overlapping
            # batch queues, the next step's _prepare_inputs may reorder
            # input_batch before the collect; reading live state would
            # slice batch-A tensors with batch-B order (the root cause of
            # high-concurrency misalignment)
            captured = (
                state[5],
                state[1],
                state[2],
                state[0],
                list(self.input_batch.req_ids),
                self.discard_request_mask.np.copy(),
            )
        self._lwd_captured_sampler_output = None
        output = super().sample_tokens(grammar_output)
        if captured is not None and self._lwd_captured_sampler_output is not None:
            payload = self._lwd_collect_down_payload(
                captured[0],
                captured[1],
                captured[2],
                captured[3],
                captured[4],
                captured[5],
                self._lwd_captured_sampler_output,
            )
            # Keep every connection on every rank: TP0 returns all C2E
            # metadata to the engine, while each real channel endpoint sends
            # only its own DOWN tensor in the worker.
            self._lwd_pending_down_payload = payload
        return output

    def take_lwd_pending_down_payload(self) -> list[LwdDownPayload] | None:
        """Take this step's independent per-connection DOWN payloads."""
        payload = getattr(self, "_lwd_pending_down_payload", None)
        self._lwd_pending_down_payload = None
        return payload

    @torch.inference_mode()
    def _lwd_collect_down_payload(
        self,
        sample_hidden_states: torch.Tensor | None,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        scheduler_output: SchedulerOutput,
        batch_req_ids: list[str],
        discard_mask_np: np.ndarray,
        sampler_output: SamplerOutput,
    ) -> list[LwdDownPayload]:
        """Production DOWN collection (rank-replay): hidden packaging + global
        ranks + num_accepted.

        Orchestration: row selection / rank computation branches on spec
        vs non-spec (_lwd_pack_plain_down_rows / _lwd_pack_spec_down_rows).
        The step meta stays on device (no pinned D2H here — the side-stream
        copy was not fenceable on this platform); a per-step ready event is
        recorded on the main stream, and the fenced harvest (dedicated
        stream + wait_event + same-stream done event) runs in the executor's
        output thread, off this collect. Zero added synchronization on the
        compute critical path; batch order/mask come from the capture-time
        snapshot (guards against batch-queue overlap reordering).
        Returns:
          (hidden_packet, meta_dev, seg_lens_host, ready_event, req_ids)
        or None (no in-flight requests this step).
        """
        _t0 = time.monotonic()
        sampled = sampler_output.sampled_token_ids
        if sampled is None or sampled.dim() != 2 or logits is None:
            return []
        # batch_req_ids/discard_mask come from the capture-time snapshot
        # (same batch order as logits/hidden); the live input_batch is
        # never read here
        valid = ~discard_mask_np[: len(batch_req_ids)]
        groups = sorted({self._lwd_request_connection(req_id) for req_id in batch_req_ids})
        payloads: list[LwdDownPayload] = []
        for connection in groups:
            group_valid = valid & np.asarray(
                [self._lwd_request_connection(req_id) == connection for req_id in batch_req_ids],
                dtype=np.bool_,
            )
            if spec_decode_metadata is None:
                packed = self._lwd_pack_plain_down_rows(
                    sample_hidden_states,
                    logits,
                    sampled,
                    group_valid,
                    batch_req_ids,
                )
            else:
                packed = self._lwd_pack_spec_down_rows(
                    sample_hidden_states,
                    logits,
                    sampled,
                    spec_decode_metadata,
                    group_valid,
                    batch_req_ids,
                )
            if packed is None:
                continue
            hidden_packet, ranks_list, counts_dev, seg_lens_host = packed
            if seg_lens_host is not None:
                meta_dev = torch.cat(ranks_list + [counts_dev])
            else:
                n_req = sum(1 for i in range(len(batch_req_ids)) if group_valid[i])
                counts_dev = torch.ones(n_req, dtype=torch.int32, device=logits.device)
                meta_dev = torch.cat(ranks_list + [counts_dev, counts_dev])
            selection_done = time.monotonic()
            ready_event = torch.npu.Event()
            ready_event.record()
            lwd_logger.info(
                "[Lwd][perf] collect connection=%s rows=%d rank=%.2f total=%.2fms",
                connection,
                hidden_packet.shape[0],
                (selection_done - _t0) * 1000,
                (time.monotonic() - _t0) * 1000,
            )
            payloads.append(
                LwdDownPayload(
                    connection,
                    hidden_packet,
                    meta_dev,
                    seg_lens_host,
                    ready_event,
                    [r for i, r in enumerate(batch_req_ids) if group_valid[i]],
                )
            )
        return payloads

    def _lwd_pack_plain_down_rows(
        self,
        sample_hidden_states: torch.Tensor | None,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        valid: np.ndarray,
        batch_req_ids: list[str],
    ) -> LwdDownSelection | None:
        """Non-spec row selection and rank computation (correspondence:
        equivalent to the not-is_spec branch of the original
        _lwd_collect_down_payload).

        Cheap computation: direct bf16 comparison (logits are natively
        bf16, bit-identical to a cast) and full-batch rows computed
        wholesale when fully covered (no fancy-index gathers)."""
        idx = [i for i in range(len(batch_req_ids)) if valid[i]]
        if not idx:
            return None
        full_cover = all(valid[: len(batch_req_ids)])
        lg_sel = logits if full_cover else logits[idx]
        sm_sel = sampled[:, 0] if full_cover else sampled[idx][:, 0]
        ranks = [self._lwd_token_rank_encoder.encode(lg_sel, sm_sel)]
        hidden_packet = torch.cat([sample_hidden_states[i : i + 1] for i in idx])
        return LwdDownSelection(
            hidden_packet=hidden_packet,
            ranks=ranks,
            counts_dev=None,
            seg_lens_host=None,
        )

    def _lwd_pack_spec_down_rows(
        self,
        sample_hidden_states: torch.Tensor,
        logits: torch.Tensor,
        sampled: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata,
        valid: np.ndarray,
        batch_req_ids: list[str],
    ) -> LwdDownSelection | None:
        """All-zero synchronous spec row selection and rank computation
        (vectorized; correspondence: equivalent to the is_spec branch of
        the original _lwd_collect_down_payload).

        Segment lengths derive from the host-side
        scheduled_spec_decode_tokens (1+draft_len); rows are packed as
        whole segments (rejected rows included; the edge takes the valid
        prefix by num_accepted). Lengths come from
        spec_decode_metadata.num_draft_tokens (host list, the real
        runtime layout, matching the sample_hidden_states segment
        structure), NOT from
        scheduler_output.scheduled_spec_decode_tokens (scheduler input
        that may disagree with what actually ran). counts derive from
        the same sampled tensor on-device (same source as ranks/rows,
        naturally batch-position aligned); seg_lens/counts only include
        valid requests so [ranks|counts|seg_lens] stay length-aligned.
        One host pass builds the indices, replacing the per-request
        device loop (the original issued ~6 small kernels per request
        and the rank segment grew past 20ms with concurrency; after
        vectorization the launch count is independent of N). Index
        semantics are bitwise-equivalent to the original loop:
        vidx/vlen = valid request batch position / actual row count
        (capped by total segment length, = the original rows_i);
        req_idx/pos_idx = per-row request position / in-segment offset
        (the original sampled[i, :rows_i]); row_sel = global row number
        of each valid row."""
        seg_lens = [d + 1 for d in spec_decode_metadata.num_draft_tokens]
        counts_all = (sampled != -1).sum(dim=1)
        seg_lens_np = np.asarray(seg_lens, dtype=np.int32)
        seg_starts = np.cumsum(seg_lens_np) - seg_lens_np
        total_rows = sample_hidden_states.shape[0]
        rows_per_req = np.minimum(seg_lens_np, np.maximum(total_rows - seg_starts, 0))
        vidx_np = np.nonzero(valid[: len(batch_req_ids)] & (rows_per_req >= 1))[0].astype(np.int64)
        if vidx_np.size == 0:
            return None
        vlen_np = rows_per_req[vidx_np].astype(np.int64)
        starts_v = seg_starts[vidx_np].astype(np.int64)
        n_rows = int(vlen_np.sum())
        req_idx_np = np.repeat(vidx_np, vlen_np)
        pos_idx_np = np.arange(n_rows, dtype=np.int64) - np.repeat(np.cumsum(vlen_np) - vlen_np, vlen_np)
        row_sel_np = np.repeat(starts_v, vlen_np) + pos_idx_np
        # All indices go through one pinned async H2D (a pageable
        # synchronous H2D would stall the main stream waiting for the
        # device to drain — the same lesson as seg_lens)
        buf = self._lwd_h2d_stage_i64(np.concatenate([req_idx_np, pos_idx_np, row_sel_np, vidx_np, vlen_np]))
        n_valid = vidx_np.size
        req_d, pos_d = buf[:n_rows], buf[n_rows : 2 * n_rows]
        sel_d = buf[2 * n_rows : 3 * n_rows]
        vidx_d = buf[3 * n_rows : 3 * n_rows + n_valid]
        vlen_d = buf[3 * n_rows + n_valid :]
        sampled_rows = sampled[req_d, pos_d]
        if n_rows == total_rows:
            # All valid with no capping: rows are a contiguous prefix; skip the gather
            lg_rows = logits[:n_rows]
            hidden_packet = sample_hidden_states[:n_rows]
        else:
            lg_rows = logits[sel_d]
            hidden_packet = sample_hidden_states[sel_d]
        ranks = [self._lwd_token_rank_encoder.encode(lg_rows, sampled_rows)]
        # Stale sampled slots (requests the spec step skipped may retain
        # the previous spec step's tokens): accepted/ranks/seg_lens are
        # all capped by the actual hidden row count — the row count is
        # the ground truth of real sampling positions
        counts_dev = torch.minimum(counts_all[vidx_d], vlen_d).to(torch.int32)
        return LwdDownSelection(
            hidden_packet=hidden_packet,
            ranks=ranks,
            counts_dev=counts_dev,
            seg_lens_host=vlen_np.tolist(),
        )

    def _lwd_h2d_stage_i64(self, arr: np.ndarray) -> torch.Tensor:
        """Async H2D staging for int64 host index arrays (pinned 4-slot
        rotation + non_blocking): replaces torch.tensor(arr, device=...)
        pageable synchronous copies — those stall the main stream waiting
        for the device to drain (under MTP, the draft-propose tail).
        Ring depth rationale mirrors the meta ring: cover batch_queue
        depth + engine lag so a later host overwrite cannot land while
        the previous copy is still queued.
        Correspondence: LwdPinnedStaging.stage_to_device ≍ the original
        hand-rolled pinned+device 4-slot pair rotation (capacity floor
        1024 unchanged)."""
        return self._lwd_staging(
            "down_index",
            dtype=torch.int64,
            min_capacity=1024,
        ).stage_to_device(arr, arr.size)
