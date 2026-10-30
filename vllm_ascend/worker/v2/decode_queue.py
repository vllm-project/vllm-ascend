# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Host-only admission and readiness tracking for early decode submission.

Models explicitly advertise an audited CPU-length capability through their
model state. Request lifecycle, uniform speculative batches and exact snapshot
ownership are framework concerns and do not depend on a model architecture.
Admission and snapshot recording never read pending pinned data. The consumer
retains the original event synchronization whenever exact host counts are used.
"""

from typing import Any, Protocol, cast

import numpy as np

from vllm_ascend.worker.v2.mm_decode_queue import (
    MMDecodeCertificateState,
    MMDecodeRequestKey,
    make_mm_decode_request_key,
)
from vllm_ascend.worker.v2.pd_decode_queue import PDDecodeQueueState


class DecodeQueueCapability(Protocol):
    """Model-state proof of the CPU consumers that can accept a length bound."""

    def supports(self, runner: Any) -> bool: ...

    def mm_context(self, runner: Any) -> tuple | None: ...


class DecodeQueueController:
    """One runner's optional, fail-closed decode submission controller."""

    def __init__(self, enabled: bool = False, multimodal: bool = False, full_graph_mode: Any = None) -> None:
        self.enabled = enabled
        self.multimodal = enabled and multimodal
        self.full_graph_mode = full_graph_mode
        self.snapshot_generation = 0
        self.snapshot_source_stream: Any = None
        self.mm_snapshot_stream: Any = None
        self.mm_decode_epoch = 0
        self.mm_decode_state = MMDecodeCertificateState()
        self.stats = {
            "eligible_steps": 0,
            "fallback_steps": 0,
            "exact_snapshots_recorded": 0,
            "host_length_wait_skipped": 0,
            "length_device_wait_enqueued": 0,
        }
        self.multimodal_stats = {"exact_certified_batches": 0, "queue_eligible_steps": 0, "invalidations": 0}
        self.last_guard: dict[str, Any] | None = None
        self._model_state_owner: Any = None
        self._pd_state: PDDecodeQueueState | None = None

    def _remember_pd_state(self, state: PDDecodeQueueState | None) -> None:
        if state is None or state is self._pd_state:
            return
        if self._pd_state is not None:
            # Replacing a connector must not carry either connector's ready
            # certificate across that ownership change. Preserve observations
            # made by a newly installed connector on its first binding only.
            self._pd_state.invalidate()
            state.invalidate()
        self._pd_state = state

    def _set_model_state_owner(self, owner: Any) -> None:
        if owner is self._model_state_owner:
            return
        had_owner = self._model_state_owner is not None
        self._model_state_owner = owner
        self.snapshot_generation = 0
        self.snapshot_source_stream = None
        self.mm_snapshot_stream = None
        self.mm_decode_state.invalidate()
        if had_owner:
            if self._pd_state is not None:
                self._pd_state.invalidate()
            self.mm_decode_epoch += 1
            self.multimodal_stats["invalidations"] += 1

    def _capability(self, runner: Any) -> DecodeQueueCapability | None:
        if not self.enabled:
            self._set_model_state_owner(None)
            return None
        try:
            owner = runner.model_state
            factory = getattr(owner, "get_decode_queue_capability", None)
            capability = factory(runner) if callable(factory) else None
            if callable(getattr(capability, "supports", None)):
                # A recreated model state cannot reuse its predecessor's exact
                # snapshot or media-prompt certificates, even on the same stream.
                self._set_model_state_owner(owner)
                return cast(DecodeQueueCapability, capability)
        except Exception:
            # Optional admission must not add errors for other model states.
            pass
        self._set_model_state_owner(None)
        return None

    def model_enabled(self, runner: Any) -> bool:
        """Check model opt-in before installing optional connector observers.

        Attention groups may not yet exist at connector initialization. Their
        runtime capability proof is therefore checked separately in can_queue.
        """
        return self._capability(runner) is not None

    def before_requests(self, output: Any) -> None:
        if not self.multimodal:
            return
        try:
            changed = bool(
                output.scheduled_new_reqs
                or output.scheduled_cached_reqs.resumed_req_ids
                or output.finished_req_ids
                or output.preempted_req_ids
                or output.scheduled_encoder_inputs
                or output.free_encoder_mm_hashes
            )
        except Exception:
            changed = True
        if changed:
            self.mm_decode_epoch += 1
            self.mm_decode_state.invalidate()
            self.multimodal_stats["invalidations"] += 1

    @staticmethod
    def batch_is_stable_decode(output: Any, batch: Any) -> bool:
        try:
            return (
                not output.scheduled_new_reqs
                and not output.scheduled_cached_reqs.resumed_req_ids
                and not output.preempted_req_ids
                and not output.finished_req_ids
                and not output.scheduled_encoder_inputs
                and not batch.has_prefill
                and not bool(np.any(batch.is_prefilling_np))
                and (batch.prefill_runs_as_decode_np is None or not bool(np.any(batch.prefill_runs_as_decode_np)))
            )
        except Exception:
            return False

    def _mm_request_keys(
        self, runner: Any, req_ids: Any, slots: Any, prefill_lens: Any, *, prefill_by_slot: bool = False
    ) -> tuple[MMDecodeRequestKey, ...] | None:
        if not self.multimodal:
            return None
        capability = self._capability(runner)
        if capability is None:
            return None
        try:
            model_context = capability.mm_context(runner)
            if not isinstance(model_context, tuple) or not req_ids or len(req_ids) != len(slots):
                return None
            context = (self.mm_decode_epoch, *model_context)
            keys = []
            for row, (req_id, raw_slot) in enumerate(zip(req_ids, slots)):
                slot = int(raw_slot)
                if runner.req_states.req_id_to_index.get(req_id) != slot:
                    return None
                prefill_len = int(prefill_lens[slot if prefill_by_slot else row])
                key = make_mm_decode_request_key(
                    req_id, slot, prefill_len, context, runner.encoder_cache.mm_features[req_id]
                )
                if key is None:
                    return None
                keys.append(key)
            return tuple(keys)
        except Exception:
            return None

    def record_snapshot(self, runner: Any, source_stream: Any, pd_state: PDDecodeQueueState | None = None) -> None:
        """Record ownership after the original exact D2H and event enqueue."""
        if self.enabled:
            self._remember_pd_state(pd_state)
        if not self.model_enabled(runner):
            return
        self.snapshot_generation += 1
        self.snapshot_source_stream = source_stream
        self.stats["exact_snapshots_recorded"] += 1
        if pd_state is not None:
            try:
                pd_state.record_snapshot(runner.req_states.req_id_to_index, self.snapshot_generation)
            except Exception:
                pd_state.invalidate()
        if self.multimodal:
            self.mm_snapshot_stream = source_stream
            try:
                indices = runner.req_states.req_id_to_index
                keys = self._mm_request_keys(
                    runner,
                    tuple(indices),
                    tuple(indices.values()),
                    runner.req_states.prefill_len.np,
                    prefill_by_slot=True,
                )
            except Exception:
                keys = None
            self.mm_decode_state.record_snapshot(keys or ())

    def certify_exact(
        self,
        runner: Any,
        output: Any,
        batch: Any,
        current_stream: Any,
        pd_state: PDDecodeQueueState | None = None,
    ) -> None:
        """Certify only after the caller consumed the exact synchronized path.

        The runner calls this after its original event.synchronize(). The
        private pinned snapshot is never accessed by record_snapshot/can_queue.
        """
        if self.enabled:
            self._remember_pd_state(pd_state)
        if not self.model_enabled(runner) or (not self.multimodal and pd_state is None):
            return
        try:
            if (
                runner.speculator is None
                or runner.sync_spec_pp_cpu_counts
                or runner._can_skip_spec_cpu_length_sync()
                or self.snapshot_source_stream != current_stream
                or not self.batch_is_stable_decode(output, batch)
            ):
                return
            keys = (
                self._mm_request_keys(runner, batch.req_ids, batch.idx_mapping_np, batch.prefill_len_np)
                if self.multimodal and self.mm_snapshot_stream == current_stream
                else None
            )
            if keys is None and pd_state is None:
                return
            exact_counts = runner.num_computed_tokens_cpu.numpy()[batch.idx_mapping_np].tolist()
            if keys is not None and self.mm_decode_state.certify(keys, exact_counts):
                self.multimodal_stats["exact_certified_batches"] += 1
            if pd_state is not None:
                pd_state.certify(
                    batch.req_ids, batch.idx_mapping_np, batch.prefill_len_np, exact_counts, self.snapshot_generation
                )
        except Exception:
            self.mm_decode_state.invalidate()
            if pd_state is not None:
                pd_state.invalidate()

    def has_local_uniform_verification(self, runner: Any, output: Any, batch: Any, desc: Any) -> bool:
        """Validate local request rows independently of DP/SP graph padding."""
        try:
            draft_tokens = runner.num_speculative_steps
            if type(draft_tokens) is not int or draft_tokens <= 0:
                return False
            query_len = draft_tokens + 1
            req_ids = batch.req_ids
            num_reqs = len(req_ids)
            if num_reqs == 0 or len(set(req_ids)) != num_reqs:
                return False
            arrays = (
                batch.num_scheduled_tokens,
                batch.num_draft_tokens_np,
                batch.idx_mapping_np,
                batch.prefill_len_np,
                batch.is_prefilling_np,
            )
            if any(not isinstance(array, np.ndarray) or array.shape != (num_reqs,) for array in arrays):
                return False
            indices = batch.idx_mapping_np
            if (
                not np.issubdtype(indices.dtype, np.integer)
                or len(np.unique(indices)) != num_reqs
                or np.any(indices < 0)
                or np.any(indices >= len(runner.req_states.num_computed_tokens_np))
                or any(
                    runner.req_states.req_id_to_index.get(req_id) != index for req_id, index in zip(req_ids, indices)
                )
            ):
                return False
            graph_num_tokens, graph_num_reqs = desc.num_tokens, desc.num_reqs
            return (
                self.full_graph_mode is not None
                and desc.cg_mode == self.full_graph_mode
                and desc.uniform_token_count == query_len
                and batch.num_tokens == num_reqs * query_len
                and graph_num_tokens >= batch.num_tokens
                and graph_num_tokens % query_len == 0
                and graph_num_reqs is not None
                and graph_num_reqs >= num_reqs
                and graph_num_tokens == graph_num_reqs * query_len
                and bool(np.all(batch.num_scheduled_tokens == query_len))
                and bool(np.all(batch.num_draft_tokens_np == draft_tokens))
                and all(
                    output.num_scheduled_tokens.get(req_id) == query_len
                    and len(output.scheduled_spec_decode_tokens.get(req_id, ())) == draft_tokens
                    for req_id in req_ids
                )
            )
        except Exception:
            return False

    def _record_guard(self, checks: dict[str, bool]) -> bool:
        failed = [name for name, passed in checks.items() if not passed]
        self.last_guard = {"checks": checks, "failed": failed, "snapshot_generation": self.snapshot_generation}
        self.stats["fallback_steps" if failed else "eligible_steps"] += 1
        return not failed

    def can_queue(
        self,
        runner: Any,
        output: Any,
        batch: Any,
        desc: Any,
        current_stream: Any,
        *,
        connector_ready: bool = False,
        pd_state: PDDecodeQueueState | None = None,
        pd_transfer_group_matches: bool = False,
    ) -> bool:
        """Admit CPU bounds while retaining all exact device dependencies."""
        if not self.enabled:
            return False
        self._remember_pd_state(pd_state)
        try:
            capability = self._capability(runner)
            if capability is None:
                return self._record_guard({"model_capability": False})
            if not self.has_local_uniform_verification(runner, output, batch, desc):
                return self._record_guard({"full_uniform_verification": False})
            if capability.supports(runner) is not True:
                return self._record_guard({"model_capability": False})

            req_ids, indices = batch.req_ids, batch.idx_mapping_np
            # This advanced-indexed CPU array cannot alias the pending D2H.
            computed_upper_bound = runner.req_states.num_computed_tokens_np[indices].astype(np.int64, copy=True)
            upper_seq_lens = computed_upper_bound + batch.num_scheduled_tokens
            draft_tokens = runner.num_speculative_steps
            draft_limit = min(runner.max_model_len, runner.speculator.draft_model_config.max_model_len)
            cache = runner.encoder_cache
            text_requests = (
                not runner.supports_mm_inputs
                if cache is None
                else all(req_id in cache.mm_features and not cache.mm_features[req_id] for req_id in req_ids)
            )
            mm_keys = (
                self._mm_request_keys(runner, req_ids, indices, batch.prefill_len_np)
                if self.multimodal and self.batch_is_stable_decode(output, batch)
                else None
            )
            certified_mm = mm_keys is not None and self.mm_decode_state.is_certified(mm_keys)
            if pd_state is not None:
                connector_ready = (
                    pd_transfer_group_matches
                    and not getattr(output, "has_sync_kv_loads", False)
                    and pd_state.can_queue(req_ids, indices, batch.prefill_len_np, self.snapshot_generation)
                )
            parallel, spec = runner.parallel_config, runner.speculative_config
            checks = {
                "model_capability": True,
                "full_uniform_verification": True,
                # Context parallelism changes request-row ownership. DP graph
                # padding and model-internal SP do not; last PP stages can opt in.
                "dcp1_pcp1": parallel.decode_context_parallel_size == 1 and parallel.prefill_context_parallel_size == 1,
                "fixed_mtp": (
                    spec is not None
                    and spec.method == "mtp"
                    and type(draft_tokens) is int
                    and draft_tokens > 0
                    and not spec.uses_dynamic_speculative_decoding()
                ),
                "pure_decode": (
                    not batch.has_prefill
                    and bool(np.all(batch.prefill_len_np > 0))
                    and bool(np.all(computed_upper_bound >= batch.prefill_len_np))
                    and not bool(np.any(batch.is_prefilling_np))
                    and (batch.prefill_runs_as_decode_np is None or not bool(np.any(batch.prefill_runs_as_decode_np)))
                ),
                "no_request_arrival_resume_or_finish": (
                    not output.scheduled_new_reqs
                    and not output.scheduled_cached_reqs.resumed_req_ids
                    and not output.preempted_req_ids
                    and not output.finished_req_ids
                ),
                "bounds_within_model_and_drafter": (
                    bool(np.all(computed_upper_bound >= 0))
                    and bool(np.all(upper_seq_lens <= runner.max_model_len))
                    and bool(np.all(upper_seq_lens + draft_tokens <= draft_limit))
                ),
                "no_decoder_replay": getattr(runner.model_state, "decoder_replay", None) is None,
                "text_without_prompt_embeddings": (
                    (text_requests or certified_mm)
                    and not runner.model_config.enable_prompt_embeds
                    and not runner.model_config.is_mm_prefix_lm
                    and not output.scheduled_encoder_inputs
                ),
                "no_dbo": not parallel.enable_dbo,
                # PLE/UVA offload retains its forward copy/lookup dependencies.
                # LoRA's metadata layout still requires exact host counts.
                "no_lora": runner.lora_config is None,
                "no_connectors_or_adaptive_verification": (
                    connector_ready
                    and runner.aux_output_connector is None
                    and runner.adaptive_verification is None
                    and not runner.sparse_kv_offload_manager
                    and output.ec_connector_metadata is None
                    and output.ec_manager_metadata is None
                    and not output.kv_cache_block_copies
                    and not output.has_structured_output_requests
                ),
                "recorded_same_stream_snapshot": (
                    self.snapshot_generation > 0
                    and self.snapshot_source_stream == current_stream
                    and not runner.sync_spec_pp_cpu_counts
                    and not runner._can_skip_spec_cpu_length_sync()
                ),
            }
            eligible = self._record_guard(checks)
            if eligible and not text_requests and certified_mm:
                self.multimodal_stats["queue_eligible_steps"] += 1
            return eligible
        except Exception:
            # Unsupported optional schemas keep the runner's original exact
            # synchronization path; admission never raises into model execution.
            return self._record_guard({"supported_batch_state": False})


def update_seq_lens_cpu(
    runner: Any,
    output: Any,
    req_ids: list[str],
    *,
    allow_cpu_upper_bound: bool = False,
    current_stream: Any = None,
    controller: DecodeQueueController | None = None,
) -> None:
    """Consume corrected host counts or enqueue their device dependency.

    The original exact producer runs for every speculative step. An admitted
    batch may use the scheduler's owned CPU upper bound, without touching the
    pending pinned snapshot or its NumPy view. The device wait still protects
    that snapshot's read-before-next-write dependency on request device state.
    """
    if (
        runner.speculator is not None or runner.sync_spec_pp_cpu_counts
    ) and not runner._can_skip_spec_cpu_length_sync():
        if allow_cpu_upper_bound and controller is not None and current_stream is not None:
            current_stream.wait_event(runner.num_computed_tokens_event)
            controller.stats["host_length_wait_skipped"] += 1
            controller.stats["length_device_wait_enqueued"] += 1
        else:
            runner.num_computed_tokens_event.synchronize()
            for req_id in output.scheduled_cached_reqs.req_ids:
                index = runner.req_states.req_id_to_index[req_id]
                runner.req_states.num_computed_tokens_cpu[index] = runner.num_computed_tokens_cpu[index]

    # Exact GPU positions, lengths, KV slots and accepted recurrent state keep
    # their existing device path; this CPU field is only a backend upper bound.
    for row, req_id in enumerate(req_ids):
        index = runner.req_states.req_id_to_index[req_id]
        computed = runner.req_states.num_computed_tokens_cpu[index]
        runner.input_buffers.seq_lens_cpu[row] = computed + output.num_scheduled_tokens[req_id]
