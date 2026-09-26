# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/v1/core/sched/scheduler.py
# at vLLM v0.25.1.
import itertools

# >>> PREFLOW: work-based ranking and aging import >>>
import math

# <<< PREFLOW <<<
import time
from collections import defaultdict, deque
from collections.abc import Iterable

# >>> PREFLOW: dataclass and field added to vanilla replace import >>>
from dataclasses import dataclass, field, replace

# <<< PREFLOW <<<
from typing import Any

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import KVEventsConfig, VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

# >>> PREFLOW: work accounting shared by request ranking and aging >>>
_PREFLOW_MIN_WORK = 1e-12
_PREFLOW_LOG_2 = math.log(2.0)
_PREFLOW_BATCH_ID_ATTR = "_vllm_ascend_preflow_batch_id"


@dataclass
class _PREFLOWBatchWork:
    """PREFLOW age metadata captured before a scheduled batch advances."""

    total_work: float
    required_work_by_req_id: dict[str, float]


@dataclass
class _PREFLOWPrefillBatchState:
    """Prefill selections made during one scheduler step."""

    micro_prefill_tokens_scheduled: int = 0
    micro_prefill_req_ids: set[str] = field(default_factory=set)
    regular_prefill_req_ids: set[str] = field(default_factory=set)

    @property
    def has_scheduled_prefill(self) -> bool:
        """Return whether this scheduler step already contains a prefill."""
        return bool(self.micro_prefill_req_ids or self.regular_prefill_req_ids)

    @property
    def regular_prefill_scheduled(self) -> bool:
        """Return whether this scheduler step contains a regular prefill."""
        return bool(self.regular_prefill_req_ids)

    def add(
        self,
        request_id: str,
        num_prefill_tokens: int,
        is_micro_prefill: bool,
    ) -> None:
        """Record a prefill selected for the current scheduler step."""
        assert num_prefill_tokens > 0
        assert request_id not in self.micro_prefill_req_ids
        assert request_id not in self.regular_prefill_req_ids
        if is_micro_prefill:
            self.micro_prefill_tokens_scheduled += num_prefill_tokens
            self.micro_prefill_req_ids.add(request_id)
        else:
            self.regular_prefill_req_ids.add(request_id)


@dataclass
class _PREFLOWWaitingBatchState:
    """Mutable scheduling state shared while traversing waiting requests."""

    scheduled_new_reqs: list[Request]
    scheduled_resumed_reqs: list[Request]
    scheduled_running_reqs: list[Request]
    req_to_new_blocks: dict[str, KVCacheBlocks]
    num_scheduled_tokens: dict[str, int]
    preflow_scheduled_chunks: list[tuple[str, int, int]]
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    scheduled_loras: set[int]
    step_skipped_waiting: RequestQueue
    token_budget: int
    encoder_compute_budget: int
    prefill_scheduled: bool
    prefill_batch_state: _PREFLOWPrefillBatchState
    scheduled_timestamp: float
    defer_prefills: bool


# <<< PREFLOW <<<


# The class below is based on the upstream Scheduler. Sections added or
# altered for PREFLOW are marked with ">>> PREFLOW" comments.
# >>> PREFLOW: custom scheduler class name >>>
class PREFLOWScheduler(SchedulerInterface):
    """vLLM scheduler with PREFLOW ordering and prefill batching."""

    # <<< PREFLOW <<<

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = defaultdict(set) if include_finished_set else None
        # Track requests scheduled in prior step (MRV1-only).
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens is not None
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = self.kv_events_config is not None and self.kv_events_config.enable_kv_cache_events
        # Diffusion models may not sample any tokens for a denoising step.
        self.num_sampled_tokens_per_step = 1 if not vllm_config.model_config.is_diffusion else 0

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.defer_block_free = False
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if kv_transfer_config is not None:
            assert not self.is_encoder_decoder, "Encoder-decoder models are not currently supported with KV connectors"
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = kv_transfer_config.kv_load_failure_policy
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

            # With overlapping batches (async scheduling or PP), a step may
            # still be writing a freed request's KV blocks. A consumer KV
            # Connector can reallocate and fill those blocks via a load that
            # isn't ordered against that write, so defer freeing them.
            multiple_inflight_batches = self.vllm_config.max_concurrent_batches > 1
            if multiple_inflight_batches and kv_transfer_config.is_kv_consumer:
                self.defer_block_free = True

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(f"Unknown scheduling policy: {self.scheduler_config.policy}") from e
        # >>> PREFLOW: WSRJF relies on stable FCFS queue semantics >>>
        if self.policy != SchedulingPolicy.FCFS:
            raise ValueError(
                "PREFLOW requires scheduler_config.policy='fcfs' because "
                "it relies on vLLM's FCFS waiting-queue semantics."
            )
        # <<< PREFLOW <<<
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # IDs of requests preempted since the last call to schedule().
        self.reset_preempted_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(vllm_config.model_config)
        mm_budget = MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = mm_budget.encoder_compute_budget if mm_budget else 0
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.num_lookahead_tokens = 0
        self.dynamic_sd_lookup: list[int] | None = None
        if speculative_config is not None:
            if speculative_config.num_speculative_tokens_per_batch_size:
                self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
                    speculative_config.num_speculative_tokens_per_batch_size,
                    vllm_max_batch_size=self.scheduler_config.max_num_seqs,
                    vllm_num_speculative_tokens=self.num_spec_tokens,
                )
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.use_dflash():
                # DFlash requires an extra lookahead slot since it uses in-fill-style
                # decoding instead of standard next-token sampling, so it has a query
                # for the last sampled token plus queries for each draft token.
                self.num_lookahead_tokens = self.num_spec_tokens + 1
            if speculative_config.use_dspark():
                # DSpark drafts a block of num_spec_tokens query tokens in which the
                # anchor itself is the first prediction position (no separate bonus
                # query), so it needs exactly num_spec_tokens lookahead slots.
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        if hash_block_size is None:
            hash_block_size = block_size
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            scheduler_block_size=self.block_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.kv_metrics_collector,
            watermark=self.scheduler_config.watermark,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = vllm_config.use_v2_model_runner
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).
        self.current_step = 0
        # DP prefill balancing: Flag to track whether the last cadence-aligned
        # prefill batch fully drained the waiting queue. Prefill throttling
        # is disabled in this case.
        self.prefill_capacity_bound = False
        self.scheduler_reserve_full_isl = self.scheduler_config.scheduler_reserve_full_isl

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"

        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()

        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = vllm_config.model_config.enable_return_routed_experts

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}

        self._pause_state: PauseState = PauseState.UNPAUSED

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()

        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config

        # >>> PREFLOW: scheduler configuration and per-request state >>>
        init_ascend_config(vllm_config)
        ascend_scheduler_config = get_ascend_config().scheduler_config
        preflow_config = ascend_scheduler_config.preflow_config
        self.preflow_age_priority_double = preflow_config.age_priority_double
        self.preflow_micro_prefill_isl_threshold = preflow_config.micro_prefill_isl_threshold
        self._preflow_validate_config()

        # PREFLOW per-request state. P is fixed from the prefix-cache state
        # observed when the request enters the scheduler. Admission still
        # performs the normal cache lookup/attachment for execution, but it
        # does not rewrite this initial scoring baseline.
        self._preflow_initial_history: dict[str, int] = {}
        self._preflow_age: dict[str, float] = {}
        self._preflow_next_batch_id: int = 0
        self._preflow_pending_batch_work: dict[int, _PREFLOWBatchWork] = {}
        # <<< PREFLOW <<<

    # >>> PREFLOW: ranking, aging, batching, and admission helpers >>>
    def _preflow_validate_config(self) -> None:
        """Validate PREFLOW values consumed by the scheduler."""
        if not math.isfinite(self.preflow_age_priority_double) or self.preflow_age_priority_double <= 0:
            raise ValueError(
                f"PREFLOW requires finite age_priority_double > 0, got {self.preflow_age_priority_double}."
            )
        if self.preflow_micro_prefill_isl_threshold < 0:
            raise ValueError(
                f"PREFLOW requires micro_prefill_isl_threshold >= 0, got {self.preflow_micro_prefill_isl_threshold}."
            )

    def _preflow_work(self, num_tokens: int) -> float:
        """Return the triangular attention-work proxy for a token length."""
        tokens = float(max(0, int(num_tokens)))
        return tokens * (tokens + 1.0) / 2.0

    def _preflow_prompt_history(
        self,
        request: Request,
        num_computed_tokens: int | None = None,
    ) -> int:
        """Clamp a request's computed prompt history to its prompt length."""
        computed_tokens = request.num_computed_tokens if num_computed_tokens is None else num_computed_tokens
        return max(0, min(int(computed_tokens), request.num_prompt_tokens))

    def _preflow_get_initial_history(
        self,
        request: Request,
    ) -> int:
        """Return the frozen initial cache history used for PREFLOW scoring."""
        request_id = request.request_id
        self._preflow_age.setdefault(request_id, 0.0)
        return self._preflow_initial_history.setdefault(request_id, 0)

    def _preflow_estimate_initial_history(self, request: Request) -> int:
        """Estimate cached prompt history without attaching cache blocks."""
        prompt_history = self._preflow_prompt_history(request)
        if prompt_history > 0:
            return prompt_history
        if (
            not getattr(self.kv_cache_manager, "enable_caching", False)
            or getattr(request, "skip_reading_prefix_cache", False)
            or request.num_tokens <= 1
        ):
            return 0

        max_cache_hit_length = request.num_tokens - 1
        if (
            self.connector is not None
            and getattr(self, "has_mamba_layers", False)
            and isinstance(
                self.kv_cache_manager.coordinator,
                HybridKVCacheCoordinator,
            )
        ):
            _, per_group_hits = self.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
                request.block_hashes,
                max_cache_hit_length,
            )
            initial_history = max(per_group_hits, default=0)
        else:
            _, initial_history = self.kv_cache_manager.coordinator.find_longest_cache_hit(
                request.block_hashes,
                max_cache_hit_length,
            )
        return self._preflow_prompt_history(request, initial_history)

    def _preflow_total_required_work(
        self,
        request: Request,
    ) -> float:
        """Return total prompt work after the request's initial cache hit."""
        prompt_tokens = request.num_prompt_tokens
        initial_history = self._preflow_get_initial_history(request)
        return max(
            0.0,
            self._preflow_work(prompt_tokens) - self._preflow_work(initial_history),
        )

    def _preflow_remaining_work(self, request: Request) -> float:
        """Return triangular prompt work remaining at the current history."""
        # The initial-work baseline is not reset by preemption. After vLLM
        # discards computed KV, R_q may be greater than P_q because R_q
        # includes recomputation while P_q remains the original isolated work.
        prompt_tokens = request.num_prompt_tokens
        history = self._preflow_prompt_history(request)
        remaining_work = self._preflow_work(prompt_tokens) - self._preflow_work(history)
        return max(0.0, remaining_work)

    def _preflow_has_unfinished_prefill(self, request: Request) -> bool:
        """Return whether a request still has prompt tokens to compute."""
        return self._preflow_prompt_history(request) < request.num_prompt_tokens

    def _preflow_remaining_prefill_tokens(
        self,
        request: Request,
        num_computed_tokens: int | None = None,
    ) -> int:
        """Return the request's remaining uncached prompt-token count."""
        prompt_history = self._preflow_prompt_history(
            request,
            num_computed_tokens,
        )
        return request.num_prompt_tokens - prompt_history

    def _preflow_is_micro_prefill(
        self,
        remaining_prefill_tokens: int,
    ) -> bool:
        """Return whether a remaining prompt qualifies as a micro-prefill."""
        return (
            self.preflow_micro_prefill_isl_threshold > 0
            and 0 < remaining_prefill_tokens <= self.preflow_micro_prefill_isl_threshold
        )

    def _preflow_micro_prefill_batch_token_limit(self) -> int:
        """Return the aggregate token limit for a micro-prefill batch."""
        prefill_chunk_size = self.scheduler_config.long_prefill_token_threshold
        if prefill_chunk_size > 0:
            return prefill_chunk_size
        return self.max_num_scheduled_tokens

    def _preflow_can_schedule_prefill(
        self,
        remaining_prefill_tokens: int,
        num_new_tokens: int,
        batch_state: _PREFLOWPrefillBatchState,
    ) -> bool:
        """Return whether a prefill can join the current scheduler step."""
        # Classify by the uncached prompt length, not the scheduled chunk:
        # token-budget and chunk-size caps must not turn a large remaining ISL
        # into a micro-prefill. A genuinely small final remainder does qualify.
        # The aggregate limit counts the work actually added.
        num_prefill_tokens = min(
            remaining_prefill_tokens,
            num_new_tokens,
        )
        if num_prefill_tokens <= 0:
            return True
        if not batch_state.has_scheduled_prefill:
            return True
        if batch_state.regular_prefill_scheduled:
            return False
        if not self._preflow_is_micro_prefill(remaining_prefill_tokens):
            return False
        return (
            batch_state.micro_prefill_tokens_scheduled + num_prefill_tokens
            <= self._preflow_micro_prefill_batch_token_limit()
        )

    def _preflow_record_scheduled_prefill(
        self,
        request_id: str,
        remaining_prefill_tokens: int,
        num_new_tokens: int,
        batch_state: _PREFLOWPrefillBatchState,
    ) -> None:
        """Record scheduled prefill tokens in the step's batching state."""
        num_prefill_tokens = min(
            remaining_prefill_tokens,
            num_new_tokens,
        )
        if num_prefill_tokens <= 0:
            return
        batch_state.add(
            request_id,
            num_prefill_tokens,
            self._preflow_is_micro_prefill(remaining_prefill_tokens),
        )

    def _preflow_priority(self, request: Request) -> float | None:
        """Return log-priority for exponential aged WSRJF ordering.

        PREFLOW uses ``2 ** (A_q / age_priority_double) / (P_q R_q)``.
        Returning its logarithm preserves the ordering while avoiding overflow
        when long overload experiments produce large normalized ages.
        """
        remaining_work = self._preflow_remaining_work(request)
        if remaining_work <= _PREFLOW_MIN_WORK:
            return None
        required_work = self._preflow_total_required_work(request)
        if required_work <= _PREFLOW_MIN_WORK:
            return None
        age = self._preflow_age.get(request.request_id, 0.0)
        return (
            age / self.preflow_age_priority_double * _PREFLOW_LOG_2 - math.log(required_work) - math.log(remaining_work)
        )

    def _preflow_order_running_requests(self) -> None:
        """Order unfinished-prefill RUNNING requests by aged weighted-SRJF."""
        prioritized_prefills: list[tuple[Request, float]] = []
        for request in self.running:
            if not self._preflow_has_unfinished_prefill(request):
                continue
            priority = self._preflow_priority(request)
            if priority is not None:
                prioritized_prefills.append((request, priority))
        if len(prioritized_prefills) <= 1:
            return

        sorted_prefills = sorted(
            prioritized_prefills,
            key=lambda item: (
                -item[1],
                item[0].arrival_time,
                item[0].request_id,
            ),
        )
        preflow_order = iter(request for request, _ in sorted_prefills)
        prefill_request_ids = {id(request) for request, _ in prioritized_prefills}
        self.running = [
            next(preflow_order) if id(request) in prefill_request_ids else request for request in self.running
        ]

    def _preflow_capture_age_targets(self) -> dict[str, float]:
        """Snapshot unfinished prefills and their fixed required work."""
        required_work_by_req_id: dict[str, float] = {}
        for request_id, request in self.requests.items():
            if request.is_finished():
                continue
            if not self._preflow_has_unfinished_prefill(request):
                continue
            required_work_by_req_id[request_id] = self._preflow_total_required_work(request)
        return required_work_by_req_id

    def _preflow_add_scheduled_chunk(
        self,
        scheduled_chunks: list[tuple[str, int, int]],
        request: Request,
        history: int,
        num_new_tokens: int,
    ) -> None:
        """Record the prompt portion of a scheduled chunk for work aging."""
        prompt_tokens = request.num_prompt_tokens
        prompt_history = self._preflow_prompt_history(request, history)
        prefill_chunk = min(num_new_tokens, prompt_tokens - prompt_history)
        if prefill_chunk <= 0:
            return
        scheduled_chunks.append((request.request_id, prompt_history, prefill_chunk))

    def _preflow_remember_batch_work(
        self,
        scheduler_output: SchedulerOutput,
        scheduled_chunks: list[tuple[str, int, int]],
        required_work_by_req_id: dict[str, float],
    ) -> None:
        """Attach dispatched prefill work to an output for later aging."""
        if not scheduled_chunks:
            return
        total_work = 0.0
        for _, history, chunk_size in scheduled_chunks:
            total_work += self._preflow_work(history + chunk_size) - (self._preflow_work(history))
        if total_work <= _PREFLOW_MIN_WORK:
            return
        batch_id = self._preflow_next_batch_id
        self._preflow_next_batch_id += 1
        setattr(scheduler_output, _PREFLOW_BATCH_ID_ATTR, batch_id)
        self._preflow_pending_batch_work[batch_id] = _PREFLOWBatchWork(
            total_work=total_work,
            required_work_by_req_id=required_work_by_req_id,
        )

    def _preflow_apply_batch_age(self, scheduler_output: SchedulerOutput) -> None:
        """Age unfinished prefills using work from a completed batch."""
        batch_id = getattr(scheduler_output, _PREFLOW_BATCH_ID_ATTR, None)
        if batch_id is None:
            return
        batch_work = self._preflow_pending_batch_work.pop(
            batch_id,
            None,
        )
        delattr(scheduler_output, _PREFLOW_BATCH_ID_ATTR)
        if batch_work is None:
            return
        for request_id, required_work in batch_work.required_work_by_req_id.items():
            if required_work <= _PREFLOW_MIN_WORK:
                continue
            request = self.requests.get(request_id)
            if request is None or request.is_finished():
                continue
            self._preflow_age[request_id] = (
                self._preflow_age.get(
                    request_id,
                    0.0,
                )
                + batch_work.total_work / required_work
            )

    def _preflow_register_request(self, request: Request) -> None:
        """Initialize scoring state for a request entering the scheduler."""
        self._preflow_age[request.request_id] = 0.0
        self._preflow_initial_history[request.request_id] = self._preflow_estimate_initial_history(request)

    def _preflow_forget_request(self, request_id: str) -> None:
        """Drop PREFLOW state for a permanently finished request.

        This is intentionally idempotent because upstream cleanup can enter via
        _free_request() or direct connector-delayed _free_blocks() paths.
        """
        self._preflow_age.pop(request_id, None)
        self._preflow_initial_history.pop(request_id, None)
        for batch_work in self._preflow_pending_batch_work.values():
            batch_work.required_work_by_req_id.pop(request_id, None)

    def _preflow_effective_waiting_order(self) -> list[Request]:
        """Return FCFS waiting order using upstream's two-queue convention."""
        assert self.policy == SchedulingPolicy.FCFS
        return list(itertools.chain(self.skipped_waiting, self.waiting))

    def _preflow_wsrjf_waiting_order(
        self,
        blocked_request_ids: set[str],
    ) -> list[Request]:
        """Return waiting order with unfinished prefills ranked by WSRJF.

        This policy moves PREFLOW's aged WSRJF choice to admission while still
        preserving vLLM's non-prefill positional slots. The selected request is
        admitted through the normal copied vLLM admission routine; this changes
        traversal order, not feasibility.
        """
        waiting_order = self._preflow_effective_waiting_order()
        prioritized_prefills: list[tuple[Request, float]] = []
        for request in waiting_order:
            if request.request_id in blocked_request_ids:
                continue
            if not self._preflow_has_unfinished_prefill(request):
                continue
            priority = self._preflow_priority(request)
            if priority is not None:
                prioritized_prefills.append((request, priority))

        if len(prioritized_prefills) <= 1:
            return [request for request in waiting_order if request.request_id not in blocked_request_ids]

        sorted_prefills = sorted(
            prioritized_prefills,
            key=lambda item: (
                -item[1],
                item[0].arrival_time,
                item[0].request_id,
            ),
        )
        preflow_order = iter(request for request, _ in sorted_prefills)
        prefill_request_ids = {id(request) for request, _ in prioritized_prefills}
        return [
            next(preflow_order) if id(request) in prefill_request_ids else request
            for request in waiting_order
            if request.request_id not in blocked_request_ids
        ]

    def _preflow_next_wsrjf_waiting_request(
        self,
        blocked_request_ids: set[str],
    ) -> tuple[RequestQueue, Request] | None:
        """Return the highest-ranked feasible waiting-queue candidate."""
        for request in self._preflow_wsrjf_waiting_order(blocked_request_ids):
            request_queue = self._preflow_waiting_queue_for_request(request)
            if request_queue is not None:
                return request_queue, request
        return None

    def _preflow_waiting_queue_for_request(
        self,
        request: Request,
    ) -> RequestQueue | None:
        """Locate the upstream waiting queue that owns a request."""
        for request_queue in (self.skipped_waiting, self.waiting):
            if any(queued is request for queued in request_queue):
                return request_queue
        return None

    @staticmethod
    def _preflow_take_request_from_queue(
        request_queue: RequestQueue,
        request: Request,
    ) -> Request:
        """Remove and return a request from its current waiting queue."""
        try:
            if request_queue.peek_request() is request:
                return request_queue.pop_request()
        except IndexError:
            pass
        request_queue.remove_request(request)
        return request

    def _preflow_skip_waiting_request(
        self,
        request_queue: RequestQueue,
        request: Request,
        state: _PREFLOWWaitingBatchState,
    ) -> str:
        """Move an unschedulable candidate into this step's skipped queue."""
        request = self._preflow_take_request_from_queue(request_queue, request)
        state.step_skipped_waiting.prepend_request(request)
        return "skipped"

    def _preflow_can_admit_full_sequence(
        self,
        request: Request,
        num_new_computed_tokens: int,
        new_computed_blocks: KVCacheBlocks,
        num_external_computed_tokens: int,
        num_encoder_tokens: int,
        reserved_blocks: int,
        has_scheduled_reqs: bool,
    ) -> bool:
        """Check full-ISL capacity without mutating KV-cache state.

        vLLM's full-sequence admission check considers the candidate in
        isolation. PREFLOW also reserves the blocks that resident prefills need
        to finish, preventing later chunks from having to preempt them.
        """
        new_computed_block_list = new_computed_blocks.blocks
        num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )

        watermark_blocks = 0
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.kv_cache_manager.watermark_blocks

        full_num_tokens = min(request.num_tokens, self.max_model_len)
        num_blocks_to_allocate = self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens,
            num_local_computed_tokens=num_local_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )
        available_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        return required_blocks <= available_blocks

    def _preflow_schedule_waiting_request(
        self,
        request_queue: RequestQueue,
        request: Request,
        state: _PREFLOWWaitingBatchState,
    ) -> str:
        """Apply vanilla admission checks to one WSRJF-selected request."""
        request_id = request.request_id

        # try to promote blocked statuses while traversing skipped queue.
        if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(request):
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                logger.debug(
                    "%s is still in WAITING_FOR_REMOTE_KVS state.",
                    request_id,
                )
            return self._preflow_skip_waiting_request(
                request_queue,
                request,
                state,
            )

        # Check that adding the request still respects the max_loras
        # constraint.
        if (
            self.lora_config
            and request.lora_request
            and (
                len(state.scheduled_loras) == self.lora_config.max_loras
                and request.lora_request.lora_int_id not in state.scheduled_loras
            )
        ):
            # Scheduling would exceed max_loras, skip.
            return self._preflow_skip_waiting_request(
                request_queue,
                request,
                state,
            )

        num_external_computed_tokens = 0
        load_kv_async = False
        connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
        num_uncached_common_prefix_tokens = 0

        # Get already-cached tokens.
        if request.num_computed_tokens == 0:
            # Get locally-cached tokens.
            if (
                self.connector is not None
                and self.has_mamba_layers
                and isinstance(
                    self.kv_cache_manager.coordinator,
                    HybridKVCacheCoordinator,
                )
            ):
                computed, per_group_hits = self.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
                    request.block_hashes,
                    request.num_tokens - 1,
                )
                new_computed_blocks = self.kv_cache_manager.create_kv_cache_blocks(computed)
                # NOTE(ZhanqiuHu): For Mamba hybrid models,
                # num_new_local_computed_tokens should be the FA hit
                # length. This value is passed to the connector's
                # get_num_new_matched_tokens which computes:
                # external = total - local_computed.
                # Using the FA hit skips re-transferring FA blocks
                # already cached on D-side. The Mamba state (always
                # the last block) is transferred unconditionally by
                # _apply_prefix_caching in nixl/worker.py.
                num_new_local_computed_tokens = max(per_group_hits)
                if self.kv_cache_manager.log_stats:
                    assert self.kv_cache_manager.prefix_cache_stats is not None
                    self.kv_cache_manager.prefix_cache_stats.record(
                        num_tokens=request.num_tokens,
                        num_hits=num_new_local_computed_tokens,
                        preempted=request.num_preemptions > 0,
                    )
            else:
                (
                    new_computed_blocks,
                    num_new_local_computed_tokens,
                    request.shared_prefix_boundary,
                ) = self.kv_cache_manager.get_computed_blocks(request)

            # In case of hybrid models, obtain hint for Marconi-style APC logic
            if self.has_mamba_layers:
                num_uncached_common_prefix_tokens = getattr(
                    self.kv_cache_manager.coordinator,
                    "num_uncached_common_prefix_tokens",
                    0,
                )

            # Get externally-cached tokens if using a KVConnector.
            if self.connector is not None:
                ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                    request,
                    num_new_local_computed_tokens,
                )

                if ext_tokens is None:
                    # The request cannot be scheduled because
                    # the KVConnector couldn't determine
                    # the number of matched tokens.
                    return self._preflow_skip_waiting_request(
                        request_queue,
                        request,
                        state,
                    )

                num_external_computed_tokens = ext_tokens

                connector_prefix_cache_queries = request.num_tokens - num_new_local_computed_tokens
                connector_prefix_cache_hits = num_external_computed_tokens

            # Total computed tokens (local + external).
            num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
            assert num_computed_tokens <= request.num_tokens

            # Skip request with pending mm encoding prefetches
            if (
                self.ec_connector is not None
                and request.mm_features
                and not self.ec_connector.ensure_cache_available(
                    request,
                    num_computed_tokens,
                )
            ):
                return self._preflow_skip_waiting_request(
                    request_queue,
                    request,
                    state,
                )

            # Track first scheduled prefill, not post-preemption repeat prefills
            if request.prefill_stats is not None:
                assert num_computed_tokens <= request.num_prompt_tokens
                request.prefill_stats.set(
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_local_cached_tokens=num_new_local_computed_tokens,
                    num_external_cached_tokens=num_external_computed_tokens,
                )
        else:
            # KVTransfer: WAITING reqs have num_computed_tokens > 0
            # after async KV recvs are completed.
            new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
            num_new_local_computed_tokens = 0
            num_computed_tokens = request.num_computed_tokens

        encoder_inputs_to_schedule = None
        external_load_encoder_input = []
        new_encoder_compute_budget = state.encoder_compute_budget
        pad_spec_decode = False

        if load_kv_async:
            # KVTransfer: loading remote KV, do not allocate for new work.
            assert num_external_computed_tokens > 0
            num_new_tokens = 0
        elif state.defer_prefills and num_computed_tokens < request.num_tokens - 1:
            # DP prefill balancing: defer this step's local prefill
            # compute to a cadence-aligned step.
            return "blocked"
        else:
            # Number of tokens to be scheduled.
            # We use `request.num_tokens` instead of
            # `request.num_prompt_tokens` to consider the resumed
            # requests, which have output tokens.
            num_new_tokens = request.num_tokens - num_computed_tokens

            # Pad new decode requests to uniform spec decoding size to
            # preserve full cudagraph for this step.
            if (
                (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None)
                and num_new_tokens == 1
                and (state.scheduled_running_reqs and not state.prefill_scheduled)
            ):
                num_new_tokens = 1 + self.num_spec_tokens
                if num_new_tokens > state.token_budget or num_computed_tokens + num_new_tokens > self.max_model_len:
                    # Prefer to not schedule than schedule un-padded here.
                    return "blocked"
                pad_spec_decode = True

            threshold = self.scheduler_config.long_prefill_token_threshold
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold

            # chunked prefill has to be enabled explicitly to allow
            # pooling requests to be chunked
            if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > state.token_budget:
                # If chunked_prefill is disabled,
                # we can stop the scheduling here.
                return "blocked"

            num_new_tokens = min(num_new_tokens, state.token_budget)
            assert num_new_tokens > 0

            # Schedule encoder inputs.
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    num_computed_tokens,
                    num_new_tokens,
                    state.encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )
                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    return "blocked"

        # Skip block alignment when setting up async receive (no local work).
        if self.need_mamba_block_aligned_split and not load_kv_async:
            num_new_tokens = self._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
                num_uncached_common_prefix_tokens,
            )
            if num_new_tokens == 0:
                return "blocked"

        # This is evaluated after the real admission-time cache lookup, so a
        # prefix hit reduces the ISL and an evicted prefix does not.
        remaining_prefill_tokens = self._preflow_remaining_prefill_tokens(
            request,
            num_computed_tokens,
        )
        if not self._preflow_can_schedule_prefill(
            remaining_prefill_tokens,
            num_new_tokens,
            state.prefill_batch_state,
        ):
            return "blocked"

        # During async KV load, no forward pass is run yet.
        # Allocate speculative lookahead slots later to avoid
        # mismatching local and remote block counts.
        limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
        effective_lookahead_tokens = 0 if limit_lookahead_tokens else self.num_lookahead_tokens

        # Determine if we need to allocate cross-attention blocks.
        num_encoder_tokens = 0
        if self.is_encoder_decoder and request.has_encoder_inputs and encoder_inputs_to_schedule:
            num_encoder_tokens = sum(request.get_num_encoder_embeds(i) for i in encoder_inputs_to_schedule)

        reserved_blocks = 0
        if self.scheduler_reserve_full_isl or load_kv_async:
            # With full-ISL admission, leave enough unallocated capacity for
            # every resident prefill to finish. Async loads need the same
            # reservation even when full-ISL admission is disabled because
            # they hold their blocks without making forward progress.
            reserved_blocks = self._inflight_prefill_reserved_blocks()

        if self.scheduler_reserve_full_isl and not self._preflow_can_admit_full_sequence(
            request=request,
            num_new_computed_tokens=num_new_local_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_external_computed_tokens=num_external_computed_tokens,
            num_encoder_tokens=num_encoder_tokens,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=bool(self.running),
        ):
            if request.has_encoder_inputs:
                self.encoder_cache_manager.free(request)
            return "kv_blocked"

        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_local_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=effective_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=load_kv_async,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=self.scheduler_reserve_full_isl,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=bool(self.running),
        )

        if new_blocks is None:
            # The request cannot be scheduled.

            # NOTE: we need to untouch the request from the encode cache
            # manager
            if request.has_encoder_inputs:
                self.encoder_cache_manager.free(request)
            return "kv_blocked"

        # KVTransfer: the connector uses this info to determine
        # if a load is needed. Note that
        # This information is used to determine if a load is
        # needed for this request.
        if self.connector is not None:
            self.connector.update_state_after_alloc(
                request,
                self.kv_cache_manager.get_blocks(request_id),
                num_external_computed_tokens,
            )
            if self.connector_prefix_cache_stats is not None and connector_prefix_cache_queries != 0:
                self.connector_prefix_cache_stats.record(
                    num_tokens=connector_prefix_cache_queries,
                    num_hits=connector_prefix_cache_hits,
                    preempted=request.num_preemptions > 0,
                )

        request = self._preflow_take_request_from_queue(request_queue, request)
        if load_kv_async:
            # If loading async, allocate memory and put request
            # into the WAITING_FOR_REMOTE_KV state.
            request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
            state.step_skipped_waiting.prepend_request(request)
            # Set num_computed_tokens even though KVs are not yet loaded.
            # request.num_computed_tokens will not be used anywhere until
            # the request finished the KV transfer.
            #
            # If a transfer error is reported by the connector,
            # request.num_computed_tokens will be re-set accordingly in
            # _update_requests_with_invalid_blocks.
            #
            # When the transfer is finished, either successfully or not,
            # request.num_computed_tokens will correctly reflect the number
            # of computed tokens.
            # _update_waiting_for_remote_kv will then cache
            # only the successfully loaded tokens.
            request.num_computed_tokens = num_computed_tokens
            self._inflight_prefills.add(request)
            return "scheduled"

        self.running.append(request)
        if self.log_stats:
            request.record_event(
                EngineCoreEventType.SCHEDULED,
                state.scheduled_timestamp,
            )
        if request.status == RequestStatus.WAITING:
            state.scheduled_new_reqs.append(request)
        elif request.status == RequestStatus.PREEMPTED:
            state.scheduled_resumed_reqs.append(request)
        else:
            raise RuntimeError(f"Invalid request status: {request.status}")

        if self.lora_config and request.lora_request:
            state.scheduled_loras.add(request.lora_request.lora_int_id)
        state.req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
        state.num_scheduled_tokens[request_id] = num_new_tokens
        self._preflow_add_scheduled_chunk(
            state.preflow_scheduled_chunks,
            request,
            num_computed_tokens,
            num_new_tokens,
        )
        self._preflow_record_scheduled_prefill(
            request_id,
            remaining_prefill_tokens,
            num_new_tokens,
            state.prefill_batch_state,
        )
        state.token_budget -= num_new_tokens
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens
        if pad_spec_decode:
            state.scheduled_spec_decode_tokens[request_id] = [-1] * self.num_spec_tokens
        # Only track requests that will still be prefilling after this chunk.
        if num_computed_tokens + num_new_tokens < request.num_tokens:
            self._inflight_prefills.add(request)
        # Encoder-related.
        if encoder_inputs_to_schedule:
            state.scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
            state.encoder_compute_budget = new_encoder_compute_budget
        # Allocate for external load encoder cache
        if external_load_encoder_input:
            for i in external_load_encoder_input:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
        return "scheduled"

    # <<< PREFLOW <<<

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        num_uncached_common_prefix_tokens: int = 0,
    ) -> int:
        num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif num_computed_tokens < last_cache_position < num_computed_tokens_after_sched:
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass

            # Marconi cache admission optimization:
            # cache common prefixes by scheduling num_new_tokens = common prefix length
            if num_uncached_common_prefix_tokens >= block_size and num_new_tokens > num_uncached_common_prefix_tokens:
                num_new_tokens = num_uncached_common_prefix_tokens
                # keep alignment to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
        return num_new_tokens

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        self.current_step += 1
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        # >>> PREFLOW: track dispatched triangular prefill work >>>
        preflow_scheduled_chunks: list[tuple[str, int, int]] = []
        # <<< PREFLOW <<<
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # Whether the running batch contains any prefill requests.
        prefill_scheduled = False

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()
        # >>> PREFLOW: snapshot requests that age during this batch >>>
        preflow_required_work_by_req_id = self._preflow_capture_age_targets()
        # <<< PREFLOW <<<

        # DP prefill balancing: on a throttled (non-cadence-aligned) step, defer
        # all prefill compute unless saturated.
        defer_prefills = (throttle_prefills and not self.prefill_capacity_bound) and any(
            not r.is_prefill_chunk for r in self.running
        )

        # First, schedule the RUNNING requests.
        # >>> PREFLOW: rank running prefills and share batching state with
        # waiting-request admission below >>>
        self._preflow_order_running_requests()
        prefill_batch_state = _PREFLOWPrefillBatchState()
        # <<< PREFLOW <<<
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            if self.current_step < request.next_decode_eligible_step:
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1
                continue

            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step,
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # >>> PREFLOW: enforce regular/micro-prefill co-batching >>>
            remaining_prefill_tokens = self._preflow_remaining_prefill_tokens(request)
            if not self._preflow_can_schedule_prefill(
                remaining_prefill_tokens,
                num_new_tokens,
                prefill_batch_state,
            ):
                req_index += 1
                continue
            # <<< PREFLOW <<<

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # >>> PREFLOW: full-ISL admission reserves the capacity
                    # owed to resident prefills, so never recover an allocation
                    # failure by preempting one. Disabling the setting
                    # explicitly restores vanilla FCFS preemption >>>
                    if self.scheduler_reserve_full_isl:
                        preempted_req = next(
                            (
                                candidate
                                for candidate in reversed(self.running[req_index:])
                                if not self._preflow_has_unfinished_prefill(candidate)
                            ),
                            None,
                        )
                        if preempted_req is None:
                            break
                        self.running.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()
                    # <<< PREFLOW <<<

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            prefill_scheduled |= request.is_prefill_chunk
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            # >>> PREFLOW: record work and prefill batch membership >>>
            self._preflow_add_scheduled_chunk(
                preflow_scheduled_chunks,
                request,
                request.num_computed_tokens,
                num_new_tokens,
            )
            self._preflow_record_scheduled_prefill(
                request_id,
                remaining_prefill_tokens,
                num_new_tokens,
                prefill_batch_state,
            )
            # <<< PREFLOW <<<
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            # >>> PREFLOW: WSRJF admission and shared micro-prefill batching
            # replace the vanilla FCFS waiting loop >>>
            step_skipped_waiting = create_request_queue(self.policy)
            waiting_state = _PREFLOWWaitingBatchState(
                scheduled_new_reqs=scheduled_new_reqs,
                scheduled_resumed_reqs=scheduled_resumed_reqs,
                scheduled_running_reqs=scheduled_running_reqs,
                req_to_new_blocks=req_to_new_blocks,
                num_scheduled_tokens=num_scheduled_tokens,
                preflow_scheduled_chunks=preflow_scheduled_chunks,
                scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
                scheduled_encoder_inputs=scheduled_encoder_inputs,
                scheduled_loras=scheduled_loras,
                step_skipped_waiting=step_skipped_waiting,
                token_budget=token_budget,
                encoder_compute_budget=encoder_compute_budget,
                prefill_scheduled=prefill_scheduled,
                prefill_batch_state=prefill_batch_state,
                scheduled_timestamp=scheduled_timestamp,
                defer_prefills=defer_prefills,
            )

            blocked_request_ids: set[str] = set()

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break

                next_waiting = self._preflow_next_wsrjf_waiting_request(blocked_request_ids)
                if next_waiting is None:
                    break
                request_queue, request = next_waiting
                result = self._preflow_schedule_waiting_request(
                    request_queue,
                    request,
                    waiting_state,
                )
                token_budget = waiting_state.token_budget
                encoder_compute_budget = waiting_state.encoder_compute_budget
                if result in ("scheduled", "skipped"):
                    continue
                if result in ("kv_blocked", "blocked"):
                    blocked_request_ids.add(request.request_id)
                    continue
                break

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

            # DP prefill balancing: on a step that admitted prefills (release),
            # record whether it was capacity-bound.
            if not defer_prefills:
                self.prefill_capacity_bound = bool(self.waiting)
            # <<< PREFLOW <<<

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs.extend(scheduled_resumed_reqs)
            scheduled_resumed_reqs.clear()
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step (MRV1-only).
        if not self.use_v2_model_runner:
            self.prev_step_scheduled_req_ids.clear()
            self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
        )

        # Dynamic speculative decoding: compute optimal K
        num_spec_tokens_to_schedule = self.num_spec_tokens
        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids=self.reset_preempted_req_ids,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            num_spec_tokens_to_schedule=num_spec_tokens_to_schedule,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        # Advance the fence only for non-empty steps (those that actually
        # write KV and have their output processed later in update_from_output).
        if self.defer_block_free and total_num_scheduled_tokens > 0:
            self.sched_step_seq += 1

        # >>> PREFLOW: attach work metadata for completion-time aging >>>
        self._preflow_remember_batch_work(
            scheduler_output,
            preflow_scheduled_chunks,
            preflow_required_work_by_req_id,
        )
        # <<< PREFLOW <<<
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
        drop_stale_output: bool = False,
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, "Only running requests can be preempted"
        self._free_request_blocks(request)
        self.encoder_cache_manager.free(request)
        self._inflight_prefills.discard(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        # >>> PREFLOW: preserve the frozen work baseline and age >>>
        # PREFLOW keeps normalized age and the original h_0
        # across preemption; recomputation loss is reflected in R_q instead.
        # <<< PREFLOW <<<
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.drop_stale_output = drop_stale_output or (
            request.drop_stale_output and request.num_stale_output_tokens > 0
        )
        request.num_stale_output_tokens = request.num_in_flight_tokens
        request.num_output_placeholders = 0
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.num_in_flight_tokens += num_scheduled_token
            if self.defer_block_free:
                # Record the in-flight step, to fence deferred block freeing.
                request.last_sched_seq = self.sched_step_seq
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)

        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid
            self._re_block_ids.update(
                {rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid] for rid in num_scheduled_tokens}
            )

        # Clear the finished and preempted request IDs.
        # NOTE: We shouldn't just clear() here because it will also affect
        # the scheduler output.
        self.finished_req_ids = set()
        self.reset_preempted_req_ids = set()

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[session.num_prompt_tokens : num_computed_tokens]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(mm_feature.mm_position, offset=mm_feature.mm_position.offset + base)
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING
        # >>> PREFLOW: reset scoring state for the new stream chunk >>>
        self._preflow_register_request(session)
        # <<< PREFLOW <<<

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(spec_decode_tokens.get(req_id, ()))
                token_ids = req.all_token_ids[req.num_computed_tokens : req.num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            if idx >= num_running_reqs:
                resumed_req_ids.add(req_id)
            if not self.use_v2_model_runner:  # noqa: SIM102
                if req_id not in self.prev_step_scheduled_req_ids:
                    all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(req_to_new_blocks[req_id].get_block_ids(allow_none=True))
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(req.num_output_tokens + req.num_output_placeholders)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0

        lo, hi = get_mm_features_in_window(
            mm_features,
            start=num_computed_tokens,
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,
        )
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0

        for i in range(lo, hi):
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens) < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(0, start_pos - (num_computed_tokens + shift_computed_tokens))
                break
            if not self.encoder_cache_manager.can_allocate(request, i, encoder_compute_budget, num_embeds_to_schedule):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (num_computed_tokens + shift_computed_tokens)
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos)
            curr_embeds_start, curr_embeds_end = mm_feature.mm_position.get_embeds_indices_in_range(
                start_idx_rel, end_idx_rel
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(item_identifier):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(self, scheduler_output: SchedulerOutput) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id)) and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # >>> PREFLOW: age requests by the completed batch's work >>>
        self._preflow_apply_batch_age(scheduler_output)
        # <<< PREFLOW <<<
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        # Every GPU write enqueued by this and earlier steps has completed, so it is
        # safe to return deferred-free blocks to the pool.
        if self.defer_block_free and scheduler_output.total_num_scheduled_tokens > 0:
            self.processed_step_seq += 1
            self._drain_deferred_frees()

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None
        routing_offsets: dict[str, int] = {}
        if model_runner_output.routed_experts is not None:
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
            routing_data = re.routing_data.astype(
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:
                routing_offsets[rid] = offset
                offset += num_scheduled_tokens[rid]

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            output_is_stale = False
            if request is not None:
                request.num_in_flight_tokens -= num_tokens_scheduled
                if request.num_stale_output_tokens > 0:
                    output_is_stale = True
                    request.num_stale_output_tokens -= num_tokens_scheduled
                    assert request.num_stale_output_tokens >= 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            if output_is_stale and request.drop_stale_output:
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if scheduled_spec_token_ids and (generated_token_ids or self.num_sampled_tokens_per_step == 0):
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                num_rejected = num_draft_tokens - num_accepted
                if not output_is_stale:
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            # Free encoder inputs only after the step has actually executed.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            ec_transfer_params = None
            status_before_stop = request.status
            num_output_tokens_before = len(request._output_token_ids)

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request,
                    new_token_ids,
                    is_stale=output_is_stale,
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar
                assert grammar is not None
                # new_token_ids can be a mixed block of reasoning content, then
                # the reasoning end marker, then the start of the grammar content.
                # Trim the reasoning content so the grammar only sees grammar content.
                advance_token_ids = self.structured_output_manager.trim_reasoning_for_advance(request, new_token_ids)
                if advance_token_ids and not grammar.accept_tokens(req_id, advance_token_ids):
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. Terminating request.",
                        advance_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR
                    request.resumable = False
                    stopped = True

            routed_experts = None
            if self.enable_return_routed_experts and routing_data is not None and new_token_ids:
                req_offset = routing_offsets[req_id]
                end = req_offset + num_tokens_scheduled
                block_ids = self._re_block_ids.pop(req_id, [])
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start is not None
                    ):
                        prompt_start = request.sampling_params.routed_experts_prompt_start
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(
                        block_ids,
                        request.num_prompt_tokens,
                        token_start=prompt_start,
                    )
                else:
                    if scheduled_spec_token_ids:
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[req_offset : req_offset + len(new_token_ids)]
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]

            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params, ec_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None and request.sampling_params.num_logprobs is not None and logprobs:
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            should_emit_output = bool(new_token_ids or pooler_output is not None or stopped)
            if should_emit_output:
                prefill_stats = request.take_prefill_stats()
                if prefill_stats is not None:
                    prefill_stats.finalize(self.kv_cache_manager.estimate_cached_tokens(request))

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        prefill_stats=prefill_stats,
                        kv_transfer_params=kv_transfer_params,
                        ec_transfer_params=ec_transfer_params,
                        trace_headers=request.trace_headers,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
            self.skipped_waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # Worker-side KV connector stats from the model runner output.
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if self.connector:
            # Scheduler-side KV connector stats collected after connector update.
            scheduler_kv_connector_stats = self.connector.get_kv_connector_stats()
            if scheduler_kv_connector_stats is not None and not scheduler_kv_connector_stats.is_empty():
                kv_connector_stats = (
                    kv_connector_stats.aggregate(scheduler_kv_connector_stats)
                    if kv_connector_stats is not None
                    else scheduler_kv_connector_stats
                )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
            finished_req_ids.clear()

        if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
        is_stale: bool = False,
    ) -> tuple[list[int], bool]:
        del is_stale
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(request)
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Defer the free by the drafter's look-ahead so an entry stays
        # referenced until the drafter's +1 read has also passed it, mirroring
        # the shift the encoder scheduling path applies.
        spec_lookahead = 1 if self.use_eagle else 0

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif (
                start_pos + num_tokens + spec_lookahead <= request.num_computed_tokens - request.num_output_placeholders
            ):
                # Processed, stored in the decoder KV cache, and far enough past
                # the placeholder range (plus the drafter's look-ahead) that no
                # rejection or drafter gather can reference it.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            # >>> PREFLOW: initialize frozen work and age state >>>
            self._preflow_register_request(request)
            # <<< PREFLOW <<<
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.connector is not None:
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus) -> list[Request]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            List of requests that were aborted. Will not include any that were
            already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = request.request_id not in self.finished_recving_kv_req_ids
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return valid_requests

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert request.is_finished()

        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)

        ec_xfer_params: dict[str, Any] | None = None
        if self.ec_connector is not None:
            ec_delay_free, ec_xfer_params = self.ec_connector.request_finished(request)
            connector_delay_free_blocks |= ec_delay_free

        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        # >>> PREFLOW: release per-request scheduler state >>>
        self._preflow_forget_request(request_id)
        # <<< PREFLOW <<<
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params, ec_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        # >>> PREFLOW: idempotent delayed-free cleanup >>>
        self._preflow_forget_request(request.request_id)
        # <<< PREFLOW <<<
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        if not self.defer_block_free or (
            # Last scheduled step already processed: no in-flight write remains
            # (always the case for a normal finish), so free now.
            request.last_sched_seq <= self.processed_step_seq
        ):
            self.kv_cache_manager.free(request)
            return
        blocks = self.kv_cache_manager.pop_blocks_for_free(request)
        if blocks:
            self.deferred_frees.append((self.sched_step_seq, blocks))

    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Entries are appended with monotonically non-decreasing fences, so
        stop at the first one that is still pending.
        """
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = len(self.waiting) + len(self.skipped_waiting) - self.num_waiting_for_streaming_input
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:
            return True
        if self.connector is None:
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        return len(self.requests) > num_in_queues

    def has_requests(self) -> bool:
        # Override the interface default to also keep the engine alive while a
        # connector still has pending push work (e.g. push-mode WRITE transfers
        # in flight after all "live" requests have finished). Without this hook
        # the engine would quiesce before the connector can drain completions.
        # TODO: replace with a more general mechanism for connectors to keep
        # the scheduler alive.
        return (
            self.has_unfinished_requests()
            or self.has_finished_requests()
            or (self.connector is not None and self.connector.has_pending_push_work())
            or (self.ec_connector is not None and self.ec_connector.has_pending_push_work())
        )

    def reset_prefix_cache(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp, drop_stale_output=True)

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug("reset_connector requested but no KV connector is configured; treating as no-op success.")
            return True

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = self.kv_metrics_collector.drain_events() if self.kv_metrics_collector is not None else []
        spec_stats = spec_decoding_stats
        connector_stats_payload = kv_connector_stats.data if kv_connector_stats else None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")
        # >>> PREFLOW: discard scoring state on shutdown >>>
        self._preflow_pending_batch_work.clear()
        self._preflow_age.clear()
        self._preflow_initial_history.clear()
        # <<< PREFLOW <<<
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def get_ec_connector(self) -> ECConnectorBase | None:
        return self.ec_connector

    def get_kv_event_publisher_config(self) -> KVEventsConfig | None:
        return self.kv_event_publisher.get_publisher_config()

    def _connector_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector. In-flight tokens have not completed, so they cannot be
        # discarded yet.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(0, request.num_computed_tokens - request.num_in_flight_tokens),
            num_prompt_tokens=request.num_prompt_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
            request_id=request.request_id,
            num_computed_tokens=request.num_computed_tokens,
        )

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            num_encoder_tokens=0,
            total_computed_tokens=request.num_computed_tokens,
            num_local_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )

    def _inflight_prefill_reserved_blocks(self) -> int:
        """Num blocks in-flight prefills still need to finish (their reservation)."""

        return sum(self._request_remaining_blocks(req) for req in self._inflight_prefills)

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            f"Unexpected blocked waiting status in promotion: {request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)

            req_num_computed_blocks = (req_num_computed_tokens + self.block_size - 1) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = req_num_computed_tokens - request.num_computed_tokens
                total_affected_tokens += num_affected_tokens

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += request.num_computed_tokens - req_num_computed_tokens
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (req for req in self.skipped_waiting if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS)
        async_failed_req_ids, num_failed_tokens, _ = self._update_requests_with_invalid_blocks(
            async_load_reqs,
            invalid_block_ids,
            num_scheduled_tokens,
            evict_blocks=False,
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = self._update_requests_with_invalid_blocks(
            self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: %d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids


# >>> PREFLOW: async counterpart of the custom scheduler >>>
class AsyncPREFLOWScheduler(PREFLOWScheduler):
    """Asynchronous PREFLOW scheduler using vLLM's async update flow."""

    # <<< PREFLOW <<<

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        # Use the latest num of scheduled draft tokens in next step as placeholder.
        self._spec_token_placeholders = [-1] * scheduler_output.num_spec_tokens_to_schedule
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate num_sampled_tokens_per_step new tokens
            # plus num_spec_tokens in this scheduling step. Diffusion has no AR
            # bonus token (num_sampled_tokens_per_step == 0) - only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += self.num_sampled_tokens_per_step + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            if self.use_v2_model_runner:
                # Set the next step index in which this request is eligible to be
                # scheduled for decode (for PP microbatching).
                request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
        is_stale: bool = False,
    ) -> tuple[list[int], bool]:
        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request,
            new_token_ids,
            is_stale=is_stale,
        )

        if not is_stale:
            request.num_output_placeholders -= len(new_token_ids)
            assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders,
            )
        return new_token_ids, stopped


# >>> PREFLOW: export the custom scheduler classes >>>
__all__ = ["AsyncPREFLOWScheduler", "PREFLOWScheduler"]
# <<< PREFLOW <<<
