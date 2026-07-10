# mypy: ignore-errors
"""Balance scheduling patch.

Keeps running-request counts even across DP ranks: every step an all-gather
of each rank's ``len(running)`` runs once via ``BalanceScheduler.balance_gather``
(invoked from the engine core, NOT from inside ``schedule()``); if any rank
was at the running cap at the end of the previous step, every rank stops
admitting new WAITING requests (``any-rank-at-cap => global freeze``). Each
rank reaches that decision independently from the same gathered snapshot --
there is no leader. See ``docs/.../balance_schedule_refactor.md`` for the
design.

The ``schedule()`` body is a verbatim copy of the **v0.23.0** release tag's
``Scheduler.schedule()`` (the production pin), plus exactly three balance
deltas: (1) the disabled-path early return that delegates to ``super()``,
(2) the ``balance_flag`` break inside the WAITING loop
(``any-rank-at-cap => global freeze``), and (3) ``if request_queue is None:
break`` in place of upstream's ``assert request_queue is not None`` (so a
drained-rank schedule does not assert when balance defers admission).

The **signature**, in contrast, must work across BOTH vllm versions that
vllm-ascend CI runs simultaneously: the release tag v0.23.0 (whose engine
calls ``schedule()`` with no args) and the main-verified commit 1f486d96
(whose engine calls ``schedule(throttle_prefills)``). So the override carries
``throttle_prefills`` with a default -- a deliberate superset of v0.23.0's
``schedule(self)`` -- making it callable by both engines. On the disabled
fast-path it then forwards ``throttle_prefills`` only when the installed
``super().schedule`` actually accepts it, decided by introspecting the
signature once at import (``_SUPER_SCHEDULE_HAS_THROTTLE``) rather than
parsing a version string (a dev checkout's ``__version__`` is not a clean
PEP 440 release and would make a ``vllm_version_is`` check raise). The body
and the signature therefore deliberately target different things: the body
tracks the stable release tag, the signature tracks the union of both
engines' call shapes. See the design doc for the full rationale.

The engine-core side is NOT copied: ``BalanceDPEngineCoreProc`` hooks
``_has_global_unfinished_reqs`` (called every iteration by upstream's
``run_busy_loop`` on every non-idle path) to inject the DP group and run
``balance_gather`` once per step. The gather lives in the engine core, NOT
inside ``schedule()``: a rank that has drained its local requests never enters
``schedule()`` (it runs a dummy batch instead), so an ``all_gather`` in
``schedule()`` would be skipped by that rank while busy ranks call it -- a
collective mismatch that deadlocks.

The gather is hooked IMMEDIATELY AFTER ``super()._has_global_unfinished_reqs()``
(not inside ``_process_engine_step``). ``_has_global_unfinished_reqs`` is
itself a cross-rank collective (an all-reduce every 32 steps internally) and is
the only point in the busy loop that re-synchronizes ranks on wave/idle state.
Hooking gather right after it keeps the every-step all-gather in the same
lock-stepped region, so ranks enter the gather having just agreed on
``engines_running``. Hooking it earlier -- inside ``_process_engine_step``,
before that sync and before the idle ``continue`` gate -- decouples the gather
from the synchronization: at wave boundaries one rank can reach the gather
while another is still blocked in ``_process_input_queue`` or
``future.result()``, deadlocking the all-gather. The stuck EngineCore then
can't drain its worker shm channel, the worker's ``sample_tokens`` response
has nowhere to land, and after 60s the engine dies with
``RPC call to sample_tokens timed out``. ``_has_global_unfinished_reqs`` is
only called on iterations that did NOT take the idle ``continue``, so gather
is skipped consistently by every rank when all are idle -- no rank does an
extra gather.

The engine core class is swapped via the module-level ``DPEngineCoreProc``
reference. This works ONLY because upstream's ``run_engine_core`` resolves
that name at call time rather than binding it at import -- if upstream ever
switches to an import-time binding the swap silently stops taking effect, so
the call-time lookup is a load-bearing assumption. The swap is done ONLY when
balance scheduling is enabled (conditional activation, so balance does not
touch configs that don't use it, e.g. PD-disaggregated recompute).
"""

import inspect
import time

import torch
import torch.distributed as dist
import vllm.v1.core.sched.scheduler as _sched_mod
import vllm.v1.engine.core as _engine_core_mod
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

# Whether the *installed* upstream ``Scheduler.schedule`` accepts the
# ``throttle_prefills`` argument. vllm-ascend CI runs against TWO vllm
# versions at once: the release tag v0.23.0 (``schedule(self)``, engine calls
# ``schedule()``) and the main-verified commit 1f486d96 (``schedule(self,
# throttle_prefills=False)``, engine calls ``schedule(throttle_prefills)``).
# The override signature carries ``throttle_prefills`` (with a default) so it
# is callable by BOTH engines; on the disabled path it must then forward the
# arg only when the installed super() actually accepts it. Introspecting the
# signature once at import (rather than parsing a version string) is robust to
# both lanes -- including dev checkouts whose ``__version__`` is not a clean
# PEP 440 release (which would make a ``vllm_version_is`` check raise).
_SUPER_SCHEDULE_HAS_THROTTLE = "throttle_prefills" in inspect.signature(Scheduler.schedule).parameters


def _balance_scheduling_enabled(vllm_config) -> bool:
    # Primary source of truth is AscendConfig. The additional_config fallback
    # covers the startup window where AscendConfig may not yet be initialized
    # (see the TODO that used to live here); once AscendConfig init is moved
    # earlier it can go away.
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().enable_balance_scheduling)
    except Exception:
        pass
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if "enable_balance_scheduling" in additional_config:
        return bool(additional_config["enable_balance_scheduling"])
    return False


class BalanceScheduler(Scheduler):
    def __init__(
        self,
        vllm_config,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )
        self._balance_enabled = _balance_scheduling_enabled(vllm_config)
        # Injected by BalanceDPEngineCoreProc._has_global_unfinished_reqs
        # before the first gather. Only used on the enabled path (balance
        # requires DP > 1).
        self.dp_group = None
        if self._balance_enabled:
            self.balance_queue = [
                torch.tensor([0], dtype=torch.int, device="cpu")
                for _ in range(self.vllm_config.parallel_config.data_parallel_size)
            ]

    def balance_gather(self):
        """All-gather per-rank running counts into ``self.balance_queue``.

        Called once per busy-loop iteration from
        ``BalanceDPEngineCoreProc._has_global_unfinished_reqs`` (immediately
        after the cross-rank all-reduce, which runs after ``schedule()`` +
        execute + ``update_from_output()``). This MUST be invoked on every
        rank every non-idle iteration, including dummy-batch iterations where
        a drained rank never enters ``schedule()`` --
        ``all_gather`` is a collective, so any rank skipping it deadlocks
        the busy ranks. Returns early when balance is disabled or the DP
        group has not been injected yet.
        """
        if not self._balance_enabled or self.dp_group is None:
            return
        running_tensor = torch.tensor([len(self.running)], dtype=torch.int, device="cpu")
        dist.all_gather(self.balance_queue, running_tensor, group=self.dp_group)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        if not self._balance_enabled:
            # Forward throttle_prefills only when the installed super() accepts
            # it (main-verified); v0.23.0's super() does not. See
            # _SUPER_SCHEDULE_HAS_THROTTLE for why this is signature-based.
            if _SUPER_SCHEDULE_HAS_THROTTLE:
                return super().schedule(throttle_prefills)
            return super().schedule()
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
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests.
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

            num_new_tokens = (
                request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)

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
                #    models with mamba cache mode "align".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

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

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i) for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

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
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
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
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                # >>> balance-scheduling delta (the whole point of this patch) <<<
                # If any DP rank was at the running cap at the end of the
                # previous step, stop admitting new WAITING requests on this
                # rank too, so load stays even across ranks
                # (leader-at-cap => global freeze).
                if max(t.item() for t in self.balance_queue) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                if request_queue is None:
                    break

                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(
                    request
                ):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(
                        request
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

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
                        and not self.ec_connector.ensure_cache_available(request, num_computed_tokens)
                    ):
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue

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
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
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
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Skip block alignment when setting up async receive (no local work).
                if self.need_mamba_block_aligned_split and not load_kv_async:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                limit_lookahead_tokens = load_kv_async and self.use_eagle
                effective_lookahead_tokens = 0 if limit_lookahead_tokens else self.num_lookahead_tokens

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if self.is_encoder_decoder and request.has_encoder_inputs and encoder_inputs_to_schedule:
                    num_encoder_tokens = sum(request.get_num_encoder_embeds(i) for i in encoder_inputs_to_schedule)

                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()

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
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

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

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
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
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

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
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
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

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
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

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output


class BalanceDPEngineCoreProc(DPEngineCoreProc):
    """Minimal DP engine core hook for balance scheduling.

    The only thing balance scheduling needs from the engine core is the DP
    process group, which the scheduler uses for its per-step all-gather. The
    group is created in ``_init_data_parallel`` (during ``__init__``, before
    the scheduler exists) and the scheduler is created in ``EngineCore.__init__``,
    so both are present by the time the busy loop runs.

    The per-step gather is hooked via ``_has_global_unfinished_reqs`` (called
    every iteration by upstream's ``run_busy_loop`` on every non-idle path),
    NOT from inside ``schedule()``: a rank that has drained its local requests
    never enters ``schedule()`` (it runs a dummy batch instead), so an
    ``all_gather`` living in ``schedule()`` would be skipped by that rank while
    busy ranks call it -- a collective mismatch that deadlocks.

    Why ``_has_global_unfinished_reqs`` and not ``_process_engine_step``:
    ``_has_global_unfinished_reqs`` is itself a cross-rank collective (an
    all-reduce every 32 steps internally) and is the only point in the busy
    loop that re-synchronizes ranks on wave/idle state. Hooking the gather
    immediately after ``super()._has_global_unfinished_reqs()`` keeps the
    every-step all-gather in the same lock-stepped region, so ranks enter the
    gather having just agreed on ``engines_running``. Hooking it earlier --
    inside ``_process_engine_step``, before that sync and before the idle
    ``continue`` gate -- decouples the gather from the synchronization: at
    wave boundaries one rank can reach the gather while another is still
    blocked in ``_process_input_queue`` or ``future.result()``, deadlocking
    the all-gather. The stuck EngineCore then can't drain its worker shm
    channel, the worker's ``sample_tokens`` response has nowhere to land, and
    after 60s the engine dies with ``RPC call to sample_tokens timed out``.
    Because ``_has_global_unfinished_reqs`` is only called on iterations that
    did NOT take the idle ``continue``, gather is skipped consistently by
    every rank when all are idle -- no rank does an extra gather. This matches
    the pre-refactor copied ``run_busy_loop``, where gather sat right after
    the all-reduce (after schedule + execute + update_from_output).
    """

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        result = super()._has_global_unfinished_reqs(local_unfinished)
        # Inject the DP group (idempotent) and refresh the cross-rank running
        # snapshot once per non-idle iteration. balance_gather is a no-op when
        # balance is disabled or dp_group is unset, so this is safe on every
        # path. MUST run immediately after the sync collective so ranks enter
        # the all-gather already aligned -- see class docstring.
        self.scheduler.dp_group = self.dp_group
        self.scheduler.balance_gather()
        return result


# The scheduler is constructed as ``Scheduler(...)`` from
# ``vllm.v1.core.sched.scheduler``. This only takes effect when scheduler_cls
# is unset (the PD-mixed balance path); vllm-ascend's recompute / dynamic-batch
# / profiling schedulers set scheduler_cls and bypass this name, which is
# correct -- balance scheduling (PD-mixed) must not touch those configs.
_sched_mod.Scheduler = BalanceScheduler

# Activate BalanceDPEngineCoreProc ONLY when balance scheduling is enabled.
# Upstream ``run_engine_core`` resolves ``DPEngineCoreProc`` via the
# ``vllm.v1.engine.core`` module-global name whenever DP>1 + MoE, so an
# unconditional swap would inject balance machinery into configs that don't use
# it -- e.g. PD-disaggregated recompute, whose scheduler is AsyncRecomputeScheduler
# and must not be touched by balance. A conditional swap at run_engine_core
# entry (where vllm_config is available) restores the pre-refactor "balance off
# => no involvement" invariant without copying run_engine_core's body.
_OriginalDPEngineCoreProc = _engine_core_mod.DPEngineCoreProc
_OriginalRunEngineCore = EngineCoreProc.run_engine_core


def _balance_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    if _balance_scheduling_enabled(vllm_config):
        _engine_core_mod.DPEngineCoreProc = BalanceDPEngineCoreProc
    else:
        _engine_core_mod.DPEngineCoreProc = _OriginalDPEngineCoreProc
    return _OriginalRunEngineCore(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


EngineCoreProc.run_engine_core = staticmethod(_balance_run_engine_core)
