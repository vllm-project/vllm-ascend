"""
Batch-Job-Aware Scheduler for vLLM v1
=====================================

Designed for **offline batch** scenarios where throughput and hardware
utilisation are the primary goals.

The main strategy is as follows:

1. By adopting the LPT (Longest Processing Time first) strategy, we
   prioritize longer tasks first and schedule shorter ones to fill
   in the gaps, particularly during the decoding step. This approach
   improves the average number of tokens computed per scheduling round.
2. Estimate and reserve KV cache budget in advance for running requests
   to reduce preemption.

Inherits from the base ``Scheduler`` and replaces the waiting queue with
a custom queue that:

1. Groups requests by **job name** (extracted from ``#job_name[${JOB_NAME}]#``
   prefix), estimates decode length per job via **EWMA**.
2. Predicts next-step **KV block needs** for running requests (reserve),
   uses **reserve-aware budget** to suppress preemption.
3. Each job has its own **RequestBucket** for efficient request management,
   uses **RequestBucket** for O(log n) best-fit search by prefill length.
4. Dynamically adjusts job scheduling order based on KV Cache availability:
   - When available tokens > threshold (default 4096): prioritize long decode jobs
   - When available tokens <= threshold: prioritize short decode jobs

Usage
-----

Pass the class (or its fully-qualified name) via ``scheduler_cls``::

    from vllm.v1.core.sched.batch_job_aware_scheduler import BatchJobAwareScheduler

    engine = LLMEngine.from_engine_args(EngineArgs(model=..., scheduler_cls=BatchJobAwareScheduler))

"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Iterator
from typing import Any

import regex as re
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.ascend_config import BatchJobSchedConfig


class JobNameParser:
    """Job name parser with caching for better performance."""

    __slots__ = ("_pattern", "_cache")

    def __init__(self) -> None:
        self._pattern = re.compile(r"#job_name\[([^\[\]]+)\]#")
        self._cache: dict[str, str] = {}

    def parse(self, request_id: str) -> str:
        """Extract the job name from a request ID with caching."""
        job_name = self._cache.get(request_id)
        if job_name is not None:
            return job_name
        m = self._pattern.search(request_id)
        job_name = m.group(1) if m else "__default__"
        self._cache[request_id] = job_name
        return job_name

    def remove(self, request_id: str) -> None:
        """Remove a request ID from the cache."""
        self._cache.pop(request_id, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


class _JobStat:
    """Per‑job running statistics."""

    __slots__ = ("sample_count", "ewma_average")

    def __init__(self) -> None:
        self.sample_count: int = 0
        # a sensible value before any observation has been recorded.
        self.ewma_average: float = 0.0


class JobDecodeEstimator:
    """Per‑job decode‑length estimator with EWMA.

    This estimator uses EWMA (Exponentially Weighted Moving Average) for
    predicting decode lengths. When no samples exist for a job, it returns
    a cold-start default value. Once the first sample is observed, the
    EWMA average is set to that value and refined with each new sample.

    EWMA Formula:
        EWMA_t = α * x_t + (1 - α) * EWMA_{t-1}

        Where:
            - EWMA_t: New EWMA average at time t
            - α: Smoothing factor (_EWMA_ALPHA = 0.1), 0 < α <= 1
            - x_t: Current observation (decode_len)
            - EWMA_{t-1}: Previous EWMA average

    Properties of EWMA:
        1. **Weight decay**: Older observations have exponentially decreasing
           influence. The weight of observation x_{t-k} is α(1-α)^k.
        2. **Effective window size**: The "half-life" of observations is
           approximately ln(0.5)/ln(1-α). With α=0.1, half-life ≈ 6.58 samples.
        3. **Initialization**: For the first sample (n=1), EWMA is simply
           set to the observed value: EWMA_1 = x_1
        4. **Sensitivity control**:
           - Higher α (e.g., 0.3): More responsive to recent changes
           - Lower α (e.g., 0.1): More stable, slower to adapt
    """

    # Fallback predicted decode length when no samples exist.
    _COLD_START_DEFAULT_DECODE: int = 128
    # EWMA smoothing factor (0 < α <= 1).
    _EWMA_ALPHA: float = 0.1
    # Maximum samples per job for decode length estimation.
    _MAX_SAMPLES_PER_JOB: int = 10

    def __init__(self, config: BatchJobSchedConfig) -> None:
        self._config = config
        self._job_stats: dict[str, _JobStat] = {}

    def predict(self, job_name: str) -> int:
        """Return the predicted decode length for *job*."""
        job_stat = self._job_stats.get(job_name)
        if job_stat is None:
            return self._COLD_START_DEFAULT_DECODE
        return int(job_stat.ewma_average)

    def observe(self, job_name: str, decode_len: int) -> None:
        """Record one decode‑length observation for *job*.

        This method updates the job statistics using the EWMA (Exponentially
        Weighted Moving Average) formula for online learning.

        Args:
            job_name: The name of the job to record observation for.
            decode_len: The observed decode length to record.
        """
        job_stat = self._job_stats.get(job_name)
        if job_stat is None:
            # Check max_jobs limit before creating a new job stat
            if len(self._job_stats) >= self._config.max_jobs:
                raise RuntimeError(
                    f"Maximum number of jobs ({self._config.max_jobs}) exceeded in JobDecodeEstimator. "
                    f"Cannot create new job stat for '{job_name}'. "
                    f"Current number of jobs: {len(self._job_stats)}."
                )
            job_stat = _JobStat()
            self._job_stats[job_name] = job_stat

        # Skip update when max samples reached for performance
        if job_stat.sample_count >= self._MAX_SAMPLES_PER_JOB:
            return

        job_stat.sample_count += 1
        if job_stat.sample_count == 1:
            job_stat.ewma_average = float(decode_len)
        else:
            job_stat.ewma_average = self._EWMA_ALPHA * decode_len + (1 - self._EWMA_ALPHA) * job_stat.ewma_average

    def get_stats(self) -> dict[str, dict[str, int | float]]:
        """Return human‑readable statistics for all tracked jobs."""
        return {
            job: {
                "sample_count": s.sample_count,
                "ewma_average": round(s.ewma_average, 1),
            }
            for job, s in self._job_stats.items()
        }


def _cdiv(a: int, b: int) -> int:
    """Ceiling division: smallest integer >= a/b."""
    return -(a // -b)


class RunningBlockReserver:
    """Predict the KV‑block demand of the next running‑loop pass.

    Results are cached per scheduling step to avoid redundant iteration
    over the running-request list. The cache uses **incremental updates**:
    when a new request is appended to the running set between consecutive
    ``predict()`` calls within the same admission loop, only the delta
    for that single request is computed (O(1)) instead of scanning the
    entire running list (O(N)).

    Call ``invalidate_cache()`` at the start of each ``schedule()``
    round to force a full recompute on the first ``predict()`` call.
    """

    def __init__(self, scheduler: Scheduler, config: BatchJobSchedConfig) -> None:
        self._scheduler = scheduler
        self._config = config
        # Pre-margin sum of per-request block predictions.
        self._cached_predict_blocks: int = 0
        # Number of running requests that _cached_raw_reserve corresponds to.
        # -1 means the cache is invalid and a full recompute is needed.
        self._cached_running_queue_len: int = -1

    def invalidate_cache(self) -> None:
        """Mark the cached prediction as stale so the next call recomputes."""
        self._cached_running_queue_len = -1

    def predict(self) -> int:
        """Return the number of blocks to reserve for the current running set."""
        current_len = len(self._scheduler.running)

        # Case 1: Length unchanged → cache hit
        if current_len == self._cached_running_queue_len:
            return self._finalize_reserve(self._cached_predict_blocks)

        # Case 2: Running grew by exactly 1 request within the same admission
        if current_len == self._cached_running_queue_len + 1 and self._cached_running_queue_len >= 0:
            new_request = self._scheduler.running[-1]
            self._cached_predict_blocks += self._compute_one(new_request)
            self._cached_running_queue_len = current_len
            return self._finalize_reserve(self._cached_predict_blocks)

        # Case 3: Complex change (new schedule round)
        self._cached_predict_blocks = self._compute_all()
        self._cached_running_queue_len = current_len
        return self._finalize_reserve(self._cached_predict_blocks)

    def _finalize_reserve(self, predict_blocks: int) -> int:
        return min(
            predict_blocks + self._config.reserve_margin_blocks,
            self._config.reserve_max_blocks,
        )

    def _compute_one(self, request: Request) -> int:
        """Return the raw (pre-margin) number of new KV blocks *request*
        will need in the next scheduling step.

        This method simulates the effect of the *current* scheduling round
        on the request's position, then predicts blocks needed for the
        *next* round from that advanced position. This eliminates a one-step
        timeline misalignment: when this method is called (during the waiting-
        queue admission phase of ``schedule()``), the running requests have
        already been assigned ``num_new_tokens`` for this round but their
        ``num_computed_tokens`` has not yet been advanced (that happens in
        ``update_from_output`` post-execution). By simulating the advance we
        get the correct starting position for the next round's prediction.
        """
        num_computed = request.num_computed_tokens
        num_prompt = request.num_prompt_tokens
        block_size = self._scheduler.block_size

        # Step 1: estimate how many tokens this round will advance
        if num_computed < num_prompt:
            remaining_prompt = num_prompt - num_computed
            this_round_tokens = min(remaining_prompt, self._scheduler.max_num_scheduled_tokens)
        else:
            this_round_tokens = 1 + self._scheduler.num_lookahead_tokens

        # Step 2: simulate position after this round's execution
        num_computed_after = num_computed + this_round_tokens

        # Step 3: estimate the NEXT round's token demand
        if num_computed_after < num_prompt:
            # Still in prefill next round
            remaining_prompt_after = num_prompt - num_computed_after
            next_round_tokens = min(remaining_prompt_after, self._scheduler.max_num_scheduled_tokens)
        else:
            # Transitions to (or stays in) decode
            next_round_tokens = 1 + self._scheduler.num_lookahead_tokens

        # Step 4: compute new blocks needed from that position
        pos_in_block = num_computed_after % block_size
        remaining = block_size - pos_in_block if pos_in_block > 0 else 0

        if next_round_tokens > remaining:
            return _cdiv(next_round_tokens - remaining, block_size)
        return 0

    def _compute_all(self) -> int:
        return sum(self._compute_one(req) for req in self._scheduler.running)


class RequestBucket:
    """Request bucket based on prefill length, supports O(log n) best-fit search."""

    def __init__(self) -> None:
        self._bucket: dict[int, deque[Request]] = {}
        self._sorted_lengths: list[int] = []
        self._num_requests: int = 0

    def put(self, request: Request) -> None:
        """Put request into bucket."""
        if request is None:
            return

        prefill_length = request.num_prompt_tokens

        if prefill_length not in self._bucket:
            self._bucket[prefill_length] = deque()
            bisect.insort(self._sorted_lengths, prefill_length)

        self._bucket[prefill_length].append(request)
        self._num_requests += 1

    def peek_best_fit_request(self, search_prefill_length: int) -> Request | None:
        """Find best-fit request from bucket based on available token count."""
        if self.is_empty() or not self._bucket or not self._sorted_lengths:
            return None

        idx = bisect.bisect_right(self._sorted_lengths, search_prefill_length)
        if idx > 0:
            best_length = self._sorted_lengths[idx - 1]
            queue = self._bucket.get(best_length)
            if queue:
                return queue[0]

        if idx < len(self._sorted_lengths):
            best_length = self._sorted_lengths[idx]
            queue = self._bucket.get(best_length)
            if queue:
                return queue[0]

        return None

    def remove(self, request: Request) -> bool:
        """Remove specified request from bucket."""
        if request is None:
            return False

        prefill_length = request.num_prompt_tokens
        queue = self._bucket.get(prefill_length)
        if queue is None:
            return False

        try:
            queue.remove(request)
            self._num_requests -= 1
            if not queue:
                del self._bucket[prefill_length]
                idx = bisect.bisect_left(self._sorted_lengths, prefill_length)
                if idx < len(self._sorted_lengths) and self._sorted_lengths[idx] == prefill_length:
                    self._sorted_lengths.pop(idx)
            return True
        except ValueError:
            return False

    def is_empty(self) -> bool:
        """Check if bucket is empty."""
        return self._num_requests == 0

    def __len__(self) -> int:
        return self._num_requests

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __iter__(self) -> Iterator[Request]:
        for queue in self._bucket.values():
            yield from queue


class BatchJobAwareRequestQueue(RequestQueue):
    """Request queue with job‑grouped sorting and reserve‑aware admission.

    Bucket Structure:
        - Each job has its own RequestBucket: dict[job_name, RequestBucket]
        - Jobs are classified as LONG/SHORT based on predicted decode length

    Scheduling Priority:
        1. Cold-start requests (first N requests per job) - highest priority
        2. Regular scheduling based on available tokens:
           - Available tokens > threshold (default 4096): prioritize long decode jobs
           - Available tokens <= threshold: prioritize short decode jobs
           - Job length is determined by short_decode_token_threshold (default 32)

    Job Classification:
        - Short decode job: predicted decode length < short_decode_token_threshold
        - Long decode job: predicted decode length >= short_decode_token_threshold

    Within-Category Sorting:
        - Long jobs: sorted by decode length descending (longest first)
        - Short jobs: sorted by decode length ascending (shortest first)
    """

    # Minimum samples needed to exit cold-start scheduling priority.
    _COLD_START_MIN_SAMPLES: int = 3

    def __init__(
        self,
        scheduler: Scheduler,
        job_decode_estimator: JobDecodeEstimator,
        block_reserver: RunningBlockReserver,
        job_name_parser: JobNameParser,
        config: BatchJobSchedConfig,
    ) -> None:
        self._scheduler = scheduler
        self._job_decode_estimator = job_decode_estimator
        self._block_reserver = block_reserver
        self._job_name_parser = job_name_parser
        self._config = config

        # Each job has its own RequestBucket
        self._job_buckets: dict[str, RequestBucket] = {}
        self._cold_start_reqs: dict[str, set[str]] = {}
        self._job_req_count: dict[str, int] = {}
        self._num_requests: int = 0
        self._peeked: Request | None = None

    def add_request(self, request: Request) -> None:
        """Add a request to the queue."""
        if request is None:
            return

        job_name = self._get_job_name(request)
        bucket = self._get_or_create_job_bucket(job_name)
        bucket.put(request)
        self._num_requests += 1
        self._track_cold_start_req(request)

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        if request is None:
            return

        job_name = self._get_job_name(request)
        if job_name is None:
            return
        bucket = self._job_buckets.get(job_name)
        if bucket is None:
            return

        if bucket.remove(request):
            self._num_requests -= 1
            if job_name in self._cold_start_reqs:
                self._cold_start_reqs[job_name].discard(request.request_id)
                if not self._cold_start_reqs[job_name]:
                    del self._cold_start_reqs[job_name]
            if bucket.is_empty():
                del self._job_buckets[job_name]
                self._job_req_count.pop(job_name, None)

    def remove_requests(self, requests: Any) -> None:
        for request in requests:
            self.remove_request(request)

    def peek_request(self) -> Request:
        if self._peeked is not None:
            return self._peeked
        request = self._find_admittable()
        if request is None:
            raise IndexError("no admittable request under reserve")
        self._peeked = request
        return request

    def pop_request(self) -> Request:
        request = self._peeked if self._peeked is not None else self._find_admittable()
        if request is None:
            raise IndexError("pop from empty admittable set")
        self.remove_request(request)
        self._peeked = None
        return request

    def _get_job_name(self, request):
        return self._job_name_parser.parse(request.request_id)

    def _track_cold_start_req(self, request: Request) -> None:
        """Track a request as a cold-start request if applicable."""
        job_name = self._get_job_name(request)

        if job_name not in self._job_req_count:
            self._job_req_count[job_name] = 0
        self._job_req_count[job_name] += 1

        if self._job_req_count[job_name] <= self._COLD_START_MIN_SAMPLES:
            if job_name not in self._cold_start_reqs:
                self._cold_start_reqs[job_name] = set()
            self._cold_start_reqs[job_name].add(request.request_id)

    def _get_or_create_job_bucket(self, job_name: str) -> RequestBucket:
        """Get or create a RequestBucket for a job."""
        if job_name not in self._job_buckets:
            # Check max_jobs limit before creating a new job bucket
            if self._config.max_jobs and len(self._job_buckets) >= self._config.max_jobs:
                raise RuntimeError(
                    f"Maximum number of jobs ({self._config.max_jobs}) exceeded. "
                    f"Cannot create new job '{job_name}'. "
                    f"Current number of jobs: {len(self._job_buckets)}."
                )
            self._job_buckets[job_name] = RequestBucket()
        return self._job_buckets[job_name]

    def _free_blocks(self) -> int:
        return self._scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    def _admission_budget(self) -> int:
        """Calculate admission budget in tokens."""
        free_blocks = self._free_blocks()
        reserve_blocks = self._block_reserver.predict()
        block_size = self._scheduler.block_size
        return max(0, (free_blocks - reserve_blocks) * block_size)

    def _find_admittable(self) -> Request | None:
        """Return the highest‑priority admittable request, or ``None``.

        Scheduling Priority:
            1. Cold-start requests (first N requests per job) - highest priority
            2. Regular scheduling based on available tokens:
               - Available tokens > threshold: prioritize long decode jobs
               - Available tokens <= threshold: prioritize short decode jobs
        """
        admission_budget = self._admission_budget()
        if admission_budget == 0:
            return None

        # Priority 1: Cold-start requests
        request = self._get_cold_start_request()
        if request is not None:
            return request

        # Collect job decode information
        long_jobs, short_jobs = self._collect_job_decode_info()

        # Priority 2: Regular scheduling based on available tokens
        return self._try_schedule_by_available_tokens(admission_budget, long_jobs, short_jobs)

    def _get_cold_start_request(self) -> Request | None:
        """Try to get a cold-start request with highest priority."""
        for job_name, cold_start_reqs in self._cold_start_reqs.items():
            bucket = self._job_buckets.get(job_name)
            if bucket is None:
                continue
            for request in bucket:
                if request.request_id in cold_start_reqs:
                    return request
        return None

    def _collect_job_decode_info(self) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        """Collect job decode length information."""
        # Separate jobs into long and short categories
        long_jobs: list[tuple[str, int]] = []
        short_jobs: list[tuple[str, int]] = []

        short_threshold = self._config.short_decode_token_threshold

        for job_name, bucket in self._job_buckets.items():
            if bucket.is_empty():
                continue
            predict_decode = self._job_decode_estimator.predict(job_name)
            if predict_decode > short_threshold:
                long_jobs.append((job_name, predict_decode))
            else:
                short_jobs.append((job_name, predict_decode))

        return long_jobs, short_jobs

    def _try_schedule_by_available_tokens(
        self, admission_budget: int, long_jobs: list[tuple[str, int]], short_jobs: list[tuple[str, int]]
    ) -> Request | None:
        """Schedule requests based on job decode length.

        Strategy:
            - When available tokens > threshold: prioritize long decode jobs
            - When available tokens <= threshold: prioritize short decode jobs
        """
        # Sort jobs
        ordered_jobs = self._sort_jobs(admission_budget, long_jobs, short_jobs)

        for job_name, _ in ordered_jobs:
            bucket = self._job_buckets.get(job_name)
            if bucket is None or bucket.is_empty():
                continue
            request = bucket.peek_best_fit_request(admission_budget)
            if request is not None:
                return request

        return None

    def _sort_jobs(
        self, admission_budget: int, long_jobs: list[tuple[str, int]], short_jobs: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        prioritize_long = admission_budget > self._config.low_available_tokens_threshold

        if len(long_jobs) > 0 and len(short_jobs) > 0:
            long_jobs.sort(key=lambda x: x[1], reverse=True)
            short_jobs.sort(key=lambda x: x[1], reverse=True)
            ordered_jobs = long_jobs + short_jobs if prioritize_long else short_jobs + long_jobs
        else:
            long_jobs.sort(key=lambda x: x[1], reverse=prioritize_long)
            short_jobs.sort(key=lambda x: x[1], reverse=prioritize_long)
            ordered_jobs = long_jobs + short_jobs

        return ordered_jobs

    def __len__(self) -> int:
        return self._num_requests

    def __iter__(self) -> Iterator[Request]:
        for bucket in self._job_buckets.values():
            yield from bucket

    def __bool__(self) -> bool:
        return self._find_admittable() is not None


class BatchJobAwareScheduler(Scheduler):
    """Batch‑job‑aware scheduler for offline / batched inference workloads."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Initialize scheduler configuration.
        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config

        init_ascend_config(self.vllm_config)
        _config: BatchJobSchedConfig = get_ascend_config().batch_job_sched_config

        self._job_name_parser = JobNameParser()
        self._job_decode_estimator = JobDecodeEstimator(_config)
        self._block_reserver = RunningBlockReserver(self, _config)

        self.waiting = BatchJobAwareRequestQueue(
            self, self._job_decode_estimator, self._block_reserver, self._job_name_parser, _config
        )

    def schedule(self):
        """Override to invalidate the block-reserve cache before each step.

        Inside the parent ``schedule()``, the waiting queue calls
        ``RunningBlockReserver.predict()`` for each request being checked.
        The cache auto-detects changes to the running set and computes
        incremental O(1) deltas when a single request is added, avoiding
        redundant full scans.
        """
        self._block_reserver.invalidate_cache()
        return super().schedule()

    def _free_request(self, request: Request, delay_free_blocks: bool = False) -> dict[str, Any] | None:
        """Observe the decode length for naturally stopped requests."""
        if request.status == RequestStatus.FINISHED_STOPPED:
            job_name = self._job_name_parser.parse(request.request_id)
            self._job_decode_estimator.observe(job_name, request.num_output_tokens)
        self._job_name_parser.remove(request.request_id)
        return super()._free_request(request, delay_free_blocks)


class BatchJobAwareAsyncScheduler(AsyncScheduler, BatchJobAwareScheduler):
    """Batch-job-aware + asynchronous scheduling, combined via multiple inhernce.

    This class inherits from both ``AsyncScheduler`` and
    ``BatchJobAwareScheduler`` to compose asynchronous scheduling with
    batch-job-aware scheduling logic.

    Dependency import chain
    -----------------------
    - ``AsyncScheduler`` — imported from ``vllm.v1.core.sched.async_scheduler``,
      provides the asynchronous scheduling loop and async-schedule interface.
    - ``BatchJobAwareScheduler`` — defined in this same module, inherits from
      ``Scheduler`` and provides batch-job-aware scheduling logic (LPT
      strategy, KV cache reservation, job grouping, etc.).
    - ``Scheduler`` — base class imported from
      ``vllm.v1.core.scheduler`` (via ``BatchJobAwareScheduler``), provides
      the core scheduling primitive that the async scheduler wraps.

    MRO: BatchJobAwareAsyncScheduler -> AsyncScheduler -> BatchJobAwareScheduler -> Scheduler
    """

    pass
