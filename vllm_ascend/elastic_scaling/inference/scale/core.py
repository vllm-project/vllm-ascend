# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


def reload_model(self) -> None:
    """
    Load weights by setting flag INFER_STATUS = 1
    """
    os.environ["INFER_STATUS"] = "1"
    self.model_executor.reload_model()


def reload_kvcache(self) -> None:
    """
    Load KVCache by setting flag INFER_STATUS = 1
    """
    os.environ["INFER_STATUS"] = "1"
    num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(self.vllm_config)

    ## We need to reinitialize
    vllm_config = self.vllm_config

    vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
    vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
    self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

    self.structured_output_manager = StructuredOutputManager(vllm_config)

    # Setup scheduler.
    Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

    if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
        # Encoder models without KV cache don't support
        # chunked prefill. But do SSM models?
        if vllm_config.scheduler_config.enable_chunked_prefill:
            logger.warning("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.enable_chunked_prefill = False

    scheduler_block_size = (
        vllm_config.cache_config.block_size
        * vllm_config.parallel_config.decode_context_parallel_size
        * vllm_config.parallel_config.prefill_context_parallel_size
    )

    self.scheduler: SchedulerInterface = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        structured_output_manager=self.structured_output_manager,
        include_finished_set=self.include_finished_set,
        log_stats=self.log_stats,
        block_size=scheduler_block_size,
    )

    self.use_spec_decode = vllm_config.speculative_config is not None
    if self.scheduler.connector is not None:  # type: ignore
        self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

    self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
    self.mm_receiver_cache = engine_receiver_cache_from_config(vllm_config, mm_registry)

    # If a KV connector is initialized for scheduler, we want to collect
    # handshake metadata from all workers so the connector in the scheduler
    # will have the full context
    kv_connector = self.scheduler.get_kv_connector()
    if kv_connector is not None:
        # Collect and store KV connector xfer metadata from workers
        # (after KV cache registration)
        xfer_handshake_metadata = self.model_executor.get_kv_connector_handshake_metadata()

        if xfer_handshake_metadata:
            # xfer_handshake_metadata is list of dicts from workers
            # Each dict already has structure {tp_rank: metadata}
            # Merge all worker dicts into a single dict
            content: dict[int, Any] = {}
            for worker_dict in xfer_handshake_metadata:
                if worker_dict is not None:
                    content.update(worker_dict)
            kv_connector.set_xfer_handshake_metadata(content)

    # Setup batch queue for pipeline parallelism.
    # Batch queue for scheduled batches. This enables us to asynchronously
    # schedule and execute batches, and is required by pipeline parallelism
    # to eliminate pipeline bubbles.
    self.batch_queue_size = self.model_executor.max_concurrent_batches
    self.batch_queue: deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None = None
    if self.batch_queue_size > 1:
        logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
        self.batch_queue = deque(maxlen=self.batch_queue_size)

    self.is_ec_producer = vllm_config.ec_transfer_config is not None and vllm_config.ec_transfer_config.is_ec_producer
    self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

    self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
    if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
        caching_hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
        init_none_hash(caching_hash_fn)

        self.request_block_hasher = get_request_block_hasher(scheduler_block_size, caching_hash_fn)

    self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue
    self.async_scheduling = vllm_config.scheduler_config.async_scheduling

    self.aborts_queue = queue.Queue[list[str]]()
    # Mark the startup heap as static so that it's ignored by GC.
    # Reduces pause times of oldest generation collections.
    freeze_gc_heap()
    # If enable, attach GC debugger after static variable freeze.
    maybe_attach_gc_debug_callback()
    # Enable environment variable cache (e.g. assume no more
    # environment variable overrides after this point)
    enable_envs_cache()
