import queue
from concurrent.futures import Future
from typing import Optional, Tuple, Type

from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm.v1.core.scheduler import Scheduler, SchedulerOutput
from vllm.v1.engine.mm_input_cache import MMInputCacheServer
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.version import __version__ as VLLM_VERSION

from vllm_ascend.core.scheduler import AscendScheduler

logger = init_logger(__name__)


def engine_core_init_with_ascend_scheduler(
    self,
    vllm_config: VllmConfig,
    executor_class: Type[Executor],
    log_stats: bool,
):
    assert vllm_config.model_config.runner_type != "pooling"

    logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION, vllm_config)

    self.log_stats = log_stats

    # Setup Model.
    self.model_executor = executor_class(vllm_config)

    # Setup KV Caches and update CacheConfig after profiling.
    num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(
        vllm_config)
    vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
    vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks

    # Setup scheduler.
    self.scheduler = AscendScheduler(
        scheduler_config=vllm_config.scheduler_config,
        model_config=vllm_config.model_config,
        cache_config=vllm_config.cache_config,
        lora_config=vllm_config.lora_config,
        speculative_config=vllm_config.speculative_config,
        log_stats=self.log_stats,
    )

    # Setup MM Input Mapper.
    self.mm_input_cache_server = MMInputCacheServer(
        vllm_config.model_config)

    # Setup batch queue for pipeline parallelism.
    # Batch queue for scheduled batches. This enables us to asynchronously
    # schedule and execute batches, and is required by pipeline parallelism
    # to eliminate pipeline bubbles.
    self.batch_queue_size = self.model_executor.max_concurrent_batches
    self.batch_queue: Optional[queue.Queue[Tuple[Future[ModelRunnerOutput],
                                                    SchedulerOutput]]] = None
    if self.batch_queue_size > 1:
        logger.info("Batch queue is enabled with size %d",
                    self.batch_queue_size)
        self.batch_queue = queue.Queue(self.batch_queue_size)