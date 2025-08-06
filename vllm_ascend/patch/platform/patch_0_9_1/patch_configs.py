import vllm.envs as envs
from vllm.config import DistributedExecutorBackend, ParallelConfig
from vllm.logger import init_logger

import vllm_ascend.envs as vllm_ascend_envs

logger = init_logger(__name__)


def __post_init__(self: ParallelConfig) -> None:
    self.world_size = self.pipeline_parallel_size * \
        self.tensor_parallel_size

    if self.data_parallel_size_local > self.data_parallel_size:
        raise ValueError(
            f"data_parallel_size_local ({self.data_parallel_size_local}) "
            f"must be <= data_parallel_size ({self.data_parallel_size})")

    self.data_parallel_size = envs.VLLM_DP_SIZE
    self.data_parallel_rank = envs.VLLM_DP_RANK
    self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
    self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
    self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

    if self.distributed_executor_backend == "external_launcher":
        import os
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        logger.info("Disabling V1 multiprocessing for external launcher.")

    ray_only_devices: list[str] = []
    from vllm.platforms import current_platform
    if (current_platform.device_type in ray_only_devices
            and self.world_size > 1):
        if self.distributed_executor_backend is None:
            self.distributed_executor_backend = "ray"
        if self.distributed_executor_backend != "ray":
            raise ValueError(
                f"{current_platform.device_type.upper()} backend only "
                "supports Ray for distributed inference.")

    if self.distributed_executor_backend is None and self.world_size > 1:
        # We use multiprocessing by default if world_size fits on the
        # current node and we aren't in a ray placement group.

        from vllm.executor import ray_utils
        backend: DistributedExecutorBackend = "mp"
        ray_found = ray_utils.ray_is_available()
        if current_platform.is_neuron():
            # neuron uses single process to control multiple devices
            backend = "uni"
        elif current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
            backend = "uni"
        elif self.data_parallel_backend == "ray":
            logger.info("Using ray distributed inference because "
                        "data_parallel_backend is ray")
            backend = "ray"
        elif ray_found:
            if self.placement_group:
                backend = "ray"
            else:
                from ray import is_initialized as ray_is_initialized
                if ray_is_initialized():
                    from ray.util import get_current_placement_group
                    if get_current_placement_group():
                        backend = "ray"
        self.distributed_executor_backend = backend
        logger.info("Defaulting to use %s for distributed inference", backend)

    if self.distributed_executor_backend is None and self.world_size == 1:
        self.distributed_executor_backend = "uni"

    self._verify_args()


# apply this patch only if the external data parallelism is enabled
if vllm_ascend_envs.VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED:
    ParallelConfig.__post_init__ = __post_init__  # type: ignore[attr-defined]
