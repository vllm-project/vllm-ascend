# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    reset_group_name_registry,
)
from vllm.logger import logger


def cleanup_dist_env_for_snapshot(shutdown_ray: bool = False) -> None:
    """Clear distributed state before rebuilding it after restore."""
    destroy_model_parallel()
    logger.info("Snapshot model-parallel groups destroyed")
    destroy_distributed_environment()
    reset_group_name_registry()
    if shutdown_ray:
        import ray

        ray.shutdown()
