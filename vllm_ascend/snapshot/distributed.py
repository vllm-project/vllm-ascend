# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    reset_group_name_registry,
)
from vllm.logger import logger

_HCCL_TEARDOWN_ENABLED = False


@contextmanager
def snapshot_hccl_teardown(enabled: bool) -> Iterator[None]:
    """Enable snapshot-specific HCCL teardown for the current cleanup."""
    global _HCCL_TEARDOWN_ENABLED
    previous = _HCCL_TEARDOWN_ENABLED
    _HCCL_TEARDOWN_ENABLED = enabled
    try:
        yield
    finally:
        _HCCL_TEARDOWN_ENABLED = previous


def is_snapshot_hccl_teardown_enabled() -> bool:
    return _HCCL_TEARDOWN_ENABLED


def cleanup_dist_env_for_snapshot(shutdown_ray: bool = False) -> None:
    """Clear distributed state before rebuilding it after restore."""
    destroy_model_parallel()
    logger.info("Snapshot model-parallel groups destroyed")
    destroy_distributed_environment()
    reset_group_name_registry()
    if shutdown_ray:
        import ray

        ray.shutdown()
