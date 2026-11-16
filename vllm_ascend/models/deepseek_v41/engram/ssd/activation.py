# SPDX-License-Identifier: Apache-2.0
"""Bounded, ordered SSD activation in engine parents, before engine READY.

This uses the engine-parent CPU group, never the worker's per-step group.
A failed activation is fatal to startup; the process group is not reused.
"""

from datetime import timedelta
from time import monotonic

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)


def activate_workers(model_executor, *, dp_group, worker_count: int, timeout_s: float) -> None:
    """Rendezvous after warmup, activate local workers, agree global success."""
    if timeout_s <= 0 or worker_count < 1:
        raise ValueError("activation requires a positive timeout and worker count")
    deadline = monotonic() + timeout_s

    def remaining() -> float:
        seconds = deadline - monotonic()
        if seconds <= 0:
            raise TimeoutError("Engram SSD activation deadline exceeded")
        return seconds

    def wait(work) -> None:
        work.wait(timeout=timedelta(seconds=remaining()))

    # No worker may switch from startup-local behavior while a peer is still
    # profiling/capturing. Parents use a separate group and submit no forwards
    # between this rendezvous and successful activation agreement.
    wait(dist.barrier(group=dp_group, async_op=True))
    local_error = None
    try:
        acknowledgements = model_executor.collective_rpc(
            "activate_engram_ssd_protocol", timeout=remaining(), args=(1,)
        )
        if (len(acknowledgements) != worker_count
                or any(type(epoch) is not int or epoch != 1 for epoch in acknowledgements)):
            raise RuntimeError(f"Invalid Engram SSD activation acknowledgements: {acknowledgements!r}")
    except Exception as exc:
        local_error = exc
        logger.exception("Engram SSD local worker activation failed")

    # Every surviving parent joins even if its local RPC failed. None may
    # announce READY while another engine has rejected a worker ack.
    success = torch.tensor([int(local_error is None)], dtype=torch.int32)
    wait(dist.all_reduce(success, op=dist.ReduceOp.MIN, group=dp_group, async_op=True))
    if not success.item():
        raise RuntimeError("Engram SSD activation failed on one or more DP engines") from local_error
    logger.info("Engram SSD activation complete: epoch=1, local_workers=%d", worker_count)


def activate_engine_protocol(engine) -> None:
    """Post-warmup hook; feature-off and unshared DP=1 take no collectives."""
    from vllm_ascend import envs
    from vllm_ascend.ascend_config import get_ascend_config

    config = get_ascend_config().engram_ssd
    if not config.enabled:
        return
    if envs.VLLM_ASCEND_ENGRAM_SSD_EPOCH_FILE:
        logger.warning(
            "VLLM_ASCEND_ENGRAM_SSD_EPOCH_FILE is ignored; Engram SSD now "
            "activates through the engine startup RPC. Remove the epoch-file watcher."
        )
    parallel = engine.vllm_config.parallel_config
    if parallel.data_parallel_size == 1:
        return
    if getattr(engine, "dp_group", None) is None:
        raise RuntimeError("Engram SSD DP activation requires the engine-parent CPU group")
    activate_workers(
        engine.model_executor,
        dp_group=engine.dp_group,
        worker_count=parallel.world_size,
        timeout_s=config.protocol_activation_timeout_s,
    )
