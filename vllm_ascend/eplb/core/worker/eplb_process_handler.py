# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import time
import numpy as np
import torch_npu
import logging
from multiprocessing import Process, Queue
import multiprocessing

logger = logging.getLogger(__name__)


def do_eplb_mp(
    device: int,
    planner_block_queue: Queue,
    block_update_queue: Queue,
    eplb_forwarder,
    eplb_planner,
):
    """
    Child process entry for EPLB. It will block on planner_block_queue.get()
    until it is woken up by a parent call.
    """
    # Bind this process to the given NPU
    torch_npu.npu.set_device(device)
    logger.info(f"[EPLB Process on NPU:{device}] Bound to device, waiting for wake-up signals")
    init_expert_map = True

    try:
        while True:
            # 1) Block until parent calls planner_block_queue.put(...)
            planner_block_queue.get()

            if init_expert_map:
                old_map = eplb_planner.get_init_map()
                init_expert_map = False

            # 2) Once woken up, fetch load info
            load_info = eplb_forwarder.fetch_and_sum_load_info()
            logger.debug(f"[EPLB Process on NPU:{device}] get load_info")

            if load_info is None:
                # Nothing to do this round
                continue

            # 3) Compute rebalance via planner

            # changed, priority, new_map = eplb_planner.calculate_rebalance_experts(load_info)
            #test new_map:
            new_map = old_map

            # 4) If warmup done and mapping changed, do one-shot H2D

            logger.debug(f"[EPLB Process on NPU:{device}] new_map differs, performing D2D")

            try:
                eplb_forwarder.load_experts_to_device()
                logger.debug(f"[EPLB Process on NPU:{device}] D2D update completed")
            except Exception as ex:
                logger.warning(f"[EPLB Process on NPU:{device}] D2D update failed: {ex}", exc_info=True)

            old_map = new_map
            eplb_forwarder.update_expert_map(new_map)
            block_update_queue.put(1)

    except Exception as e:
        logger.warning(f"[EPLB Process on NPU:{device}] Exception: {e}", exc_info=True)
        # Decide whether to exit or retry
        pass


def launch_eplb_process(
    device: int,
    eplb_forwarder,
    eplb_planner
):
    """
    Launch a separate process to run do_eplb_mp. Return (planner_block_queue, block_update_queue, process).
    - device: NPU ID
    - eplb_forwarder, eplb_planner: interfaces
    """

    ctx = multiprocessing.get_context("spawn")

    planner_block_queue = ctx.Queue()
    block_update_queue = ctx.Queue()

    p = ctx.Process(
        target=do_eplb_mp,
        args=(
            device,
            planner_block_queue,
            block_update_queue,
            eplb_forwarder,
            eplb_planner
        ),
        daemon=False
    )
    p.start()
    return planner_block_queue, block_update_queue, p