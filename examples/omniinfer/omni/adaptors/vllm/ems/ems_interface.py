# Copyright (c) HuaWei Technologies Co., Ltd. 2025-2025. All rights reserved

from typing import List

from omni.adaptors.vllm.ems.ems_env import EmsEnv
from omni.adaptors.vllm.ems.ems_metrics import MetricsType, collect_metric
from vllm.v1.engine.core import EngineCoreOutputs, SchedulerOutput
from vllm.v1.executor.multiproc_executor import EXECUTE_MODEL_TIMEOUT_S, logger

if EmsEnv.enable_vllm_ems:
    from omni.adaptors.vllm.ems.ems_utils import cal_hash_blocks


def step(self) -> EngineCoreOutputs:
    """Schedule, execute, and make output."""
    # Check for any requests remaining in the scheduler - unfinished
    # or finished and not yet removed from the batch
    if not self.scheduler.has_requests():
        return EngineCoreOutputs(
            outputs=[],
            scheduler_stats=self.scheduler.make_stats(),
        )
    scheduler_output = self.scheduler.schedule()

    if EmsEnv.enable_vllm_ems:
        self._pre_cc_handle(scheduler_output)

    model_output = self.execute_model(scheduler_output)
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)  #type: ignore
    return engine_core_outputs


@collect_metric(MetricsType.PRE_CC_HANDLE)
def _pre_cc_handle(self, scheduler_output: SchedulerOutput) -> None:
    if len(scheduler_output.scheduled_new_reqs) == 0:
        return

    info_load_reqs, load_scheduled_new_reqs = _cal_info_load_reqs(
        scheduler_output, self.scheduler.block_size)
    if not info_load_reqs:
        return

    output = self.model_executor.load_kv_cache(info_load_reqs)
    if len(output) == 0:
        return

    if not hasattr(self, "total_blocks"):
        self.total_blocks = 0
        self.success_blocks = 0

    self.total_blocks += sum(len(v) for _, v, _ in info_load_reqs)
    self.success_blocks += sum(output)
    logger.info("[EMS] Load hit rate: %.2f%%",
                100.0 * self.success_blocks / max(self.total_blocks, 1))

    for num_block, new_req in zip(output, load_scheduled_new_reqs):
        ems_computed_tokens = num_block * self.scheduler.block_size
        new_req.num_computed_tokens += ems_computed_tokens
        scheduler_output.num_scheduled_tokens[
            new_req.req_id] -= ems_computed_tokens
        scheduler_output.total_num_scheduled_tokens -= ems_computed_tokens


@collect_metric(MetricsType.LOAD_KV_CACHE)
def load_kv_cache(self, info_load_reqs) -> List[int]:
    output = self.collective_rpc("load_kv_cache",
                                 args=(info_load_reqs, ),
                                 timeout=EXECUTE_MODEL_TIMEOUT_S)
    min_num_blocks = [min(tup) for tup in list(zip(*output))]
    logger.info(
        f'[EMS] num_blocks_list: {output}, min_num_blocks: {min_num_blocks}')
    return min_num_blocks


def _cal_info_load_reqs(scheduler_output: SchedulerOutput, block_size):
    info_load_reqs = []
    load_scheduled_new_reqs = []

    for new_req in scheduler_output.scheduled_new_reqs:
        # v1调度不区分prefill和deocde，会把token_budget填满，这会导致请求被截断，一次step只计算部分token
        num_total_blocks = (
            scheduler_output.num_scheduled_tokens[new_req.req_id] +
            new_req.num_computed_tokens - 1) // block_size
        num_computed_blocks = new_req.num_computed_tokens // block_size

        logger.info(
            f"[EMS] Load req {new_req.req_id}, block_ids: {new_req.block_ids}, "
            f"prompt len: {len(new_req.prompt_token_ids)},"
            f"num_scheduled_tokens: {scheduler_output.num_scheduled_tokens[new_req.req_id]}, "
            f"num_computed_tokens: {new_req.num_computed_tokens}, "
            f"num_total_blocks: {num_total_blocks}, num_computed_blocks: {num_computed_blocks}"
        )

        if num_computed_blocks >= num_total_blocks:
            continue

        req_hash_blocks = cal_hash_blocks(new_req.prompt_token_ids, block_size)
        hash_blocks = req_hash_blocks[num_computed_blocks:num_total_blocks]
        block_ids = new_req.block_ids[0][num_computed_blocks:num_total_blocks]

        info_load_reqs.append((new_req.req_id, hash_blocks, block_ids))
        load_scheduled_new_reqs.append(new_req)

    return info_load_reqs, load_scheduled_new_reqs
