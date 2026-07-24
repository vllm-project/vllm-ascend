import datetime
import itertools
from typing import Any

from vllm.v1.request import RequestStatus


def diagnostics_enabled(vllm_config: Any) -> bool:
    """Read the diagnostic toggle independently of NonBSP activation."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return False

    scheduler_config = additional_config.get("scheduler_config")
    if not isinstance(scheduler_config, dict):
        return False

    nonbsp_config = scheduler_config.get("nonbsp_config")
    if not isinstance(nonbsp_config, dict):
        return False

    return nonbsp_config.get("enable_diagnostics") is True


def print_scheduler_summary(scheduler: Any, scheduler_output: Any) -> None:
    waiting_reqs = list(itertools.chain(scheduler.waiting, scheduler.skipped_waiting))
    lb_paused_req_ids: set[str] = getattr(scheduler, "_lb_paused_req_ids", set())

    waiting_num = sum(
        request.status == RequestStatus.WAITING or request.request_id in lb_paused_req_ids for request in waiting_reqs
    )
    waiting_for_remote_kvs_num = sum(request.status == RequestStatus.WAITING_FOR_REMOTE_KVS for request in waiting_reqs)
    structured_output_waiting_status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    waiting_for_fsm_num = sum(request.status == structured_output_waiting_status for request in waiting_reqs)
    preempted_num = sum(
        request.status == RequestStatus.PREEMPTED and request.request_id not in lb_paused_req_ids
        for request in waiting_reqs
    )
    running_block_num = sum(
        (len(request.all_token_ids) + scheduler.block_size - 1) // scheduler.block_size for request in scheduler.running
    )
    waiting_block_num = sum(
        (len(request.all_token_ids) + scheduler.block_size - 1) // scheduler.block_size
        for request in waiting_reqs
        if request.status == RequestStatus.WAITING
    )
    waiting_for_remote_kvs_block_num = sum(
        (len(request.all_token_ids) + scheduler.block_size - 1) // scheduler.block_size
        for request in waiting_reqs
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    print(
        f"{datetime.datetime.now()} | schedule() | "
        "scheduler req num: "
        f"[{len(scheduler_output.num_scheduled_tokens)}, "
        f"{len(waiting_reqs)}, {waiting_num}, "
        f"{waiting_for_remote_kvs_num}, {waiting_for_fsm_num}, "
        f"{preempted_num}] | blk num "
        f"[{running_block_num}, {waiting_block_num}, "
        f"{waiting_for_remote_kvs_block_num}]",
        flush=True,
    )
