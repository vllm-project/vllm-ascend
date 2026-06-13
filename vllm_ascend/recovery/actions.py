import time
from typing import Any, Callable, Tuple

import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request, RequestStatus
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.platform import NPUPlatform

ActionFunc = Callable[[Any, dict], Tuple[dict, bool]]

_ENGINE_CORE_ACTIONS: dict[str, ActionFunc] = {}
_WORKER_ACTIONS: dict[str, ActionFunc] = {}


def engine_core_action(name: str):
    def wrapper(func: ActionFunc):
        _ENGINE_CORE_ACTIONS[name] = func
        return func
    return wrapper


def worker_action(name: str):
    def wrapper(func: ActionFunc):
        _WORKER_ACTIONS[name] = func
        return func
    return wrapper


def get_engine_core_action(name: str) -> ActionFunc:
    func = _ENGINE_CORE_ACTIONS.get(name)
    if func is None:
        raise ValueError(f"Unknown engine_core action: {name}")
    return func


def get_worker_action(name: str) -> ActionFunc:
    func = _WORKER_ACTIONS.get(name)
    if func is None:
        raise ValueError(f"Unknown worker action: {name}")
    return func


@engine_core_action("label_dirty_requests")
def _label_dirty_requests(executer: Any, cfg: dict) -> Tuple[dict, bool]:
    scheduler = executer.scheduler

    scheduled_req_ids: set[str] = set()
    if executer.batch_queue is not None:
        for _, scheduler_output, _ in executer.batch_queue:
            scheduled_req_ids.update(scheduler_output.num_scheduled_tokens.keys())

    # Also collect from last_scheduler_output which may not be in batch_queue yet
    if hasattr(executer, "last_scheduler_output") and executer.last_scheduler_output is not None:
        scheduled_req_ids.update(executer.last_scheduler_output.num_scheduled_tokens.keys())

    dirty_req_ids = {req.request_id for req in scheduler.running if not req.is_finished()}

    waiting_req_ids = {req.request_id for req in scheduler.waiting}
    waiting_dirty_req_ids = scheduled_req_ids & waiting_req_ids
    
    logger.info(
        "[dp_rank=%d] label_dirty_requests: %d running dirty, %d waiting dirty (scheduled=%d)",
        executer.dp_rank,
        len(dirty_req_ids),
        len(waiting_dirty_req_ids),
        len(scheduled_req_ids),
    )

    cfg["dirty_requests_list"] = list(dirty_req_ids | waiting_dirty_req_ids)
    cfg["waiting_dirty_requests_list"] = list(waiting_dirty_req_ids)
    return cfg, True


@engine_core_action("clean_batch_queue")
def _clean_batch_queue(executer: Any, cfg: dict) -> Tuple[dict, bool]:
    if executer.batch_queue is None:
        logger.info("[dp_rank=%d] batch_queue is None, skip clean", executer.dp_rank)
        return cfg, True

    while executer.batch_queue:
        future, _, _ = executer.batch_queue.pop()
        try:
            future.result()
        except Exception:
            pass
    logger.info("[dp_rank=%d] batch_queue drained", executer.dp_rank)
    return cfg, True

def _collect_request_block_ids(scheduler, req_id: str) -> set[int]:
    old_blocks = scheduler.kv_cache_manager.coordinator.get_blocks(req_id)
    block_ids: set[int] = set()
    for blocks_list in old_blocks:
        for block in blocks_list:
            block_ids.add(block.block_id)
    return block_ids


def _rebuild_request(executer, request: Request) -> Request:
    new_engine_core_request = EngineCoreRequest(
        request_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,
        pooling_params=request.pooling_params,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
        data_parallel_rank=None,
        prompt_embeds=request.prompt_embeds,
        client_index=request.client_index,
        priority=request.priority,
        trace_headers=request.trace_headers,
        resumable=request.resumable,
        reasoning_ended=None,
    )
    new_request, _wave = executer.preprocess_add_request(new_engine_core_request)
    executer.scheduler.requests[new_request.request_id] = new_request
    executer.scheduler.waiting.prepend_request(new_request)
    executer.scheduler.prev_step_scheduled_req_ids.discard(request.request_id)
    return new_request


@engine_core_action("recompute_dirty_requests")
def _recompute_dirty_requests(executer: Any, cfg: dict) -> Tuple[dict, bool]:
    dirty_req_ids = cfg.get("dirty_requests_list", [])
    waiting_dirty_req_ids = set(cfg.get("waiting_dirty_requests_list", []))

    logger.info(
        "[dp_rank=%d] recompute_dirty_requests: %d total dirty (%d from waiting)",
        executer.dp_rank,
        len(dirty_req_ids),
        len(waiting_dirty_req_ids),
    )

    scheduler = executer.scheduler
    all_dirty_block_ids: set[int] = set()
    running_to_remove: set[Request] = set()
    waiting_to_remove: list[Request] = []

    for req_id in dirty_req_ids:
        request = scheduler.requests.get(req_id)
        if request is None:
            continue
        all_dirty_block_ids |= _collect_request_block_ids(scheduler, req_id)
        scheduler.kv_cache_manager.free(request)
        scheduler.encoder_cache_manager.free(request)
        _rebuild_request(executer, request)

        if req_id in waiting_dirty_req_ids:
            waiting_to_remove.append(request)
        elif req_id in scheduler.running:
            running_to_remove.add(request)
        else:
            logger.warning(f"Request {req_id} is not found in running or waiting, skip")


    scheduler.running = remove_all(scheduler.running, running_to_remove)
    scheduler.waiting.remove_requests(waiting_to_remove)
    scheduler.kv_cache_manager.evict_blocks(all_dirty_block_ids)

    logger.info(
        "[dp_rank=%d] recompute_dirty_requests done, %d running + %d waiting reprocessed",
        executer.dp_rank,
        len(running_to_remove),
        len(waiting_to_remove),
    )
    return cfg, True

@worker_action("stop_device")
def _stop_device(executor: Any, cfg: dict | None) -> bool:
    try:
        NPUPlatform.set_device(executor.device)
        stop_result = torch_npu.npu.stop_device(executor.device.index)
        if stop_result != 0:
            logger.error(f"stop_device failed with result: {stop_result}")
            return cfg, False
        logger.info("stop_device executed successfully")
        return cfg, True
    except Exception as e:
        logger.error(f"stop_device executed failed with exception: {e}")
        return cfg, False

@worker_action("restart_device")
def _restart_device(executor: Any, cfg:dict | None) -> bool:
    from vllm_ascend.ascend_config import get_ascend_config

    # Wait for exception_occur before restarting device
    max_wait = get_ascend_config().recovery_config.cpu_process_group_timeout
    waited = 0
    while not executor.exception_occur and waited <= max_wait:
        logger.warning("restart_device waiting for exception_occur")
        time.sleep(1)
        waited += 1
    if not executor.exception_occur:
        logger.error(
            "restart_device timed out after %ds waiting for exception_occur",
            max_wait,
        )
        return cfg, False

    try:
        cfg = cfg or {}
        NPUPlatform.set_device(executor.device)
        rebuild_all_resources = cfg.get("restart_device_rebuild_all_resources", False)
        logger.info(
            "restart_device execute with rebuild_all_resources: %s", 
            rebuild_all_resources
        )
        torch_npu.npu.restart_device(
            executor.device.index, rebuild_all_resources=rebuild_all_resources
        )
        logger.info("restart_device executed successfully")
        return cfg, True
    except Exception as e:
        logger.error(f"restart_device executed failed with exception: {e}")
        return cfg, False


@worker_action("reinit_process_group")
def _reinit_process_group(executor: Any, cfg:dict | None) -> bool:
    try:
        cfg = cfg or {}
        NPUPlatform.set_device(executor.device)
        rebuild_link = cfg.get("reinit_process_group_rebuild_link", False)
        logger.info(
            "reinit_process_group execute with rebuild_link: %s", 
            rebuild_link
        )
        torch.distributed.reinit_process_group(
            group=cfg.get("group", None), rebuild_link=rebuild_link 
        )
        logger.info("reinit_process_group executed successfully")
        return cfg, True
    except Exception as e:
        logger.error(f"reinit_process_group executed failed with exception: {e}")
        return cfg, False


@worker_action("worker_clean_dirty_requests_cache")
def _worker_clean_dirty_requests_cache(executor: Any, cfg:dict | None) -> bool:
    try:
        cfg = cfg or {}
        model_runner = executor.model_runner
        dirty_requests_list = cfg.get("dirty_requests_list", [])
        logger.info(
            "worker clean_dirty_requests_cache execute with dirty_requests_list: %s", 
            dirty_requests_list
        )
        for req_id in dirty_requests_list:
            model_runner.requests.pop(req_id, None)
            model_runner.num_prompt_logprobs.pop(req_id, None)
            model_runner.input_batch.remove_request(req_id)
        model_runner.input_batch.condense()
        model_runner.input_batch.refresh_metadata()
        return cfg, True
    except Exception as e:
        logger.error(f"worker clean_dirty_requests_cache failed with exception: {e}")
        return cfg, False


@worker_action("worker_rebuild_cpu_group")
def _worker_rebuild_cpu_group(executor: Any, cfg:dict | None) -> bool:
    try:
        from vllm.distributed.parallel_state import get_dp_group, get_pp_group
        try:
            get_dp_group().reinit_cpu_group()
        except AssertionError:
            pass
        try:
            get_pp_group().reinit_cpu_group()
        except AssertionError:
            pass
        logger.info("worker rebuild_cpu_group executed successfully")
        return cfg, True
    except Exception as e:
        logger.error(f"worker rebuild_cpu_group failed with exception: {e}")
        return cfg, False

@worker_action("worker_recapture_graph")
def _worker_recapture_graph(executor: Any, cfg:dict | None) -> bool:
    try:
        NPUPlatform.set_device(executor.device)
        model_runner = executor.model_runner
        ACLGraphWrapper.label_reset_all_graphs(reset_graph_pool=True)
        model_runner.capture_model()
        logger.info("worker recapture_graph executed successfully")
        return cfg, True
    except Exception as e:
        logger.error(f"worker recapture_graph failed with exception: {e}")
        return cfg, False

@worker_action("recovery_begin")
def _recovery_begin(executor: Any, cfg:dict | None) -> bool:
    executor.in_recovery = True
    logger.info("recovery_begin executed successfully")
    return cfg, True

@worker_action("recovery_finished")
def _recovery_finished(executor: Any, cfg: dict | None) -> bool:
    executor.in_recovery = False
    executor.exception_occur = False
    logger.info("recovery_finished executed successfully")
    return cfg, True