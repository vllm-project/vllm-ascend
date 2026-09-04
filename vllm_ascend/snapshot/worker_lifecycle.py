# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import platform
import time
from ctypes import CDLL, c_int, c_void_p

import torch
import torch_npu
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.utils.network_utils import get_distributed_init_method

from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel
from vllm_ascend.snapshot.distributed import cleanup_dist_env_for_snapshot, snapshot_hccl_teardown
from vllm_ascend.snapshot.model_restore import (
    dump_model_runner,
    restore_drafter_runtime_buffers,
    restore_model_runner,
)
from vllm_ascend.snapshot.tensor_state import reset_runtime_tensor_state
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

_ACL_RT_LIB: CDLL | None = None


def _get_acl_rt_lib() -> CDLL:
    global _ACL_RT_LIB
    if _ACL_RT_LIB is not None:
        return _ACL_RT_LIB
    try:
        _ACL_RT_LIB = CDLL("libacl_rt.so")
    except OSError:
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")
        arch = "aarch64" if platform.machine() == "aarch64" else "x86_64"
        lib_path = os.path.join(ascend_home, f"{arch}-linux", "lib64", "libacl_rt.so")
        _ACL_RT_LIB = CDLL(lib_path)
    return _ACL_RT_LIB


def _call_aclrt_snapshot_api(worker, api_name: str) -> None:
    api = getattr(_get_acl_rt_lib(), api_name)
    api.argtypes = [c_int, c_void_p]
    api.restype = c_int
    result = api(os.getpid(), None)
    if result == 0:
        logger.info("[snapshot] [worker] [rank:%s] %s success.", worker.rank, api_name)
    else:
        logger.error("[snapshot] [worker] [rank:%s] %s failed %s.", worker.rank, api_name, result)


def _run_timed_steps(worker, steps) -> None:
    for step_name, step_fn in steps:
        logger.info("[snapshot] [worker] rank %s: start %s", worker.rank, step_name)
        start = time.perf_counter()
        step_fn()
        logger.info(
            "[snapshot] [worker] rank %s: %s cost %.2fs",
            worker.rank,
            step_name,
            time.perf_counter() - start,
        )


def suspend_worker(worker, model_save_path: str | None = None) -> None:
    steps = (
        ("dump_model", lambda: dump_model_runner(worker.model_runner, model_save_path)),
        ("gc.collect", gc.collect),
        ("snapshot_process_lock", lambda: _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessLock")),
        ("snapshot_process_backup", lambda: _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessBackup")),
    )
    _run_timed_steps(worker, steps)


def unlock_worker(worker) -> None:
    _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessUnlock")


def resume_worker(
    worker,
    local_ip: str,
    data_parallel_master_ip: str,
    model_path: str | None = None,
    new_engine_id: str | None = None,
) -> None:
    steps = (
        ("snapshot_process_restore", lambda: _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessRestore")),
        ("snapshot_process_unlock", lambda: _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessUnlock")),
        (
            "update_worker_info_after_resume",
            lambda: _update_worker_info(worker, local_ip, data_parallel_master_ip),
        ),
        ("rebuild_parallel_group_after_resume", lambda: _rebuild_parallel_groups(worker)),
        ("re_load_weights", lambda: restore_model_runner(worker.model_runner, model_path)),
        ("recapture_graph", lambda: _recapture_graph(worker)),
        (
            "rebuild_kv_transfer_engine_after_resume",
            lambda: _rebuild_kv_transfer_engine(worker, local_ip, new_engine_id),
        ),
    )
    _run_timed_steps(worker, steps)


def _parallel_group_cleanup(worker) -> None:
    snapshot_enabled = worker.vllm_config.snapshot_config is not None
    with snapshot_hccl_teardown(snapshot_enabled):
        destroy_ascend_model_parallel()
        logger.info("[snapshot] [parallel] rank %s: destroy_ascend_model_parallel done", worker.rank)
        cleanup_dist_env_for_snapshot()
        logger.info("[snapshot] [parallel] rank %s: cleanup_dist_env_for_snapshot done", worker.rank)


def _rebuild_parallel_groups(worker) -> None:
    import torch.distributed as dist

    # DEBUG level triggers a known torchair bug, so keep INFO level.
    dist.set_debug_level(dist.DebugLevel.INFO)

    rebuild_time_start = time.time()
    logger.info("[snapshot] [parallel] rank %s: destroying HCCL and model-parallel groups", worker.rank)
    _parallel_group_cleanup(worker)
    logger.info("[snapshot] [parallel] rank %s: rebuilding HCCL and model-parallel groups", worker.rank)

    master_ip = worker.vllm_config.parallel_config.data_parallel_master_ip
    if not master_ip:
        raise RuntimeError("Unable to resolve master IP for distributed init method")
    resume_ports = worker.vllm_config.parallel_config._snapshot_data_parallel_port_list
    if not resume_ports:
        raise RuntimeError("Snapshot world-group resume port is not configured")
    worker.distributed_init_method = get_distributed_init_method(master_ip, resume_ports[-1])

    with set_current_vllm_config(worker.vllm_config):
        worker._init_worker_distributed_environment()

        from vllm.distributed.parallel_state import get_dp_group, get_ep_group

        from vllm_ascend.distributed.parallel_state import get_mc2_group
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        snapshot_state_owners = []
        for comm_method in _MoECommMethods.values():
            moe_config = getattr(comm_method, "moe_config", None)
            if moe_config is not None:
                moe_config.tp_group = get_tp_group()
                moe_config.dp_group = get_dp_group()
                if moe_config.ep_size > 1:
                    moe_config.ep_group = get_ep_group()
                    moe_config.mc2_group = get_mc2_group()

            dispatcher = getattr(comm_method, "token_dispatcher", None)
            snapshot_state_owners.extend((comm_method, dispatcher))
            refresh_fn = getattr(dispatcher, "refresh_hccl_group", None)
            if callable(refresh_fn):
                refresh_fn()

        reset_runtime_tensor_state(snapshot_state_owners)
        logger.info("[snapshot] [parallel] rank %s: refreshed cached MoE parallel and HCCL groups", worker.rank)

    logger.info(
        "[snapshot] [parallel] rank %s: rebuild_parallel_group cost %.2fs",
        worker.rank,
        time.time() - rebuild_time_start,
    )


def _update_worker_info(worker, local_ip: str, data_parallel_master_ip: str) -> None:
    os.environ["HCCL_IF_IP"] = local_ip
    worker.vllm_config.parallel_config.data_parallel_master_ip = data_parallel_master_ip
    logger.info(
        "[snapshot] [worker] rank %s: HCCL_IF_IP=%s data_parallel_master_ip=%s",
        worker.rank,
        local_ip,
        data_parallel_master_ip,
    )


def _rebuild_kv_transfer_engine(worker, local_ip: str, new_engine_id: str | None = None) -> None:
    kv_cfg = worker.vllm_config.kv_transfer_config
    if kv_cfg is None:
        return
    if not (getattr(kv_cfg, "is_kv_producer", False) or getattr(kv_cfg, "is_kv_consumer", False)):
        return
    if not has_kv_transfer_group():
        return
    rebuild = getattr(
        getattr(get_kv_transfer_group(), "connector_worker", None),
        "rebuild_kv_transfer_endpoint",
        None,
    )
    if callable(rebuild):
        rebuild(local_ip, new_engine_id)


def _recapture_graph(worker) -> None:
    if worker.model_config.enforce_eager:
        logger.info("[snapshot][worker] rank %s: enforce_eager is True, skip recapture graph", worker.rank)
        return

    from vllm_ascend.compilation.acl_graph import clear_all_aclgraph_entries, clear_graph_params_for_recapture

    clear_all_aclgraph_entries()
    clear_graph_params_for_recapture()
    worker.model_runner.capture_model()
    restore_drafter_runtime_buffers(worker.model_runner)

    if get_ascend_device_type() != AscendDeviceType.A5:
        _warm_up_atb()


def _warm_up_atb() -> None:
    x = torch.rand((2, 4), dtype=torch.float16).npu()
    weight = torch.rand((2, 4), dtype=torch.float16).npu()
    c = torch.rand((4, 4), dtype=torch.float32).npu()
    torch_npu._npu_matmul_add_fp32(x, weight, c)
