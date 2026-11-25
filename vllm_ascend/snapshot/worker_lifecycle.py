# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import platform
import sys
import time
from ctypes import CDLL, c_int, c_void_p

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import logger
from vllm.utils.network_utils import get_distributed_init_method, get_ip

from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel
from vllm_ascend.snapshot.distributed import cleanup_dist_env_for_snapshot, snapshot_hccl_teardown
from vllm_ascend.snapshot.model_runner_lifecycle.checkpoint import dump_model_runner
from vllm_ascend.snapshot.model_runner_lifecycle.restore import restore_model_runner

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
    if result != 0:
        logger.error(
            "[snapshot][worker] runtime API failed: rank=%s api=%s status=%s",
            worker.rank,
            api_name,
            result,
        )
        raise RuntimeError(f"Snapshot runtime API failed: api={api_name} status={result}")


def _run_timed_steps(worker, operation: str, steps) -> None:
    operation_start = time.perf_counter()
    logger.info(
        "[snapshot][worker] %s started: rank=%s",
        operation,
        worker.rank,
    )
    for _, step_fn in steps:
        step_fn()
    logger.info(
        "[snapshot][worker] %s completed: rank=%s duration=%.2f s",
        operation,
        worker.rank,
        time.perf_counter() - operation_start,
    )


def suspend_worker(worker, model_save_path: str | None = None) -> None:
    if worker.vllm_config.parallel_config.pipeline_parallel_size > 1:
        from vllm_ascend.snapshot.pipeline import prepare_pipeline_for_snapshot

        prepare_pipeline_for_snapshot(worker)
    steps = (
        ("dump_model_checkpoint", lambda: dump_model_runner(worker.model_runner, model_save_path)),
        ("synchronize_npu", torch.npu.synchronize),
        ("collect_garbage", gc.collect),
        ("lock_snapshot_process", lambda: _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessLock")),
        ("snapshot_process_backup", lambda: _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessBackup")),
    )
    _run_timed_steps(worker, "suspend", steps)


def unlock_worker(worker) -> None:
    _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessUnlock")


def resume_worker(
    worker,
    data_parallel_master_ip: str,
    model_path: str | None = None,
    new_engine_id: str | None = None,
) -> None:
    # Resolve on each worker's node, bypassing the checkpoint's VLLM_HOST_IP.
    local_ip = get_ip(force=True)
    if local_ip in ("", "0.0.0.0", "::"):
        raise RuntimeError("Unable to resolve local worker IP for snapshot restore")
    steps = (
        ("restore_snapshot_process", lambda: _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessRestore")),
        ("unlock_snapshot_process", lambda: _call_aclrt_snapshot_api(worker, "aclrtSnapShotProcessUnlock")),
        ("reset_triton_kernel_caches", _reset_triton_kernel_caches),
        (
            "update_worker_network",
            lambda: _update_worker_info(worker, local_ip, data_parallel_master_ip),
        ),
        # Old HCCL QPs must release their references before KV transport frees
        # the shared RDMA context. Finish both teardowns before creating groups.
        ("destroy_parallel_groups", lambda: _parallel_group_cleanup(worker)),
        ("destroy_kv_transfer_endpoint", lambda: _prepare_kv_transfer_for_snapshot_restore(worker)),
        ("rebuild_parallel_groups", lambda: _rebuild_parallel_groups(worker)),
        ("restore_model_checkpoint", lambda: restore_model_runner(worker.model_runner, model_path)),
        ("recapture_graph", lambda: _recapture_graph(worker)),
        (
            "rebuild_kv_transfer_endpoint",
            lambda: _rebuild_kv_transfer_engine(worker, local_ip, new_engine_id),
        ),
    )
    _run_timed_steps(worker, "resume", steps)


def _reset_triton_kernel_caches() -> None:
    """Discard restored Triton launchers that may retain stale NPU handles."""
    from triton.runtime import Autotuner, Heuristics, JITFunction

    seen: set[int] = set()

    def reset_kernel(kernel) -> None:
        if id(kernel) in seen:
            return
        seen.add(id(kernel))
        if isinstance(kernel, JITFunction):
            # Triton Ascend 3.2.2 uses cache; newer runtimes use device_caches.
            # Clear in place to preserve the defaultdict's device-cache factory.
            if hasattr(kernel, "device_caches"):
                kernel.device_caches.clear()
            elif hasattr(kernel, "cache"):
                kernel.cache.clear()
            else:
                raise RuntimeError("Unsupported Triton JIT cache layout during snapshot restore")
        elif isinstance(kernel, Autotuner):
            kernel.cache.clear()
            reset_kernel(kernel.fn)
        elif isinstance(kernel, Heuristics):
            reset_kernel(kernel.fn)

    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith(("vllm_ascend.", "vllm.")) or module is None:
            continue
        for obj in vars(module).values():
            reset_kernel(obj)


def _parallel_group_cleanup(worker) -> None:
    snapshot_enabled = worker.vllm_config.snapshot_config is not None
    with snapshot_hccl_teardown(snapshot_enabled):
        if worker.vllm_config.parallel_config.pipeline_parallel_size > 1:
            from vllm_ascend.snapshot.pipeline import destroy_pipeline_runtime

            destroy_pipeline_runtime(worker)
        destroy_ascend_model_parallel()
        cleanup_dist_env_for_snapshot()


def _rebuild_parallel_groups(worker) -> None:
    import torch.distributed as dist

    # DEBUG level triggers a known torchair bug, so keep INFO level.
    dist.set_debug_level(dist.DebugLevel.INFO)

    rebuild_time_start = time.time()

    parallel_config = worker.vllm_config.parallel_config
    master_ip = parallel_config.data_parallel_master_ip
    # Independent single-node DP replicas have a local TP/PP rank zero.
    if parallel_config.data_parallel_size == 1 and parallel_config.nnodes == 1:
        master_ip = get_ip(force=True)
    if not master_ip or master_ip in ("0.0.0.0", "::"):
        raise RuntimeError("Unable to resolve master IP for distributed init method")
    resume_ports = worker.vllm_config.parallel_config._snapshot_data_parallel_port_list
    if not resume_ports:
        raise RuntimeError("Snapshot world-group resume port is not configured")
    worker.distributed_init_method = get_distributed_init_method(master_ip, resume_ports[-1])
    logger.info(
        "[snapshot][port] rebuilding worker global distributed group: master=%s:%d",
        master_ip,
        resume_ports[-1],
    )

    with set_current_vllm_config(worker.vllm_config):
        worker._init_worker_distributed_environment()

        from vllm.distributed.parallel_state import get_dp_group, get_ep_group

        from vllm_ascend.distributed.parallel_state import get_mc2_group
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        moe_comm_methods_and_dispatchers = []
        for comm_method in _MoECommMethods.values():
            moe_config = getattr(comm_method, "moe_config", None)
            if moe_config is not None:
                moe_config.tp_group = get_tp_group()
                moe_config.dp_group = get_dp_group()
                if moe_config.ep_size > 1:
                    moe_config.ep_group = get_ep_group()
                    moe_config.mc2_group = get_mc2_group()

            dispatcher = getattr(comm_method, "token_dispatcher", None)
            moe_comm_methods_and_dispatchers.extend((comm_method, dispatcher))
            refresh_fn = getattr(dispatcher, "refresh_hccl_group", None)
            if callable(refresh_fn):
                refresh_fn()

        reset_ids: set[int] = set()
        for comm_method_or_dispatcher in moe_comm_methods_and_dispatchers:
            if comm_method_or_dispatcher is None or id(comm_method_or_dispatcher) in reset_ids:
                continue
            reset_ids.add(id(comm_method_or_dispatcher))
            reset_state = getattr(comm_method_or_dispatcher, "reset_runtime_state_after_snapshot_restore", None)
            if callable(reset_state):
                reset_state()
    logger.info(
        "[snapshot][parallel] group rebuild completed: rank=%s duration=%.2f s",
        worker.rank,
        time.time() - rebuild_time_start,
    )


def _update_worker_info(worker, local_ip: str, data_parallel_master_ip: str) -> None:
    os.environ["HCCL_IF_IP"] = local_ip
    worker.vllm_config.parallel_config.data_parallel_master_ip = data_parallel_master_ip
    logger.info(
        "[snapshot][worker] network configuration updated: rank=%s local_ip=%s data_parallel_master_ip=%s",
        worker.rank,
        local_ip,
        data_parallel_master_ip,
    )


def _prepare_kv_transfer_for_snapshot_restore(worker) -> None:
    kv_cfg = worker.vllm_config.kv_transfer_config
    if kv_cfg is None or not (kv_cfg.is_kv_producer or kv_cfg.is_kv_consumer):
        return
    if has_kv_transfer_group():
        get_kv_transfer_group().prepare_for_snapshot_restore()


def _rebuild_kv_transfer_engine(worker, local_ip: str, new_engine_id: str | None = None) -> None:
    kv_cfg = worker.vllm_config.kv_transfer_config
    if kv_cfg is None:
        return
    if not (getattr(kv_cfg, "is_kv_producer", False) or getattr(kv_cfg, "is_kv_consumer", False)):
        return
    if not has_kv_transfer_group():
        return
    get_kv_transfer_group().rebuild_kv_transfer_endpoint(local_ip, new_engine_id)


def _recapture_graph(worker) -> None:
    if worker.model_config.enforce_eager:
        logger.debug(
            "[snapshot][worker] graph recapture skipped: rank=%s reason=enforce_eager",
            worker.rank,
        )
        return

    from vllm_ascend.compilation.acl_graph import clear_all_aclgraph_entries, clear_graph_params_for_recapture

    clear_all_aclgraph_entries()
    clear_graph_params_for_recapture()
    from vllm_ascend.snapshot.model_runner_lifecycle.restore import reset_graph_managers

    reset_graph_managers(worker.model_runner)
    worker.model_runner.capture_model()
