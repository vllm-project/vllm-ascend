# import pickle
# from functools import partial

# import cloudpickle
# import zmq
# from vllm.logger import logger
# from vllm.v1.executor.multiproc_executor import WorkerProc
# from vllm.v1.utils import get_engine_client_zmq_addr
# from vllm.utils.network_utils import make_zmq_socket
# from vllm_ascend.recovery import RecoveryMonitor


# _original_worker_proc_init = WorkerProc.__init__


# def _patched_worker_proc_init(
#     self,
#     vllm_config,
#     local_rank,
#     rank,
#     distributed_init_method,
#     input_shm_handle,
#     shared_worker_lock,
#     is_driver_worker,
#     recovery_addrs=None,
# ):
#     main_monitor_addr = get_engine_client_zmq_addr(True, "127.0.0.1")
#     self._recovery_push_sock = make_zmq_socket(
#         zmq.Context(), main_monitor_addr, zmq.PUSH, bind=False,
#     )

#     _original_worker_proc_init(
#         self,
#         vllm_config,
#         local_rank,
#         rank,
#         distributed_init_method,
#         input_shm_handle,
#         shared_worker_lock,
#         is_driver_worker,
#     )

#     recover_step_sub_addr = recovery_addrs.get("recover_step_pub_addr") if recovery_addrs else None
#     recover_report_push_addr = recovery_addrs.get("recover_report_pull_addr") if recovery_addrs else None

#     self._recovery_monitor = RecoveryMonitor(
#         worker_rank=rank,
#         local_rank=local_rank,
#         main_monitor_addr=main_monitor_addr,
#         recover_step_sub_addr=recover_step_sub_addr,
#         recover_report_push_addr=recover_report_push_addr,
#     )
#     self._recovery_monitor.start()

#     logger.info(
#         "[WorkerProc] RecoveryMonitor started for rank %d local_rank=%d (main_monitor=%s, recover_step=%s, recover_report=%s)",
#         rank, local_rank, main_monitor_addr, recover_step_sub_addr, recover_report_push_addr,
#     )


# WorkerProc.__init__ = _patched_worker_proc_init
