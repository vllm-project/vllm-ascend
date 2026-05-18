# from multiprocessing import get_context

# from vllm.v1.executor.multiproc_executor import UnreadyWorkerProcHandle, WorkerProc
# from vllm.logger import logger

# from vllm_ascend.patch.platform.patch_multiproc_executor import AscendWorkerProc


# _original_worker_make_worker_process = WorkerProc.make_worker_process


# def _patched_worker_make_worker_process(
#     vllm_config,
#     local_rank,
#     rank,
#     distributed_init_method,
#     input_shm_handle,
#     shared_worker_lock,
#     is_driver_worker,
#     inherited_fds=None,
# ):
#     context = get_context()
#     ready_reader, ready_writer = context.Pipe(duplex=False)
#     death_reader, death_writer = context.Pipe(duplex=False)
#     if inherited_fds is not None:
#         inherited_fds = inherited_fds.copy()
#         inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))

#     recovery_addrs = getattr(WorkerProc, "_recovery_addrs_for_spawn", None)
#     logger.info("[RAS] enginecore<->worker: recovery_addrs=%s", recovery_addrs)

#     process_kwargs = {
#         "vllm_config": vllm_config,
#         "local_rank": local_rank,
#         "rank": rank,
#         "distributed_init_method": distributed_init_method,
#         "input_shm_handle": input_shm_handle,
#         "ready_pipe": ready_writer,
#         "death_pipe": death_reader,
#         "shared_worker_lock": shared_worker_lock,
#         "is_driver_worker": is_driver_worker,
#         "inherited_fds": inherited_fds if inherited_fds is not None else [],
#         "recovery_addrs": recovery_addrs,
#     }
#     proc = context.Process(
#         target=WorkerProc.worker_main,
#         kwargs=process_kwargs,
#         name=f"VllmWorker-{rank}",
#         daemon=True,
#     )
#     proc.start()
#     ready_writer.close()
#     death_reader.close()
#     return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)


# WorkerProc.make_worker_process = staticmethod(_patched_worker_make_worker_process)


# _original_ascend_make_worker_process = AscendWorkerProc.make_worker_process


# def _patched_ascend_make_worker_process(
#     vllm_config,
#     local_rank,
#     rank,
#     distributed_init_method,
#     input_shm_handle,
#     shared_worker_lock,
#     is_driver_worker,
#     inherited_fds=None,
# ):
#     context = get_context()
#     ready_reader, ready_writer = context.Pipe(duplex=False)
#     death_reader, death_writer = context.Pipe(duplex=False)
#     if inherited_fds is not None:
#         inherited_fds = inherited_fds.copy()
#         inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))

#     recovery_addrs = getattr(WorkerProc, "_recovery_addrs_for_spawn", None)
#     logger.info("[RAS] enginecore<->worker: recovery_addrs=%s", recovery_addrs)

#     process_kwargs = {
#         "vllm_config": vllm_config,
#         "local_rank": local_rank,
#         "rank": rank,
#         "distributed_init_method": distributed_init_method,
#         "input_shm_handle": input_shm_handle,
#         "ready_pipe": ready_writer,
#         "death_pipe": death_reader,
#         "shared_worker_lock": shared_worker_lock,
#         "is_driver_worker": is_driver_worker,
#         "inherited_fds": inherited_fds if inherited_fds is not None else [],
#         "recovery_addrs": recovery_addrs,
#     }
#     proc = context.Process(
#         target=WorkerProc.worker_main,
#         kwargs=process_kwargs,
#         name=f"VllmWorker-{rank}",
#         daemon=True,
#     )
#     proc.start()
#     ready_writer.close()
#     death_reader.close()
#     return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)


# AscendWorkerProc.make_worker_process = staticmethod(_patched_ascend_make_worker_process)
