import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.synchronize import Lock as LockType
from typing import Optional

import vllm.v1.executor.multiproc_executor
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.utils import (get_distributed_init_method, get_loopback_ip,
                        get_mp_context, get_open_port)
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor, UnreadyWorkerProcHandle, WorkerProc,
    set_multiprocessing_worker_envs)


class AscendMultiprocExecutor(MultiprocExecutor):
    supports_pp: bool = True

    def _init_executor(self) -> None:
        # Call self.shutdown at exit to clean up
        # and ensure workers will be terminated.
        self._finalizer = weakref.finalize(self, self.shutdown)
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None

        self.world_size = self.parallel_config.world_size
        tensor_parallel_size = self.parallel_config.tensor_parallel_size
        pp_parallel_size = self.parallel_config.pipeline_parallel_size
        assert self.world_size == tensor_parallel_size * pp_parallel_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tensor_parallel_size}) x pipeline"
            f"_parallel_size ({pp_parallel_size}). ")

        # Set multiprocessing envs
        set_multiprocessing_worker_envs()

        # Multiprocessing-based executor does not support multi-node setting.
        # Since it only works for single node, we can use the loopback address
        # get_loopback_ip() for communication.
        distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())

        # Initialize worker and set up message queues for SchedulerOutputs
        # and ModelRunnerOutputs
        max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
        self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                             self.world_size,
                                             max_chunk_bytes=max_chunk_bytes)
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()

        # Create workers
        context = get_mp_context()
        shared_worker_lock = context.Lock()
        unready_workers: list[UnreadyWorkerProcHandle] = []
        success = False
        try:
            for rank in range(self.world_size):
                unready_workers.append(
                    AscendWorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                    ))

            # Workers must be created before wait_for_ready to avoid
            # deadlock, since worker.init_device() does a device sync.
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            # Ensure message queues are ready. Will deadlock if re-ordered
            # Must be kept consistent with the WorkerProc.
            self.rpc_broadcast_mq.wait_until_ready()
            for w in self.workers:
                w.worker_response_mq.wait_until_ready()

            self.start_worker_monitor()
            success = True
        finally:
            if not success:
                # Clean up the worker procs if there was a failure.
                # Close death_writers first to signal workers to exit
                for uw in unready_workers:
                    if uw.death_writer is not None:
                        uw.death_writer.close()
                self._ensure_worker_termination(
                    [uw.proc for uw in unready_workers])

        # For pipeline parallel, we use a thread pool for asynchronous
        # execute_model.
        if self.max_concurrent_batches > 1:
            # Note: must use only 1 IO thread to keep dequeue sequence
            # from the response queue
            # _async_aggregate_workers_output also assumes a single IO thread
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")

        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None


class AscendWorkerProc(WorkerProc):

    @staticmethod
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,  # Receive SchedulerOutput
        shared_worker_lock: LockType,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        # (reader, writer)
        reader, writer = context.Pipe(duplex=False)

        # Create death pipe to detect parent process exit
        death_reader, death_writer = context.Pipe(duplex=False)

        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": (reader, writer),
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
        }
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=False,
        )

        proc.start()
        writer.close()
        # Keep death_writer open in parent - when parent exits,
        # death_reader in child will get EOFError
        return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)


vllm.v1.executor.multiproc_executor.MultiprocExecutor = AscendMultiprocExecutor
