# mypy: ignore-errors
import signal

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import \
    maybe_register_config_serialize_by_value
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc

logger = init_logger(__name__)


class BalanceDPEngineCoreProc(DPEngineCoreProc):

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs)
            self.scheduler.balance_gather(self.dp_group)

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug("Wave %d finished, pausing engine loop.",
                                 self.current_wave)
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait((
                        client_index,
                        EngineCoreOutputs(wave_complete=self.current_wave),
                    ))
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0


def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    """Launch EngineCore busy loop in background process."""

    # Signal handler used for graceful termination.
    # SystemExit exception is only raised once to allow this and worker
    # processes to terminate without error
    shutdown_requested = False

    # Ensure we can serialize transformer config after spawning
    maybe_register_config_serialize_by_value()

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    # Either SIGTERM or SIGINT will terminate the engine_core
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    engine_core: EngineCoreProc | None = None
    try:
        parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
        if parallel_config.data_parallel_size > 1 or dp_rank > 0:
            set_process_title("EngineCore", f"DP{dp_rank}")
            decorate_logs()
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            parallel_config.data_parallel_rank_local = local_dp_rank
            engine_core = BalanceDPEngineCoreProc(*args, **kwargs)
        else:
            set_process_title("EngineCore")
            decorate_logs()
            engine_core = EngineCoreProc(*args, **kwargs)

        engine_core.run_busy_loop()

    except SystemExit:
        logger.debug("EngineCore exiting.")
        raise
    except Exception as e:
        if engine_core is None:
            logger.exception("EngineCore failed to start.")
        else:
            logger.exception("EngineCore encountered a fatal error.")
            engine_core._send_engine_dead()
        raise e
    finally:
        if engine_core is not None:
            engine_core.shutdown()


EngineCoreProc.run_engine_core = run_engine_core
