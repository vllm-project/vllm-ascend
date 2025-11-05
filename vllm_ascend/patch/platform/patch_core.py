import signal
from typing import Optional

from vllm.config import ParallelConfig
from vllm.logger import logger
from vllm.transformers_utils.config import \
    maybe_register_config_serialize_by_value
from vllm.utils import decorate_logs, set_process_title
from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc


def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    """Launch EngineCore busy loop in background process."""

    from vllm.distributed.device_communicators.shm_broadcast import \
        MessageQueue  # noqa

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

    engine_core: Optional[EngineCoreProc] = None
    try:
        parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
        if parallel_config.data_parallel_size > 1 or dp_rank > 0:
            set_process_title("EngineCore", f"DP{dp_rank}")
            decorate_logs()
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            parallel_config.data_parallel_rank_local = local_dp_rank
            engine_core = DPEngineCoreProc(*args, **kwargs)
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
