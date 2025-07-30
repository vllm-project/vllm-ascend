import os
import signal
from typing import Optional
from vllm.config import VllmConfig, ParallelConfig
from vllm.v1.engine.core import EngineCoreProc, DPEngineCoreProc
from vllm.utils import logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value)
import vllm_ascend.envs as vllm_ascend_envs

class ExternealDPEngineCoreProc(DPEngineCoreProc):
  def __init__(self, *args, **kwargs):
      # Use the external data parallelism master port from envs
      super().__init__(*args, **kwargs)
      self.engines_running = True
  
  def _has_global_unfinished_reqs(self, local_unfinished):
      return True

  def _init_data_parallel(self, vllm_config: VllmConfig):

      # Configure GPUs and stateless process group for data parallel.
      dp_rank = vllm_config.parallel_config.data_parallel_rank
      dp_size = vllm_config.parallel_config.data_parallel_size
      local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

      assert dp_size > 1
      assert 0 <= local_dp_rank <= dp_rank < dp_size

      if vllm_config.kv_transfer_config is not None:
          # modify the engine_id and append the local_dp_rank to it to ensure
          # that the kv_transfer_config is unique for each DP rank.
          vllm_config.kv_transfer_config.engine_id = (
              f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
          )
          logger.debug("Setting kv_transfer_config.engine_id to %s",
                        vllm_config.kv_transfer_config.engine_id)

      from vllm.platforms import current_platform
      device_control_env_var = current_platform.device_control_env_var
      world_size = vllm_config.parallel_config.world_size
      os.environ[device_control_env_var] = ",".join(
          str(current_platform.device_id_to_physical_device_id(i))
          for i in range(local_dp_rank * world_size, (local_dp_rank + 1) *
                          world_size))

      self.dp_rank = dp_rank

  def run_busy_loop(self):
      """Core busy loop of the EngineCore for data parallel case."""
      # Note: In customized DPEngineCoreProc, no idle time will exist. We assume the another dp groups are always 
      # running.

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

def run_engine_core(*args,
                    dp_rank: int = 0,
                    local_dp_rank: int = 0,
                    **kwargs):
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

    engine_core: Optional[EngineCoreProc] = None
    try:
        parallel_config: ParallelConfig = kwargs[
            "vllm_config"].parallel_config
        if parallel_config.data_parallel_size > 1 or dp_rank > 0:
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            parallel_config.data_parallel_rank_local = local_dp_rank
            engine_core = ExternealDPEngineCoreProc(*args, **kwargs)
        else:
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

# Apply this patch only if the external data parallelism is enabled
if vllm_ascend_envs.VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED:
  # Patch the EngineCoreClient to use the custom make_async_mp_client
  EngineCoreProc.run_engine_core = run_engine_core  # type: ignore[attr-defined]