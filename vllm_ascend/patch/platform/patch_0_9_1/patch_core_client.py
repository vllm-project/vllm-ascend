from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.engine.core_client import (AsyncMPClient, EngineCoreClient,
                                        MPClient)
from vllm.v1.executor.abstract import Executor

import vllm_ascend.envs as vllm_ascend_envs


def make_async_mp_client(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    client_addresses: Optional[dict[str, str]] = None,
    client_index: int = 0,
) -> "MPClient":
    # Use only AsyncMPClient here for dp scenario and use nginx for the dp request routering
    return AsyncMPClient(vllm_config, executor_class, log_stats,
                         client_addresses, client_index)


# Apply this patch only if the external data parallelism is enabled
if vllm_ascend_envs.VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED:
    # Patch the EngineCoreClient to use the custom make_async_mp_client
    EngineCoreClient.make_async_mp_client = make_async_mp_client  # type: ignore[attr-defined]
