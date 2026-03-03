import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch
import vllm
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

@staticmethod
@contextmanager
def _get_kv_connector_output(
    scheduler_output: "SchedulerOutput", wait_for_save: bool = True
) -> Generator[KVConnectorOutput, None, None]:
    output = KVConnectorOutput()

    # Update KVConnector with the KVConnector metadata forward().
    kv_connector = get_kv_transfer_group()
    assert isinstance(kv_connector, KVConnectorBase)
    assert scheduler_output.kv_connector_metadata is not None
    kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

    # Background KV cache transfers happen here.
    # These transfers are designed to be async and the requests
    # involved may be disjoint from the running requests.
    # Do this here to save a collective_rpc.
    kv_connector.start_load_kv(get_forward_context())
    try:
        yield output
    finally:
        if wait_for_save:
            kv_connector.wait_for_save()

        output.finished_sending, output.finished_recving = (
            kv_connector.get_finished(scheduler_output.finished_req_ids)
        )
        output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

        output.kv_connector_stats = kv_connector.get_kv_connector_stats()
        output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()

vllm.v1.worker.kv_connector_model_runner_mixin.KVConnectorModelRunnerMixin._get_kv_connector_output = _get_kv_connector_output
