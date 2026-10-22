from unittest.mock import MagicMock

import pytest
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import EmbeddingStoreOperationError

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_memory import (
    AscendStoreMemory,
    AscendStoreTopology,
)


@pytest.mark.parametrize("topology", list(AscendStoreTopology))
def test_bind_once_and_validate_only_fixed_pool(topology):
    store, transfer = MagicMock(), MagicMock()
    store.register_buffer.return_value = 0
    memory = AscendStoreMemory(store, transfer, topology)
    slab = MagicMock(nbytes=4096)
    slab.data_ptr.return_value = 8192
    memory.bind_staging(slab)
    memory.bind_staging(slab)
    memory.validate_buffers([8192, 8256], [24, 128])
    if topology is AscendStoreTopology.INDEPENDENT:
        store.register_buffer.assert_called_once_with(8192, 4096)
    else:
        store.register_buffer.assert_not_called()
    with pytest.raises(EmbeddingStoreOperationError):
        memory.validate_buffers([8191], [128])
    with pytest.raises(EmbeddingStoreOperationError):
        memory.validate_buffers([12280], [24])
    with pytest.raises(EmbeddingStoreOperationError):
        memory.validate_buffers([8192], [])
    store.unregister_buffer.assert_not_called()
    transfer.acquire_registration_ranges.assert_not_called()
