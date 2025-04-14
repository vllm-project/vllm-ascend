from vllm.distributed.kv_transfer.kv_connector.factory import \
    KVConnectorFactory

KVConnectorFactory.register_connector(
    "AscendHcclConnector", "vllm_ascend.distributed.llmdatadist_connector",
    "LLMDataDistConnector")