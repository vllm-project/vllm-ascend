import vllm.distributed.ec_transfer.ec_connector.shared_storage_connector
from safetensors.torch import load_file
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.ec_transfer.ec_connector.shared_storage_connector import (
    ECSharedStorageConnector, ECSharedStorageConnectorMetadata)
from vllm.logger import logger


class AscendECSharedStorageConnector(ECSharedStorageConnector):

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECSharedStorageConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning((
                "In connector.start_load_caches, ",
                "but the connector metadata is None",
            ))
            return
        # Load the EC for each mm data
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            filename = self._generate_filename_debug(mm_data.mm_hash)
            ec_cache = load_file(filename)["ec_cache"].npu()
            encoder_cache[mm_data.mm_hash] = ec_cache
            logger.debug("Success load encoder cache for hash %s",
                         mm_data.mm_hash)


vllm.distributed.ec_transfer.ec_connector.shared_storage_connector.ECSharedStorageConnector = AscendECSharedStorageConnector
