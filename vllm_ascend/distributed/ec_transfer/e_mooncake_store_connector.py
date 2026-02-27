from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnectorMetadata,
    MMMeta,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.distributed.ec_transfer.e_mooncake_backend import EMooncakeBackend

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class EMoonCakeStoreConnector(ECConnectorBase):
    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        self.ec_transfer_config = vllm_config.ec_transfer_config
        self._mm_datas_need_loads: dict[str, int] = {}
        if role == ECConnectorRole.SCHEDULER:
            torch.npu.set_device(0)

        # init ec mooncacke store like pool worker
        # if vllm_config.ec_transfer_config.is_ec_producer:
        parallel_config = vllm_config.parallel_config
        self.ec_store = EMooncakeBackend(parallel_config, init_tcp=(role == ECConnectorRole.SCHEDULER))

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        # self.send_queue.put((mm_hash, encoder_cache[mm_hash].cpu()))
        # feat_key, tensor = self.send_queue.get()
        # self.ec_store.put_single(mm_hash, encoder_cache[mm_hash].cpu())
        self.ec_store.put_single(mm_hash, encoder_cache[mm_hash])
        logger.info(f"Successfully put key:{mm_hash} to Mooncacke")

    def start_load_caches(self, encoder_cache, **kwargs):
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECExampleConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                (
                    "In connector.start_load_caches, ",
                    "but the connector metadata is None",
                )
            )
            return
        # Load the EC for each mm data
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            try:
                ec_cache = self.ec_store.get_single(mm_data.mm_hash)
                if ec_cache is not None:
                    logger.info(f"Successfully get key:{mm_data.mm_hash} from Mooncacke")
                encoder_cache[mm_data.mm_hash] = ec_cache.npu()
            except Exception:
                logger.error("Failed to load mm_data.mm_hash: %s from ec_store", mm_data.mm_hash)

            logger.debug("Success load encoder cache for hash %s", mm_data.mm_hash)

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        num_encoder_token = request.get_num_encoder_embeds(index)
        # Insert mm_hash only if this block has not been recorded yet.
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only build for load mm_data only
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ECExampleConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    def has_caches(
        self,
        request: "Request",
    ):
        result = []
        for feature in request.mm_features:
            logger.info(f"has cache feature.identifier is:{feature.identifier}")
            res = self.ec_store.exist_single(feature.identifier)
            logger.info(f"res:{res}")
            result.append(res)
        return result
