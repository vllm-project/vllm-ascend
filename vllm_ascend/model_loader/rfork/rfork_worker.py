import logging
import threading

from vllm_ascend.model_loader.rfork.seed_server import start_rfork_server
from vllm_ascend.model_loader.rfork.transfer_engine import (
    RForkTransferEngineBackendWorker,
)
from vllm_ascend.model_loader.rfork.utils import (
    get_local_seed_key,
    get_seed,
    release_seed,
    report_seed,
)

logger = logging.getLogger(__name__)


class RForkWorker:
    def __init__(
        self,
        disaggregation_mode: str,
        node_rank: int,
        tp_rank: int,
        gpu_id: int,
        dtype: str,
        is_draft_model: bool = False,
    ):
        self.disaggregation_mode = disaggregation_mode
        self.node_rank = node_rank
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.dtype = dtype
        self.is_draft_model = is_draft_model
        self.rfork_seed = None
        self.transfer_engine_backend_worker = RForkTransferEngineBackendWorker()
        self.transfer_result = False
        self.ready_to_start_seed_service = False
        self.seed_service_started = False

        self.local_seed_key = get_local_seed_key(
            self.disaggregation_mode,
            self.node_rank,
            self.tp_rank,
            self.is_draft_model,
        )

    def is_seed_available(self) -> bool:
        self.rfork_seed = get_seed(
            self.disaggregation_mode,
            self.node_rank,
            self.tp_rank,
            self.is_draft_model,
        )
        return self.rfork_seed is not None

    def is_transfer_succeeded(self) -> bool:
        return self.transfer_result

    def set_transfer_result(self, result: bool):
        self.transfer_result = result

    def pre_transfer(self, model) -> bool:
        try:
            assert self.transfer_engine_backend_worker.is_initialized(), (
                "transfer_engine_backend_worker is not initialized, cannot pre_transfer."
            )
            result = self.transfer_engine_backend_worker.register_memory_region_v2(model)
            self.ready_to_start_seed_service = result
            return result
        except AssertionError as e:
            logger.exception("Pre-transfer failed: %s", e)
            return False

    def transfer(self, model) -> bool:
        try:
            assert self.transfer_engine_backend_worker.is_initialized(), (
                "transfer_engine_backend_worker is not initialized, cannot transfer."
            )
            assert self.rfork_seed is not None, "rfork seed is None, cannot transfer."
            return self.transfer_engine_backend_worker.recv_from_source(
                model=model,
                seed_instance_ip=self.rfork_seed["seed_ip"],
                seed_instance_service_port=self.rfork_seed["seed_port"],
                local_seed_key=self.local_seed_key,
            )
        except AssertionError as e:
            logger.exception("Transfer failed: %s", e)
            return False

    def post_transfer(self):
        if self.rfork_seed is None:
            logger.warning("rfork seed is None, no need to release.")
            return True
        release_seed(self.rfork_seed)
        return True

    def start_seed_service(self, model):
        if self.seed_service_started:
            logger.info("Seed service already started, skipping.")
            return

        if not self.ready_to_start_seed_service:
            if not self.pre_transfer(model):
                return

        port = start_rfork_server(
            self.local_seed_key,
            (
                self.transfer_engine_backend_worker.rfork_transfer_engine_session_id,
                self.transfer_engine_backend_worker.rfork_transfer_engine_weights_info_dict,
            ),
        )
        if port > 0:
            self.rfork_heartbeat_thread = threading.Thread(
                target=report_seed,
                args=(
                    port,
                    self.disaggregation_mode,
                    self.node_rank,
                    self.tp_rank,
                    self.is_draft_model,
                ),
                daemon=True,
                name="RForkHeartbeat",
            )
            self.rfork_heartbeat_thread.start()
        self.seed_service_started = True
