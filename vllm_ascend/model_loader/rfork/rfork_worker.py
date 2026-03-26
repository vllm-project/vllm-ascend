#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import threading

from vllm.logger import logger

from vllm_ascend.model_loader.rfork.seed_protocol import RForkSeedProtocol
from vllm_ascend.model_loader.rfork.seed_server import start_rfork_server
from vllm_ascend.model_loader.rfork.transfer_backend import (
    RForkTransferBackend,
)


class RForkWorker:
    def __init__(
        self,
        disaggregation_mode: str,
        node_rank: int,
        tp_rank: int,
        device_id: int,
        scheduler_url: str,
        model_url: str,
        model_deploy_strategy_name: str,
        seed_timeout_sec: float = 30.0,
        seed_key_separator: str = "$",
        is_draft_model: bool = False,
    ):
        self.device_id = device_id
        self.rfork_seed = None
        self.transfer_backend = RForkTransferBackend()
        self.ready_to_start_seed_service = False
        self.seed_service_started = False
        self.seed_timeout_sec = seed_timeout_sec
        self.seed_protocol = RForkSeedProtocol(
            disaggregation_mode=disaggregation_mode,
            node_rank=node_rank,
            tp_rank=tp_rank,
            scheduler_url=scheduler_url,
            model_url=model_url,
            model_deploy_strategy_name=model_deploy_strategy_name,
            seed_key_separator=seed_key_separator,
            is_draft_worker=is_draft_model,
        )

    def is_seed_available(self) -> bool:
        self.rfork_seed = self.seed_protocol.get_seed()
        return self.rfork_seed is not None

    def pre_transfer(self, model) -> bool:
        try:
            assert self.transfer_backend.is_initialized(), "transfer_backend is not initialized, cannot pre_transfer."
            result = self.transfer_backend.register_memory_region(model)
            self.ready_to_start_seed_service = result
            return result
        except AssertionError as e:
            logger.exception("Pre-transfer failed: %s", e)
            return False

    def transfer(self, model) -> bool:
        try:
            assert self.transfer_backend.is_initialized(), "transfer_backend is not initialized, cannot transfer."
            assert self.rfork_seed is not None, "rfork seed is None, cannot transfer."
            return self.transfer_backend.recv_from_source(
                model=model,
                seed_instance_ip=self.rfork_seed["seed_ip"],
                seed_instance_service_port=self.rfork_seed["seed_port"],
                local_seed_key=self.seed_protocol.get_local_seed_key(),
            )
        except AssertionError as e:
            logger.exception("Transfer failed: %s", e)
            return False

    def post_transfer(self):
        if self.rfork_seed is None:
            logger.info("rfork seed is None, no need to release.")
            return True
        self.seed_protocol.release_seed(self.rfork_seed)
        return True

    def start_seed_service(self, model):
        if self.seed_service_started:
            logger.info("Seed service already started, skipping.")
            return

        if not self.ready_to_start_seed_service:
            if not self.pre_transfer(model):
                return

        port = start_rfork_server(
            self.seed_protocol.get_local_seed_key(),
            (
                self.transfer_backend.rfork_transfer_engine_session_id,
                self.transfer_backend.rfork_transfer_engine_weights_info_dict,
            ),
            health_timeout_sec=self.seed_timeout_sec,
        )
        if port > 0:
            self.rfork_heartbeat_thread = threading.Thread(
                target=self.seed_protocol.report_seed,
                args=(port,),
                daemon=True,
                name="RForkHeartbeat",
            )
            self.rfork_heartbeat_thread.start()
        self.seed_service_started = True
