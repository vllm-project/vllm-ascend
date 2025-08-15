# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Tuple
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlConnector
)


class NixlNpuConnector(NixlConnector):
    
    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(
            self,
            kv_caches: dict[
                str,  # type: ignore[override]
                Tuple[torch.Tensor]]):
        converted_kv_caches: dict[str, torch.Tensor] = {
            key: tensor_tuple[0] for key, tensor_tuple in kv_caches.items()
        }
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(converted_kv_caches)
    

