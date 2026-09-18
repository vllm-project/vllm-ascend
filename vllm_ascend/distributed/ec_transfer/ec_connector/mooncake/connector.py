# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Role shell for the Ascend Mooncake encoder-cache connector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake.scheduler import (
    ECMooncakeScheduler,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
)

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import (
    AscendECMooncakeWorker,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class AscendECMooncakeConnector(ECMooncakeConnector):
    """Reuse upstream scheduling and replace only worker device primitives."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole) -> None:
        # Initialize the public connector shell directly.  Upstream currently
        # constructs its concrete worker in ``ECMooncakeConnector.__init__``;
        # relying on downstream factory hooks would couple this plugin to a
        # locally patched vLLM checkout.
        ECConnectorBase.__init__(self, vllm_config=vllm_config, role=role)
        self._scheduler = None
        self._worker = None
        self._closed = False

        if role == ECConnectorRole.SCHEDULER:
            self._scheduler = ECMooncakeScheduler(vllm_config)
        elif role == ECConnectorRole.WORKER:
            self._worker = AscendECMooncakeWorker(vllm_config)
        else:
            raise ValueError(f"Unknown EC connector role: {role}")
