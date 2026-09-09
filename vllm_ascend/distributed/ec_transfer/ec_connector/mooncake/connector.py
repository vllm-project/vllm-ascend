# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Role shell for the Ascend Mooncake encoder-cache connector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.ec_transfer.ec_connector.mooncake_ec_connector import (
    ECMooncakeConnector,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.ec_transfer.ec_connector.mooncake.worker import (
        ECMooncakeWorker,
    )


class AscendECMooncakeConnector(ECMooncakeConnector):
    """Reuse upstream scheduling and replace only worker device primitives."""

    def _make_worker(self, vllm_config: VllmConfig) -> ECMooncakeWorker:
        from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.worker import (
            AscendECMooncakeWorker,
        )

        return AscendECMooncakeWorker(vllm_config)
