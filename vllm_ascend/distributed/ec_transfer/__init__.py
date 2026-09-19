# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def register_connector() -> None:
    """Replace vLLM's CUDA Mooncake data plane with the Ascend adapter."""
    from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory

    ECConnectorFactory._registry.pop("ECMooncakeConnector", None)
    ECConnectorFactory.register_connector(
        "ECMooncakeConnector",
        "vllm_ascend.distributed.ec_transfer.mooncake",
        "AscendECMooncakeConnector",
    )
