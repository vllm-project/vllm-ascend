# SPDX-License-Identifier: Apache-2.0
"""Load optional recipes SiTU operators before Kimi model tracing."""

from importlib import import_module

import torch
from vllm.logger import logger

SITU_OP_NAMES = ("npu_situ_and_mul_sparse", "grouped_situ_mx_quant")


def initialize_kimi_situ_ops() -> None:
    """Register the installed recipes extension in each model worker.

    Merely checking torch.ops.custom does not load the extension. Do this at
    model construction, before Dynamo can specialize an unavailable-op branch.
    Keep the optional native fallback, but make missing operators visible.
    """
    if not all(hasattr(torch.ops.custom, name) for name in SITU_OP_NAMES):
        try:
            import_module("custom_ops")
        except (ImportError, OSError) as exc:
            logger.warning_once("Could not load the recipes custom_ops extension for Kimi SiTU: %s", str(exc))

    missing = [name for name in SITU_OP_NAMES if not hasattr(torch.ops.custom, name)]
    if missing:
        logger.warning_once(
            "Kimi SiTU operators unavailable: %s. Their paths use native fallbacks. "
            "Install the matching recipes ops/ascendc/torch_ops_extension package "
            "and its CANN operator package in every worker environment.",
            ", ".join(missing),
        )
    else:
        logger.info_once("Kimi recipes SiTU operators registered: %s", ", ".join(SITU_OP_NAMES))
