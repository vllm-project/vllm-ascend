import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Union

import torch
from vllm.logger import init_logger
from vllm.v1.sample import logits_processor
from vllm.v1.sample.logits_processor.builtin import (LogitBiasLogitsProcessor,
                                                     MinTokensLogitsProcessor)
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.logits_processor.state import LogitsProcessors

from vllm_ascend.sample.logits_processor.builtin import \
    AscendMinPLogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Error message when the user tries to initialize vLLM with a pooling model
# and custom logitsproces
STR_POOLING_REJECTS_LOGITSPROCS = ("Pooling models do not support custom"
                                   " logits processors.")

BUILTIN_LOGITS_PROCESSORS: list[type[LogitsProcessor]] = [
    MinTokensLogitsProcessor,
    LogitBiasLogitsProcessor,
    AscendMinPLogitsProcessor,
]


def build_logitsprocs(
    vllm_config: "VllmConfig",
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[Union[str, type[LogitsProcessor]]] = (),
) -> LogitsProcessors:
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug("Skipping logits processor loading because pooling models"
                     " do not support logits processors.")
        return LogitsProcessors()
    custom_logitsprocs_classes = logits_processor._load_custom_logitsprocs(
        custom_logitsprocs)
    return LogitsProcessors(
        ctor(vllm_config, device, is_pin_memory) for ctor in itertools.chain(
            BUILTIN_LOGITS_PROCESSORS, custom_logitsprocs_classes))
