import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache


def init_asecnd_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: EncoderCache | None,
    device: torch.device,
):
    from vllm_ascend.worker.v2.model_states.default import AscendModelState

    return AscendModelState(vllm_config, model, encoder_cache, device)
