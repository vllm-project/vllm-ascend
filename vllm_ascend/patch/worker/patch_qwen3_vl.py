import torch
import torch.nn as nn
from vllm.model_executor.models.qwen3_vl import Qwen3VLProcessingInfo
from transformers.models.qwen3_vl import Qwen3VLProcessor


class AscendQwen3VLProcessingInfo(nn.Module):
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            do_rescale = False,
            do_nomarlize = False,
            **kwargs,
        )

Qwen3VLProcessingInfo.get_hf_processor = AscendQwen3VLProcessingInfo.get_hf_processor
