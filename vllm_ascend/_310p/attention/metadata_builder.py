from __future__ import annotations

import torch
from typing import Any
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder as _BaseBuilder
from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder


class AscendAttentionMetadataBuilder310P(_BaseBuilder):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        
        self.attn_mask_builder: Any = AttentionMaskBuilder(self.device)
