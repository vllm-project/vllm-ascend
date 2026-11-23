# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""CPU-reference-exact Triton pixel normalization for Kimi on A3/A5."""

import torch
from torch import nn
from vllm.model_executor.models.utils import StageMissingLayer
from vllm.transformers_utils.processor import get_processor_config
from vllm.transformers_utils.processors.kimi_k25_vision_fused import KimiK25FusedVisionProcessor
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from vllm_ascend.ops.triton.kimi_pixel_norm import _KIMI_PIXEL_NORM_KERNEL


class KimiPixelNormLut(nn.Module):
    lut: torch.Tensor

    def __init__(self, reference: torch.Tensor, *, dtype: torch.dtype, device: torch.device):
        super().__init__()
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("visual dtype must be BF16 or FP16")
        # The Kimi processor provides the fixed FP32 [256,3] CPU table.
        # Retain only the two supported 16-bit tables, outside checkpoint state.
        # Dtype moves restore their exact FP32-reference rounding, never BF16->FP16.
        self._table_bits = {
            target: tuple(reference[:, 0].to(target).view(torch.int16).tolist())
            for target in (torch.bfloat16, torch.float16)
        }
        self.register_buffer("lut", torch.empty(0, dtype=dtype, device=device), persistent=False)
        self._vector_cores = 0
        self._refresh_table()

    @property
    def input_dtype(self) -> torch.dtype:
        return torch.uint8

    def _refresh_table(self) -> None:
        reference = torch.tensor(self._table_bits[self.lut.dtype], dtype=torch.int16, device="cpu").view(self.lut.dtype)
        self.lut = reference.to(device=self.lut.device)
        if self.lut.device.type == "npu":
            self._vector_cores = int(torch.npu.get_device_properties(self.lut.device).vector_core_num)
            if self._vector_cores <= 0:
                raise ValueError("vector core count must be positive")

    def _apply(self, fn, recurse=True):
        moved = fn(self.lut)
        if moved.dtype not in self._table_bits:
            raise ValueError("visual dtype must be BF16 or FP16")
        self.lut = moved
        self._refresh_table()
        return self

    def forward(self, pixels: torch.Tensor, visual_dtype: torch.dtype) -> torch.Tensor:
        if visual_dtype != self.lut.dtype:
            raise ValueError("lookup table dtype must match the patch projection dtype")
        return torch.ops.vllm.kimi_pixel_norm_lut(pixels, self.lut, self._vector_cores)

    def register_warmup(self) -> None:
        """Register the module for compile-only startup warmup."""
        _KIMI_PIXEL_NORM_KERNEL.register_warmup(self)


def build_kimi_pixel_norm(model_config, vision_tower: nn.Module) -> KimiPixelNormLut | None:
    if not model_config.multimodal_config.mm_device_do_normalize or isinstance(vision_tower, StageMissingLayer):
        return None
    hf_config = model_config.hf_config
    if hf_config.model_type not in ("kimi_k3", "kimi_k31") or not any(
        architecture in ("KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration")
        for architecture in hf_config.architectures
    ):
        raise ValueError("device pixel normalization is limited to Kimi-K3/K3.1")
    config = get_processor_config(model_config.model, revision=model_config.revision)
    config = config.get("image_processor", config)
    if "media_proc_cfg" not in config:
        config = model_config.hf_image_processor_config
    if "media_proc_cfg" not in config:
        # Legacy preprocessor files need not declare processor_class, so the
        # processor-class resolver can skip otherwise valid media metadata.
        config = (
            get_hf_file_to_dict("preprocessor_config.json", model_config.model, revision=model_config.revision) or {}
        )
    media_config = config["media_proc_cfg"]
    if media_config["patch_size"] != 14:
        raise ValueError("Kimi pixel LUT currently requires 14x14 RGB patches")
    if list(media_config["image_mean"]) != [0.5, 0.5, 0.5] or list(media_config["image_std"]) != [0.5, 0.5, 0.5]:
        raise ValueError("Kimi pixel LUT requires RGB mean/std equal to 0.5")
    processor = KimiK25FusedVisionProcessor(media_proc_cfg=media_config)
    projection = vision_tower.patch_embed.proj
    norm = KimiPixelNormLut(
        torch.from_numpy(processor.normalize_lut), dtype=projection.weight.dtype, device=projection.weight.device
    )
    norm.register_warmup()
    return norm
