# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from ..base import AscendAttentionScheme


def _hif8_quant_scale_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Weight loader for the HIF8 C8 per-tensor scales.

    The HIF8 C8 recipe quantizes the whole Q (or latent KV) tensor with a
    single scalar scale. ModelSlim may export it as a bare scalar or as a
    per-head column vector whose entries all repeat the same per-tensor
    value. Accept any shape; when more than one element is present, they
    must all agree, otherwise the checkpoint is not a per-tensor recipe.

    Attention-module parameters bypass ColumnParallelLinear sharding, so
    every TP rank receives the full tensor; taking the (uniform) first
    element keeps the per-tensor scale replicated across ranks.
    """
    values = loaded_weight.detach().flatten()
    if values.numel() == 0:
        raise ValueError("[vllm-ascend/HIF8] Received an empty HIF8 quantization scale tensor.")
    first = values[0].to(torch.float32)
    if values.numel() > 1:
        values_fp32 = values.to(torch.float32)
        if not torch.equal(values_fp32, first.expand_as(values_fp32)):
            raise ValueError(
                "[vllm-ascend/HIF8] The QK_HIF8_PER_TENSOR recipe expects one uniform "
                f"per-tensor scale, but the checkpoint carries {values.numel()} "
                "non-uniform values (per-head scales are not supported by the HIF8 "
                "C8 path)."
            )
    param.data.copy_(first)


class AscendHIF8KVCacheAttentionMethod(AscendAttentionScheme):
    """HIF8 C8 KV cache storage for sparse-attention MLA models (GLM-5.3-Flash).

    Both the latent KV cache and the absorbed query are quantized to HIF8
    with per-tensor scales exported by ModelSlim (``fa_q``/``fa_k`` scale
    under the ``QK_HIF8_PER_TENSOR`` recipe). The main KV cache
    becomes a single uint8 byte tensor (HIF8 latents, ``kv_lora_rank`` bytes
    per token) and attention runs on aclnnQuantSparseFlashMla.

    Enabled with ``--kv-cache-dtype hif8``; the checkpoint must carry the
    matching recipe (validated at model construction time).
    """

    def __init__(self, prefix: str):
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        for name in ("fa_q", "fa_k"):
            module = torch.nn.Module()
            setattr(layer, name, module)
            scale = torch.nn.Parameter(torch.full((1,), float("nan"), dtype=torch.float32), requires_grad=False)
            scale.weight_loader = _hif8_quant_scale_weight_loader
            module.register_parameter("scale", scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        q_descale = layer.fa_q.scale.detach()
        kv_descale = layer.fa_k.scale.detach()
        descales = torch.cat((q_descale, kv_descale))
        reciprocals = descales.reciprocal()
        if not torch.all((descales > 0) & torch.isfinite(descales) & torch.isfinite(reciprocals)):
            raise ValueError(
                f"{self.prefix}: HIF8 requires loaded, finite positive fa_q.scale and fa_k.scale "
                "with finite reciprocals."
            )
        layer.register_buffer("hif8_q_descale", q_descale, persistent=False)
        layer.register_buffer("hif8_kv_descale", kv_descale, persistent=False)
        layer.register_buffer("hif8_q_scale_reciprocal", reciprocals[:1], persistent=False)
        layer.register_buffer("hif8_kv_scale_reciprocal", reciprocals[1:], persistent=False)
        layer.hif8_quant_layer = True

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "AscendHIF8KVCacheAttentionMethod.apply should not be called. "
            "HIF8 C8 KV cache quantization is handled by the attention backend."
        )
