# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.device import WeightFormat, get_current_hardware_profile

from ..base import AscendAttentionScheme
from ..registry import register_scheme


def validate_sparse_sfa_hif8(vllm_config, hif8_quant_layers) -> None:
    """Hardware and parallel-mode checks for ``--kv-cache-dtype hif8``.

    aclnnQuantSparseFlashMla (and the HIF8 quantize/dequantize chain) only
    ships for Ascend 950PR/950DT. PCP/DCP gather flows would also have to
    move the packed HIF8 payload; reject them until they are explicitly
    qualified.
    """
    if not get_current_hardware_profile().supports_weight_format(WeightFormat.HIF8):
        raise ValueError("--kv-cache-dtype hif8 (HIF8 C8 KV cache) requires Ascend 950PR/950DT products.")
    if not hif8_quant_layers:
        raise ValueError("--kv-cache-dtype hif8 requires ModelSlim QK_HIF8_PER_TENSOR layer recipes.")
    parallel_config = vllm_config.parallel_config
    if parallel_config.prefill_context_parallel_size * parallel_config.decode_context_parallel_size > 1:
        raise ValueError("--kv-cache-dtype hif8 does not support context parallel (PCP/DCP) yet.")


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


@register_scheme("QK_HIF8_PER_TENSOR", "attention")
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

    def __init__(self):
        pass

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
        layer.hif8_q_descale = torch.nn.Parameter(q_descale, requires_grad=False)
        layer.hif8_kv_descale = torch.nn.Parameter(kv_descale, requires_grad=False)
        layer.hif8_q_scale_reciprocal = torch.nn.Parameter(q_descale.reciprocal(), requires_grad=False)
        layer.hif8_kv_scale_reciprocal = torch.nn.Parameter(kv_descale.reciprocal(), requires_grad=False)
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
