import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger

from .base import AscendAttentionScheme
from .registry import register_scheme


def _int2_kv_scale_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Weight loader for dense-attention INT2 KV cache scales.

    INT2 uses symmetric quantization (scale only, no offset).
    Handles TP sharding when the scale is per-head.
    """
    loaded_weight = loaded_weight.squeeze()
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    elif param.data.shape != loaded_weight.shape:
        # TP-sharded: loaded_weight is [total_kv_heads, head_size], shard to this rank
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        total_kv_heads = loaded_weight.shape[0]
        local_kv_heads = total_kv_heads // tp_size
        start = tp_rank * local_kv_heads
        shard = loaded_weight[start : start + local_kv_heads].contiguous()
        param.data = shard.to(param.dtype).clone()
    else:
        param.data.copy_(loaded_weight)


@register_scheme("INT2_PER_TOKEN_HEAD", "attention")
class AscendINT2KVCacheAttentionMethod(AscendAttentionScheme):
    """INT2 KV cache quantization for dense-attention models.

    This implements static per-channel symmetric INT2 quantization where
    KV cache tensors are stored as packed uint8 (4 INT2 values per byte).

    The quantization formula is:
        q = clamp(round(x / scale), -2, 1)   # signed INT2 range
        x_dequant = q * scale

    Packing: 4 consecutive INT2 values along the head_dim axis are packed
    into a single uint8 byte. The cache tensor shape is therefore:
        (2, num_blocks, block_size, num_kv_heads, head_size // 4)
    with dtype torch.uint8.

    This class performs class surgery on the attention layer's impl to
    attach AscendINT2AttentionBackendImpl which handles the INT2-specific
    forward path.
    """

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix
        vllm_config = get_current_vllm_config()
        self.is_kv_producer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer

    def create_weights(self, layer: torch.nn.Module) -> None:
        """Create INT2 KV cache scale parameters and perform class surgery.

        Sets layer.kv_cache_torch_dtype to torch.uint8 (packed INT2) and
        rebinds layer.impl.__class__ to AscendINT2AttentionBackendImpl.
        """
        if not self.is_kv_producer:
            logger.info_once(
                "[vllm-ascend/INT2_KV] KV cache producer is disabled; "
                "setting kv_cache_torch_dtype to torch.uint8 (packed INT2)."
            )
            layer.kv_cache_torch_dtype = torch.uint8

        # Class surgery: rebind impl to INT2-specific subclass
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_int2 import AscendINT2AttentionBackendImpl

            layer.impl.__class__ = AscendINT2AttentionBackendImpl

        # Create per-channel scale parameters (symmetric — no offset for INT2)
        dtype = torch.get_default_dtype()
        layer.k_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        layer.k_cache_scale.weight_loader = _int2_kv_scale_weight_loader
        layer.v_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        layer.v_cache_scale.weight_loader = _int2_kv_scale_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Flatten scale parameters after weight loading."""
        layer.k_cache_scale.data = layer.k_cache_scale.data.flatten()
        layer.v_cache_scale.data = layer.v_cache_scale.data.flatten()

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
        """This method should not be called -- INT2 is handled by the attention backend.

        The class surgery in create_weights() ensures the forward path goes
        through AscendINT2AttentionBackendImpl.forward() instead.
        """
        err_msg = (
            "[vllm-ascend/INT2_KV] AscendINT2KVCacheAttentionMethod.apply should "
            "not be called. INT2 KV cache quantization is handled by the "
            "attention backend."
        )
        raise RuntimeError(err_msg)
