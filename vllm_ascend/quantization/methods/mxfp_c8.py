import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger

from .base import AscendAttentionScheme


def _quant_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        shard_size = loaded_weight.shape[0] // tp_size
        loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)
        assert param.size() == loaded_weight.size(), (
            "[vllm-ascend/MXFP8_PER_CHANNEL] Attempted to load weight "
            f"({loaded_weight.size()}) into parameter ({param.size()}) "
            f"when TP size is {tp_size} and TP rank is {tp_rank}."
        )

        param.data.copy_(loaded_weight)


class AscendC8MXFPKVCacheAttentionMethod(AscendAttentionScheme):
    """MXFP8 KV cache storage for dense-attention models.

    This method only changes the cache storage path: K/V are cached as FP8 E4M3
    and their per-32-element E8M0 scales are stored in extra cache tensors. The
    attention operator call path intentionally stays unchanged for now.
    """

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        layer.kv_cache_torch_dtype = torch.float8_e4m3fn
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_v1 import AscendC8MXFPAttentionBackendImpl
            layer.impl.__class__ = AscendC8MXFPAttentionBackendImpl
            layer.impl.save_v_scale_flag = False

        # Load v_cache static quantization scale
        hidden_size = layer.num_kv_heads * layer.head_size_v
        weight_param = torch.nn.Parameter(torch.empty(hidden_size, dtype=torch.uint8), requires_grad=False)
        layer.register_parameter("v_cache_scale", weight_param)
        # When loading weights, segment them according to TP
        weight_param.weight_loader = _quant_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        vllm_config = get_current_vllm_config()
        target_dtype = vllm_config.model_config.dtype
        exponent = layer.v_cache_scale.data.to(torch.float32) - 127
        layer.v_cache_scale_float = torch.nn.Parameter(torch.exp2(exponent).to(target_dtype), requires_grad=False)
        layer.v_cache_scale_float_reciprocal = torch.nn.Parameter(1 / torch.exp2(exponent).to(target_dtype), requires_grad=False)

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
            "AscendC8MXFPKVCacheAttentionMethod.apply should not be called. "
            "C8_MXFP KV cache quantization is handled by the attention backend."
        )
