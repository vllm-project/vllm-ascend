import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from ..base import AscendAttentionScheme


def _quant_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        # ModelSlim exports the per-channel V cache scale as a column vector
        # ([hidden, 1]); flatten it first. The loader delivers the FULL-width
        # scale on every rank (attention-module parameters bypass the
        # ColumnParallelLinear sharding), so slice it here following the same
        # rule vLLM uses to replicate KV heads under GQA TP: with
        # num_kv_heads < tp_size each head is replicated across
        # tp_size // num_kv_heads ranks and rank r owns the 256-dim slice of
        # head r // (tp_size // num_kv_heads). When num_kv_heads >= tp_size
        # this degenerates to the plain contiguous narrow.
        if loaded_weight.dim() != 1:
            loaded_weight = loaded_weight.flatten()
        if loaded_weight.numel() != param.numel():
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            head_dim = param.numel()  # this rank's full (possibly replicated) head width
            total = loaded_weight.numel()
            num_heads_total = total // head_dim if total % head_dim == 0 else None
            if num_heads_total is None or num_heads_total < 1:
                raise AssertionError(
                    "[vllm-ascend/MXFP8_PER_CHANNEL] Cannot map V cache scale of "
                    f"{total} elements onto a per-rank head width of {head_dim} "
                    f"(TP size {tp_size}, TP rank {tp_rank})."
                )
            heads_per_rank_group = max(1, tp_size // num_heads_total)
            src_head = tp_rank // heads_per_rank_group
            loaded_weight = loaded_weight.narrow(0, src_head * head_dim, head_dim)
        assert param.numel() == loaded_weight.numel(), (
            "[vllm-ascend/MXFP8_PER_CHANNEL] V cache scale size mismatch: parameter "
            f"has {param.numel()} elements but the sliced weight has "
            f"{loaded_weight.numel()} (TP size "
            f"{get_tensor_model_parallel_world_size()}, TP rank "
            f"{get_tensor_model_parallel_rank()})."
        )
        param.data.copy_(loaded_weight.view_as(param))


class AscendC8MXFPKVCacheAttentionMethod(AscendAttentionScheme):
    """MXFP8 KV cache storage for dense-attention models.

    K/V are cached as FP8 E4M3 and their E8M0 scales are stored in extra cache
    tensors: K uses dynamic per-token-group scales written at scatter time, V
    uses the static per-channel E8M0 scale stored in the ModelSlim checkpoint
    (kv_cache_type == "K_DYNAMIC_V_STATIC_MXFP8_PER_CHANNEL"). The C8-MXFP
    backend owns the matching 512-token kernel block size, keeping hybrid
    cache scheduling and cache views aligned.
    """

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        layer.kv_cache_torch_dtype = torch.float8_e4m3fn
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_v1 import (
                AscendC8MXFPAttentionBackend,
                AscendC8MXFPAttentionBackendImpl,
            )

            layer.attn_backend = AscendC8MXFPAttentionBackend
            layer.impl.__class__ = AscendC8MXFPAttentionBackendImpl
            # Changing __class__ does not invoke the new class's __init__, so
            # initialize the state the impl relies on here. The V-scale fill
            # tracker is keyed by cache tensor identity, which stays correct
            # across memory-profiling runs (fresh dummy caches) and the real
            # KV cache without any explicit reset hook.
            layer.impl.enable_hamming_sparse = False
            layer.impl._v_scale_filled_caches = set()

        # Load v_cache static quantization scale
        hidden_size = layer.num_kv_heads * layer.head_size_v
        # E8M0 stores the exponent with a bias of 127, so 127 represents a
        # neutral scale of 1.0. Use it as a deterministic fallback instead of
        # leaving the parameter with uninitialized memory when a checkpoint is
        # missing a layer's V-cache scale.
        weight_param = torch.nn.Parameter(
            torch.full((hidden_size,), 127, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("v_cache_scale", weight_param)
        # When loading weights, segment them according to TP
        weight_param.weight_loader = _quant_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        vllm_config = get_current_vllm_config()
        target_dtype = vllm_config.model_config.dtype
        raw = layer.v_cache_scale.data
        # A minmax calibrator emits 0 for a channel whose absmax was 0, and
        # 2^-127 there would make the quantization reciprocal 2^127 -- any
        # activation that is not exactly zero at inference would go to inf.
        # Sanitize the stored bytes in place so BOTH consumers stay neutral:
        # the reciprocal below and the raw bytes broadcast into the V-scale
        # cache (2^-127 there would zero the channel on dequant)  -- a
        # real-checkpoint pitfall hit during the vendored-QFA bring-up.
        if bool((raw == 0).any()):
            raw[raw == 0] = 127
        exponent = raw.to(torch.float32) - 127
        # Only the reciprocal is consumed (npu_quantize needs 1/scale); the
        # forward scale itself is written into the V-scale cache as raw E8M0
        # bytes straight from v_cache_scale.
        layer.v_cache_scale_float_reciprocal = torch.nn.Parameter(
            (1 / torch.exp2(exponent)).to(target_dtype),
            requires_grad=False,
        )

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
