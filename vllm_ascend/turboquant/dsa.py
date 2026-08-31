import torch

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

DSA_TQ_HEAD_DIM = 512
DSA_TQ_ROPE_HEAD_DIM = 64
DSA_TQ_SCALE_BYTES = 2
DSA_TQ_SLOT_SIZE = DSA_TQ_HEAD_DIM // 2 + DSA_TQ_SCALE_BYTES
DSA_TQ_KV_QUANT_MODE = 3


def _tq_store():
    from vllm_ascend.turboquant import tq_latent_store

    return tq_latent_store


def validate_dsa_tq(enabled: bool, head_dim: int, rope_head_dim: int | None) -> None:
    if not enabled:
        return
    device_type = get_ascend_device_type()
    if device_type not in {AscendDeviceType.A2, AscendDeviceType.A3}:
        raise RuntimeError(
            "DeepSeek V4 TurboQuant KV cache is only supported on Ascend A2/A3, "
            f"got {device_type}."
        )
    if head_dim != DSA_TQ_HEAD_DIM or rope_head_dim != DSA_TQ_ROPE_HEAD_DIM:
        raise ValueError(
            "DeepSeek V4 TurboQuant requires head_dim=512 and rope_head_dim=64, "
            f"got head_dim={head_dim}, rope_head_dim={rope_head_dim}."
        )


def write_dsa_kv_cache(
    enabled: bool,
    cache: torch.Tensor,
    kv: torch.Tensor | None,
    slot_mapping: torch.Tensor,
) -> None:
    if kv is None or kv.shape[0] == 0:
        return
    if cache.dtype != torch.uint8:
        if enabled:
            kv = _tq_store().had_fwd(
                kv.reshape(-1, DSA_TQ_HEAD_DIM),
                head_dim=DSA_TQ_HEAD_DIM,
            ).view_as(kv)
        DeviceOperator.dsa_kv_compress_scatter(cache, kv, slot_mapping)
        return
    if not enabled:
        raise RuntimeError("DeepSeek V4 uint8 KV cache requires TurboQuant on a sparse compressor layer.")
    if cache.shape[-1] != DSA_TQ_SLOT_SIZE:
        raise RuntimeError(
            f"DeepSeek V4 TurboQuant cache must use uint8 slots with width {DSA_TQ_SLOT_SIZE}, "
            f"got dtype={cache.dtype}, shape={tuple(cache.shape)}."
        )
    tq_store = _tq_store()
    packed_kv, workspace = tq_store.compress_kernel(
        kv.reshape(-1, DSA_TQ_HEAD_DIM),
        head_dim=DSA_TQ_HEAD_DIM,
        output_mode=tq_store.COMPRESS_OUTPUT_COMPACT_CORRECTED,
    )
    DeviceOperator.dsa_kv_compress_scatter(
        cache,
        packed_kv.view(-1, 1, DSA_TQ_SLOT_SIZE),
        slot_mapping,
    )
    # Keep the Hadamard-space source alive until scatter has been queued.
    del workspace


def transform_dsa_query(enabled: bool, query: torch.Tensor) -> torch.Tensor:
    if not enabled:
        return query
    return _tq_store().had_fwd(query, head_dim=DSA_TQ_HEAD_DIM)


def restore_dsa_output(enabled: bool, output: torch.Tensor) -> torch.Tensor:
    if not enabled:
        return output
    return _tq_store().had_inv(output, head_dim=DSA_TQ_HEAD_DIM)
