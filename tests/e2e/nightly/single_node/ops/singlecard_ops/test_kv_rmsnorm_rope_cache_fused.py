import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op, get_ascend_device_type

enable_custom_op()


def _require_a5():
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("This test targets A5 fused kv_rmsnorm_rope_cache with float8 KV cache.")


def _run_small_op_golden(
    kv_no_split: torch.Tensor,
    layernorm_weight: torch.Tensor,
    epsilon: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    fa_kscale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_nope, k_pe = torch.split(kv_no_split, [512, 64], dim=-1)
    k_nope, _ = torch_npu.npu_rms_norm(k_nope, layernorm_weight, epsilon)
    k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
    quant_k_nope = torch_npu.npu_quantize(
        k_nope,
        fa_kscale,
        None,
        kv_cache[0].dtype,
        -1,
        False,
    )
    torch_npu.npu_scatter_pa_kv_cache(
        key=quant_k_nope.squeeze(2),
        value=quant_k_nope.squeeze(2),
        slot_mapping=slots.to(torch.int64),
        key_cache=kv_cache[0],
        value_cache=kv_cache[0],
    )
    torch_npu.npu_scatter_pa_kv_cache(
        key=k_pe.squeeze(2),
        value=k_pe.squeeze(2),
        slot_mapping=slots.to(torch.int64),
        key_cache=kv_cache[1],
        value_cache=kv_cache[1],
    )
    return k_pe, k_nope


def _run_fused_op(
    kv_no_split: torch.Tensor,
    layernorm_weight: torch.Tensor,
    epsilon: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    fa_kscale: torch.Tensor,
    cache_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
        kv_no_split,
        layernorm_weight,
        cos,
        sin,
        slots.to(torch.int64),
        kv_cache[1],
        kv_cache[0],
        c_kv_scale=fa_kscale,
        epsilon=epsilon,
        cache_mode=cache_mode,
        is_output_kv=True,
    )
    return k_pe, k_nope


def _collect_touched_blocks(slots: torch.Tensor, block_size: int) -> list[int]:
    slots_cpu = slots.detach().cpu().to(torch.int64)
    return sorted({int(slot.item()) // block_size for slot in slots_cpu})


def _take_blocks(tensor: torch.Tensor, block_ids: list[int]) -> torch.Tensor:
    if not block_ids:
        return tensor[:0].clone()
    if len(block_ids) == 1:
        return tensor.narrow(0, block_ids[0], 1).contiguous()
    return torch.cat([tensor.narrow(0, block_id, 1) for block_id in block_ids], dim=0).contiguous()


@torch.inference_mode()
def test_fused_kv_rmsnorm_rope_cache_matches_small_ops_golden():
    _require_a5()

    torch.manual_seed(20260422)
    torch.npu.set_device("npu:0")
    torch.npu.config.allow_internal_format = True

    token_num = 7
    num_kv_heads = 1
    latent_dim = 512
    rope_dim = 64
    total_dim = latent_dim + rope_dim
    num_blocks = 128
    block_size = 128
    epsilon = 1e-6
    device = torch.device("npu:0")

    kv_no_split = torch.randn((token_num, num_kv_heads, 1, total_dim), dtype=torch.bfloat16, device=device)
    layernorm_weight = torch.randn((latent_dim,), dtype=torch.bfloat16, device=device)
    cos = torch.randn((token_num, num_kv_heads, 1, rope_dim), dtype=torch.bfloat16, device=device)
    sin = torch.randn((token_num, num_kv_heads, 1, rope_dim), dtype=torch.bfloat16, device=device)
    slots = torch.tensor([5, 127, 128, 255, 511, 1023, 4095], dtype=torch.int32, device=device)
    fa_kscale = torch.rand((1,), dtype=torch.float32, device=device) + 0.5

    fused_k_cache = torch.zeros((num_blocks, block_size, num_kv_heads, latent_dim), dtype=torch.float8_e4m3fn, device=device)
    fused_rope_cache = torch.zeros((num_blocks, block_size, num_kv_heads, rope_dim), dtype=torch.bfloat16, device=device)
    golden_k_cache = torch.zeros_like(fused_k_cache)
    golden_rope_cache = torch.zeros_like(fused_rope_cache)

    golden_k_pe, golden_k_nope = _run_small_op_golden(
        kv_no_split=kv_no_split.clone(),
        layernorm_weight=layernorm_weight,
        epsilon=epsilon,
        cos=cos,
        sin=sin,
        slots=slots,
        kv_cache=(golden_k_cache, golden_rope_cache),
        fa_kscale=fa_kscale,
    )
    fused_k_pe, fused_k_nope = _run_fused_op(
        kv_no_split=kv_no_split.clone(),
        layernorm_weight=layernorm_weight,
        epsilon=epsilon,
        cos=cos,
        sin=sin,
        slots=slots,
        kv_cache=(fused_k_cache, fused_rope_cache),
        fa_kscale=fa_kscale,
        cache_mode="PA",
    )

    torch.testing.assert_close(fused_k_pe.float(), golden_k_pe.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fused_k_nope.float(), golden_k_nope.float(), atol=2e-2, rtol=2e-2)

    touched_blocks = _collect_touched_blocks(slots, block_size)
    if touched_blocks:
        fused_k_cache_touched = _take_blocks(fused_k_cache, touched_blocks)
        golden_k_cache_touched = _take_blocks(golden_k_cache, touched_blocks)
        fused_rope_cache_touched = _take_blocks(fused_rope_cache, touched_blocks)
        golden_rope_cache_touched = _take_blocks(golden_rope_cache, touched_blocks)

        torch.testing.assert_close(
            fused_k_cache_touched.float(),
            golden_k_cache_touched.float(),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            fused_rope_cache_touched.float(),
            golden_rope_cache_touched.float(),
            atol=2e-2,
            rtol=2e-2,
        )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
