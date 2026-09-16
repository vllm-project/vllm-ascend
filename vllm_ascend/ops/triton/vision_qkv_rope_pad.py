import torch
from vllm.logger import logger
from vllm.triton_utils import tl, triton

from vllm_ascend import envs

_NUM_VECTOR_CORES_910B3 = 40
_DEFAULT_BLOCK_T = 4
# Upper bound for BLOCK_T: beyond it the BiShengIR local-buffer (UB) budget
# of the 910B3 vector cores is exceeded (e.g. 2752512 bits required vs
# 1572864 available at BLOCK_T=16) and kernel compilation fails.
_MAX_BLOCK_T = 8
# Divisors of 40 used by the "bucket" core mode so the per-core tile split
# stays even for every bucket.
_CORE_BUCKETS = (1, 2, 4, 5, 8, 10, 20, 40)

_BLOCK_T_CLAMP_WARNED = False


def _get_num_cores(token_count: int, block_t: int, core_mode: str) -> int:
    """Choose the vector-core count for the token tiles.

    - ``dynamic``: launch exactly one core per token tile (capped at 40);
    - ``fixed``: always launch all 40 vector cores;
    - ``bucket``: round the dynamic count up to a divisor of 40 so idle cores
      stay aligned with the hardware core groups.
    """
    token_block_count = max(1, (token_count + block_t - 1) // block_t)
    dynamic = min(_NUM_VECTOR_CORES_910B3, token_block_count)
    if core_mode == "fixed":
        return _NUM_VECTOR_CORES_910B3
    if core_mode == "bucket":
        for bucket in _CORE_BUCKETS:
            if bucket >= dynamic:
                return bucket
        return _NUM_VECTOR_CORES_910B3
    return dynamic


@triton.jit(do_not_specialize=["token_count", "token_block_count", "blocks_per_core"])
def _vision_qkv_rope_pad_kernel(
    qkv_proj,
    cos_half,
    sin_half,
    q_out,
    k_out,
    v_out,
    token_count,
    token_block_count,
    blocks_per_core,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    HALF_TILE: tl.constexpr,
    PAD_TILE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    STORE_MASKED: tl.constexpr,
    STORE_SINGLE: tl.constexpr,
):
    core_id = tl.program_id(0)
    qkv_token_stride = 3 * NUM_HEADS * HEAD_DIM
    plane_stride = NUM_HEADS * HEAD_DIM
    out_token_stride = NUM_HEADS * PADDED_DIM

    block_start = core_id * blocks_per_core
    block_end = tl.minimum(block_start + blocks_per_core, token_block_count)
    for block_id in tl.range(block_start, block_end):
        token_start = block_id * BLOCK_T

        # Load the half-width tables once per token tile. Boundary padding makes
        # the 64-column DMA safe while retaining the profiled logical width 36.
        p_cos = tl.make_block_ptr(
            cos_half,
            (token_count, HALF_DIM),
            (HALF_DIM, 1),
            (token_start, 0),
            (BLOCK_T, HALF_TILE),
            (1, 0),
        )
        p_sin = tl.make_block_ptr(
            sin_half,
            (token_count, HALF_DIM),
            (HALF_DIM, 1),
            (token_start, 0),
            (BLOCK_T, HALF_TILE),
            (1, 0),
        )
        cos = tl.load(p_cos, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        sin = tl.load(p_sin, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        cos = cos[:, None, :]
        sin = sin[:, None, :]

        if STORE_SINGLE:
            # Full-width formulation: build the complete 128-column result in
            # registers and issue exactly one store per Q/K plane. The partner
            # tensor (rotate_half source) is gathered directly, and the
            # cos/sin tables are expanded to 128 columns, so no second store
            # or zero-fill store is needed.
            offs_t = token_start + tl.arange(0, BLOCK_T)
            offs_h = tl.arange(0, NUM_HEADS)
            offs_j = tl.arange(0, PADDED_DIM)
            t_mask = offs_t[:, None, None] < token_count
            col_mask = offs_j[None, None, :] < HEAD_DIM
            load_mask = t_mask & col_mask

            j = offs_j[None, None, :]
            jmod = tl.where(j < HALF_DIM, j, tl.where(j < HEAD_DIM, j - HALF_DIM, 0))
            sign = tl.where(j < HALF_DIM, -1.0, 1.0)

            for plane in tl.static_range(2):
                plane_base = qkv_proj + plane * plane_stride
                out_base = q_out if plane == 0 else k_out
                qkv_offsets = offs_t[:, None, None] * qkv_token_stride + offs_h[None, :, None] * HEAD_DIM + j
                partner_j = tl.where(j < HALF_DIM, j + HALF_DIM, tl.where(j < HEAD_DIM, j - HALF_DIM, 0))
                partner_offsets = (
                    offs_t[:, None, None] * qkv_token_stride + offs_h[None, :, None] * HEAD_DIM + partner_j
                )
                x = tl.load(plane_base + qkv_offsets, mask=load_mask, other=0.0).to(tl.float32)
                partner = tl.load(plane_base + partner_offsets, mask=load_mask, other=0.0).to(tl.float32)
                table_offsets = offs_t[:, None, None] * HALF_DIM + jmod
                cos_full = tl.load(cos_half + table_offsets, mask=load_mask, other=0.0).to(tl.float32)
                sin_full = tl.load(sin_half + table_offsets, mask=load_mask, other=0.0).to(tl.float32)
                rotated = x * cos_full + partner * sin_full * sign
                out_offsets = (
                    offs_t[:, None, None] * out_token_stride
                    + offs_h[None, :, None] * PADDED_DIM
                    + offs_j[None, None, :]
                )
                tl.store(out_base + out_offsets, rotated.to(tl.bfloat16), mask=t_mask)
        else:
            # Q first/second halves are two contiguous block loads. This avoids the
            # gather generated by a virtual rotate_half index tensor.
            p_q0 = tl.make_block_ptr(
                qkv_proj,
                (token_count, NUM_HEADS, HEAD_DIM),
                (qkv_token_stride, HEAD_DIM, 1),
                (token_start, 0, 0),
                (BLOCK_T, NUM_HEADS, HALF_TILE),
                (2, 1, 0),
            )
            p_q1 = tl.make_block_ptr(
                qkv_proj,
                (token_count, NUM_HEADS, HEAD_DIM),
                (qkv_token_stride, HEAD_DIM, 1),
                (token_start, 0, HALF_DIM),
                (BLOCK_T, NUM_HEADS, HALF_TILE),
                (2, 1, 0),
            )
            q0 = tl.load(p_q0, boundary_check=(0, 2), padding_option="zero").to(tl.float32)
            q1 = tl.load(p_q1, boundary_check=(0, 2), padding_option="zero").to(tl.float32)
            q_rope0 = (q0 * cos - q1 * sin).to(tl.bfloat16)
            q_rope1 = (q1 * cos + q0 * sin).to(tl.bfloat16)

            # K has the same contiguous two-half mapping at plane offset H*D.
            p_k0 = tl.make_block_ptr(
                qkv_proj + plane_stride,
                (token_count, NUM_HEADS, HEAD_DIM),
                (qkv_token_stride, HEAD_DIM, 1),
                (token_start, 0, 0),
                (BLOCK_T, NUM_HEADS, HALF_TILE),
                (2, 1, 0),
            )
            p_k1 = tl.make_block_ptr(
                qkv_proj + plane_stride,
                (token_count, NUM_HEADS, HEAD_DIM),
                (qkv_token_stride, HEAD_DIM, 1),
                (token_start, 0, HALF_DIM),
                (BLOCK_T, NUM_HEADS, HALF_TILE),
                (2, 1, 0),
            )
            k0 = tl.load(p_k0, boundary_check=(0, 2), padding_option="zero").to(tl.float32)
            k1 = tl.load(p_k1, boundary_check=(0, 2), padding_option="zero").to(tl.float32)
            k_rope0 = (k0 * cos - k1 * sin).to(tl.bfloat16)
            k_rope1 = (k1 * cos + k0 * sin).to(tl.bfloat16)

            if STORE_MASKED:
                # Non-overlapping masked stores: 36 + 36 + 56 = 128 columns
                # per plane, removing the redundant column traffic of the
                # block-store layout.
                offs_t = token_start + tl.arange(0, BLOCK_T)
                offs_h = tl.arange(0, NUM_HEADS)
                half_j = tl.arange(0, HALF_TILE)
                t_mask = offs_t[:, None, None] < token_count
                half_mask = t_mask & (half_j[None, None, :] < HALF_DIM)
                out_base_offs = offs_t[:, None, None] * out_token_stride + offs_h[None, :, None] * PADDED_DIM
                pad_j = tl.arange(0, PAD_TILE * 2)
                pad_mask = t_mask & (pad_j[None, None, :] < PADDED_DIM - HEAD_DIM)
                zero_pad = tl.zeros((BLOCK_T, NUM_HEADS, PAD_TILE * 2), dtype=tl.bfloat16)
                tl.store(q_out + out_base_offs + half_j, q_rope0, mask=half_mask)
                tl.store(q_out + out_base_offs + HALF_DIM + half_j, q_rope1, mask=half_mask)
                tl.store(q_out + out_base_offs + HEAD_DIM + pad_j, zero_pad, mask=pad_mask)
                tl.store(k_out + out_base_offs + half_j, k_rope0, mask=half_mask)
                tl.store(k_out + out_base_offs + HALF_DIM + half_j, k_rope1, mask=half_mask)
                tl.store(k_out + out_base_offs + HEAD_DIM + pad_j, zero_pad, mask=pad_mask)
            else:
                # Block stores. q1/k1 stores cover pad columns 72:99 with
                # zeros; the final store finishes 96:127.
                p_qo0 = tl.make_block_ptr(
                    q_out,
                    (token_count, NUM_HEADS, PADDED_DIM),
                    (out_token_stride, PADDED_DIM, 1),
                    (token_start, 0, 0),
                    (BLOCK_T, NUM_HEADS, HALF_TILE),
                    (2, 1, 0),
                )
                p_qo1 = tl.make_block_ptr(
                    q_out,
                    (token_count, NUM_HEADS, PADDED_DIM),
                    (out_token_stride, PADDED_DIM, 1),
                    (token_start, 0, HALF_DIM),
                    (BLOCK_T, NUM_HEADS, HALF_TILE),
                    (2, 1, 0),
                )
                tl.store(p_qo0, q_rope0, boundary_check=(0, 2))
                tl.store(p_qo1, q_rope1, boundary_check=(0, 2))

                p_ko0 = tl.make_block_ptr(
                    k_out,
                    (token_count, NUM_HEADS, PADDED_DIM),
                    (out_token_stride, PADDED_DIM, 1),
                    (token_start, 0, 0),
                    (BLOCK_T, NUM_HEADS, HALF_TILE),
                    (2, 1, 0),
                )
                p_ko1 = tl.make_block_ptr(
                    k_out,
                    (token_count, NUM_HEADS, PADDED_DIM),
                    (out_token_stride, PADDED_DIM, 1),
                    (token_start, 0, HALF_DIM),
                    (BLOCK_T, NUM_HEADS, HALF_TILE),
                    (2, 1, 0),
                )
                tl.store(p_ko0, k_rope0, boundary_check=(0, 2))
                tl.store(p_ko1, k_rope1, boundary_check=(0, 2))

                zero_pad = tl.zeros((BLOCK_T, NUM_HEADS, PAD_TILE), dtype=tl.bfloat16)
                p_qpad = tl.make_block_ptr(
                    q_out,
                    (token_count, NUM_HEADS, PADDED_DIM),
                    (out_token_stride, PADDED_DIM, 1),
                    (token_start, 0, PADDED_DIM - PAD_TILE),
                    (BLOCK_T, NUM_HEADS, PAD_TILE),
                    (2, 1, 0),
                )
                p_kpad = tl.make_block_ptr(
                    k_out,
                    (token_count, NUM_HEADS, PADDED_DIM),
                    (out_token_stride, PADDED_DIM, 1),
                    (token_start, 0, PADDED_DIM - PAD_TILE),
                    (BLOCK_T, NUM_HEADS, PAD_TILE),
                    (2, 1, 0),
                )
                tl.store(p_qpad, zero_pad, boundary_check=(0, 2))
                tl.store(p_kpad, zero_pad, boundary_check=(0, 2))

        # V needs no RoPE; load logical D=72 and let boundary padding form 128.
        p_v = tl.make_block_ptr(
            qkv_proj + 2 * plane_stride,
            (token_count, NUM_HEADS, HEAD_DIM),
            (qkv_token_stride, HEAD_DIM, 1),
            (token_start, 0, 0),
            (BLOCK_T, NUM_HEADS, PADDED_DIM),
            (2, 1, 0),
        )
        p_vo = tl.make_block_ptr(
            v_out,
            (token_count, NUM_HEADS, PADDED_DIM),
            (out_token_stride, PADDED_DIM, 1),
            (token_start, 0, 0),
            (BLOCK_T, NUM_HEADS, PADDED_DIM),
            (2, 1, 0),
        )
        v = tl.load(p_v, boundary_check=(0, 2), padding_option="zero")
        tl.store(p_vo, v, boundary_check=(0, 2))


def _resolve_launch_options() -> tuple[int, str, str, str]:
    core_mode = envs.VLLM_ASCEND_VIT_ROPE_PAD_CORE_MODE
    if core_mode not in ("dynamic", "fixed", "bucket"):
        core_mode = "dynamic"
    store_mode = envs.VLLM_ASCEND_VIT_ROPE_PAD_STORE_MODE
    if store_mode not in ("block", "masked", "single"):
        store_mode = "block"
    alloc_mode = envs.VLLM_ASCEND_VIT_ROPE_PAD_ALLOC_MODE
    if alloc_mode not in ("fused", "split"):
        alloc_mode = "fused"
    try:
        block_t = int(envs.VLLM_ASCEND_VIT_ROPE_PAD_BLOCK_T)
    except (TypeError, ValueError):
        block_t = _DEFAULT_BLOCK_T
    if block_t <= 0 or block_t & (block_t - 1) != 0:
        block_t = _DEFAULT_BLOCK_T
    if block_t > _MAX_BLOCK_T:
        global _BLOCK_T_CLAMP_WARNED
        if not _BLOCK_T_CLAMP_WARNED:
            _BLOCK_T_CLAMP_WARNED = True
            logger.warning(
                "VLLM_ASCEND_VIT_ROPE_PAD_BLOCK_T=%d exceeds the local buffer "
                "budget of the 910B3 vector cores; clamping to %d.",
                block_t,
                _MAX_BLOCK_T,
            )
        block_t = _MAX_BLOCK_T
    return block_t, core_mode, store_mode, alloc_mode


def vision_qkv_rope_pad(
    qkv_proj: torch.Tensor,
    cos_half: torch.Tensor,
    sin_half: torch.Tensor,
    q_out: torch.Tensor = None,
    k_out: torch.Tensor = None,
    v_out: torch.Tensor = None,
):
    """Generate padded TND Q/K/V tensors consumed directly by FIA.

    Expected production shapes are qkv_proj=[T, 1728] and
    cos_half=sin_half=[T, 36], all contiguous BF16 tensors on Ascend NPU.

    Launch options (A/B experiments, see ``vllm_ascend/envs.py``):

    - ``VLLM_ASCEND_VIT_ROPE_PAD_CORE_MODE``: ``dynamic`` (default) | ``fixed``
      | ``bucket`` vector-core launch strategy;
    - ``VLLM_ASCEND_VIT_ROPE_PAD_BLOCK_T``: token tile size (default 4);
    - ``VLLM_ASCEND_VIT_ROPE_PAD_STORE_MODE``: ``block`` (default) |
      ``masked`` (non-overlapping stores) | ``single`` (one full-width store
      per Q/K plane);
    - ``VLLM_ASCEND_VIT_ROPE_PAD_ALLOC_MODE``: ``fused`` (default, one buffer)
      | ``split`` (three allocations).

    The per-core tile split is computed on the host, so the kernel performs no
    runtime integer division.
    """
    if qkv_proj.ndim != 2 or cos_half.ndim != 2 or sin_half.ndim != 2:
        raise ValueError("all inputs must be rank-2")
    if qkv_proj.shape[1] != 3 * 8 * 72:
        raise ValueError("qkv_proj must have shape [T, 1728]")
    if cos_half.shape != (qkv_proj.shape[0], 36) or sin_half.shape != cos_half.shape:
        raise ValueError("cos_half and sin_half must have shape [T, 36]")
    if not (qkv_proj.is_contiguous() and cos_half.is_contiguous() and sin_half.is_contiguous()):
        raise ValueError("all inputs must be contiguous")
    if not (qkv_proj.dtype == cos_half.dtype == sin_half.dtype == torch.bfloat16):
        raise ValueError("all inputs must be torch.bfloat16")
    if not (qkv_proj.device == cos_half.device == sin_half.device):
        raise ValueError("all inputs must be on the same device")

    token_count = qkv_proj.shape[0]
    out_shape = (token_count, 8, 128)
    block_t, core_mode, store_mode, alloc_mode = _resolve_launch_options()
    provided = (q_out is not None, k_out is not None, v_out is not None)
    if any(provided) and not all(provided):
        raise ValueError("q_out, k_out and v_out must be provided together")
    if not any(provided):
        if alloc_mode == "split":
            q_out = torch.empty(out_shape, dtype=qkv_proj.dtype, device=qkv_proj.device)
            k_out = torch.empty(out_shape, dtype=qkv_proj.dtype, device=qkv_proj.device)
            v_out = torch.empty(out_shape, dtype=qkv_proj.dtype, device=qkv_proj.device)
        else:
            # Keep Q/K/V as contiguous non-overlapping views of one allocation.
            # This reduces allocator calls on the hot eager path.
            fused_out = torch.empty((3, *out_shape), dtype=qkv_proj.dtype, device=qkv_proj.device)
            q_out, k_out, v_out = fused_out.unbind(0)
    else:
        for name, out in (("q_out", q_out), ("k_out", k_out), ("v_out", v_out)):
            if out.shape != out_shape:
                raise ValueError(f"{name} must have shape {out_shape}")
            if out.dtype != qkv_proj.dtype or out.device != qkv_proj.device:
                raise ValueError(f"{name} must match the input dtype and device")
            if not out.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

    num_cores = _get_num_cores(token_count, block_t, core_mode)
    token_block_count = max(1, (token_count + block_t - 1) // block_t)
    blocks_per_core = (token_block_count + num_cores - 1) // num_cores
    _vision_qkv_rope_pad_kernel[(num_cores,)](
        qkv_proj,
        cos_half,
        sin_half,
        q_out,
        k_out,
        v_out,
        token_count,
        token_block_count,
        blocks_per_core,
        NUM_HEADS=8,
        HEAD_DIM=72,
        PADDED_DIM=128,
        HALF_DIM=36,
        HALF_TILE=64,
        PAD_TILE=32,
        BLOCK_T=block_t,
        STORE_MASKED=store_mode == "masked",
        STORE_SINGLE=store_mode == "single",
    )
    return q_out, k_out, v_out
