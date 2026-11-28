"""Dense paged FlashMLA inference on Ascend A3."""

import math
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import torch

_INT_MAX = 2**31 - 1
_LIBRARY_DIR = Path(__file__).resolve().parents[2] / "lib" / "flash_mla_a3"


@lru_cache(maxsize=1)
def _ops():
    # Register the NPU dispatcher and current stream before loading the operator.
    import torch_npu  # noqa: F401

    library = _LIBRARY_DIR / "libflash_mla_a3_ops.so"
    if not library.is_file():
        raise RuntimeError(f"Missing packaged FlashMLA library {library}")
    torch.ops.load_library(str(library))
    return torch.ops.flash_mla_a3


@lru_cache(maxsize=1)
def _has_triton_ascend():
    try:
        return find_spec("triton.backends.ascend") is not None
    except ModuleNotFoundError:
        return False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(x, name, device=None, dtype=None):
    _require(isinstance(x, torch.Tensor), f"{name} must be a tensor")
    _require(
        x.device.type == "npu" and (device is None or x.device == device), f"{name} must be on {device or 'an NPU'}"
    )
    if dtype is not None:
        _require(x.dtype == dtype, f"{name} must have dtype {dtype}")


def _plain(x):
    # A framework-level contiguous view may still use an internal blocked NPU
    # storage format. The native kernel expects linear storage after data_ptr().
    import torch_npu

    if torch_npu.get_npu_format(x) not in (0, 2):
        x = torch_npu.npu_format_cast(x, 2)
    x = x.contiguous()
    if x.storage_offset() * x.element_size() % 32:
        x = x.clone()
    return x


def _optional_vector(x, size, name, device, dtype):
    if x is None:
        return None
    _tensor(x, name, device, dtype)
    _require(x.ndim == 1 and x.numel() in (0, size), f"{name} must have shape ({size},) or (0,)")
    return None if x.numel() == 0 else _plain(x)


def _range(value, name):
    value = -1 if value is None else value
    _require(isinstance(value, int) and -1 <= value <= _INT_MAX, f"{name} must be in [-1, INT32_MAX]")
    return value


def _mask(mode, left, right):
    mode = 0 if mode is None else mode
    _require(mode in (0, 3, 4), "mask_mode must be 0, 3 or 4")
    _range(left, "win_left")
    _range(right, "win_right")
    _require(mode == 4 or (left == -1 and right == -1), "windows require mask_mode=4")
    return mode


def _lengths(kv, cu, used, layout, max_q, max_kv):
    _tensor(kv, "cache_seqlens", dtype=torch.int32)
    _require(kv.ndim == 1 and 0 < kv.numel() < 65536, "cache_seqlens must have shape [B], 0 < B < 65536")
    _require(layout in ("BSND", "BNSD", "TND"), "layout_q must be BSND, BNSD or TND")
    cu = _optional_vector(cu, kv.numel() + 1, "cu_seqlens_q", kv.device, torch.int32)
    used = _optional_vector(used, kv.numel(), "seqused_q", kv.device, torch.int32)
    _require((layout == "TND") == (cu is not None), "only TND requires nonempty cu_seqlens_q")
    max_q = _range(max_q, "max_seqlen_q")
    max_kv = _range(max_kv, "max_seqlen_kv")
    _require(
        cu is not None or used is not None or max_q >= 0,
        "supply max_seqlen_q when both query length tensors are absent",
    )
    return _plain(kv), cu, used, max_q, max_kv


def flash_mla_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
):
    """Build opaque A3 metadata on the calling NPU stream for these lengths."""
    _require(1 <= num_heads_q <= 128 and num_heads_kv == 1, "expected 1..128 query heads and one KV head")
    _require((head_dim_qk, head_dim_v) == (576, 512), "expected QK dimension 576 and V dimension 512")
    _mask(mask_mode, win_left, win_right)
    lengths = _lengths(cache_seqlens, cu_seqlens_q, seqused_q, layout_q or "BSND", max_seqlen_q, max_seqlen_kv)
    return _ops().plan(*lengths).view(-1)


def _single_mtp_tail(batches, tokens, width, causal):
    # At three pages per group, 73–96 pages leave 25–32 groups for 24 Cube cores.
    return batches == 1 and tokens == 4 and causal and 72 < width <= 96


def _paged_cache(cache, layout):
    if layout == "PA_NZ":
        _require(
            cache.ndim == 5 and cache.shape[1:3] == (1, 36) and cache.size(4) == 16,
            "PA_NZ expects [pages, 1, 36, block_size, 16]",
        )
        block = cache.size(3)
        cache = _plain(cache).squeeze(1).transpose(1, 2).contiguous().view(cache.size(0), block, 576)
    else:
        _require(layout in ("PA_BBND", "PA_BNBD"), "unsupported layout_kv")
        kv_axis = 2 if layout == "PA_BBND" else 1
        _require(
            cache.ndim == 4 and cache.size(kv_axis) == 1 and cache.size(-1) == 576,
            f"{layout} expects a single KV head and a final dimension of 576",
        )
        cache = _plain(cache).squeeze(kv_axis)
        block = cache.size(1)
    _require(
        0 < cache.size(0) <= _INT_MAX and 16 <= block <= 1024 and block % 16 == 0,
        "expected nonempty cache and a block_size divisible by 16 in [16, 1024]",
    )
    return cache, block


def flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
):
    """Return paged MLA output and FP32 LSE (an empty tensor when disabled)."""
    layout_q, layout_out = layout_q or "BSND", layout_out or "BSND"
    packed = layout_q == "TND"
    _require(
        layout_out == layout_q or (packed and layout_out == "NTD"), "layout_out must match layout_q or be NTD for TND"
    )
    _tensor(q, "q")
    _tensor(k_cache, "k_cache", q.device, q.dtype)
    _tensor(cache_seqlens, "cache_seqlens", q.device, torch.int32)
    _tensor(block_table, "block_table", q.device, torch.int32)
    _require(q.dtype in (torch.float16, torch.bfloat16) and head_dim_v == 512, "expected FP16/BF16 and head_dim_v=512")
    _require(
        q.ndim == (3 if packed else 4) and q.size(-1) == 576 and q.numel() > 0,
        "q must be nonempty with the declared query layout and QK dimension 576",
    )
    kv, cu, used, max_q, max_kv = _lengths(
        cache_seqlens, cu_seqlens_q, seqused_q, layout_q, max_seqlen_q, max_seqlen_kv
    )
    batches = kv.numel()
    if not packed:
        _require(q.size(0) == batches, "q batch dimension must match cache_seqlens")
        seq = q.size(2 if layout_q == "BNSD" else 1)
        _require(
            used is not None or max_q == seq, "max_seqlen_q must match physical query length when seqused_q is absent"
        )
    q = _plain(q)
    if layout_q == "BNSD":
        q = q.transpose(1, 2).contiguous()
    heads = q.size(-2)
    _require(1 <= heads <= 128, "expected 1..128 query heads")
    q = q.view(-1, heads, 576)
    tokens = q.size(0)
    _require(tokens <= _INT_MAX, "token count exceeds INT32_MAX")
    cache, block = _paged_cache(k_cache, layout_kv or "PA_BBND")
    _require(
        block_table.ndim == 2 and block_table.size(0) == batches and block_table.size(1) > 0,
        "block_table must have shape [B, positive page count]",
    )
    capacity = block_table.size(1) * block
    _require(capacity <= _INT_MAX, "per-sequence cache capacity exceeds INT32_MAX")
    mode = _mask(mask_mode, win_left, win_right)
    if mode == 0:
        _require(attn_mask is None, "omit attn_mask for mask_mode=0")
    else:
        _tensor(attn_mask, "attn_mask", q.device, torch.int8)
        _require(attn_mask.shape == (2048, 2048), "expected the 2048 x 2048 upper-triangular mask template")
    sinks = _optional_vector(sinks, heads, "sinks", q.device, torch.float32)
    scale = 1.0 if softmax_scale is None else float(softmax_scale)
    _require(math.isfinite(scale), "softmax_scale must be finite")
    ops = _ops()
    plan = None
    if metadata is not None:
        _tensor(metadata, "metadata", q.device, torch.int32)
        _require(metadata.shape == ((batches + 1) * 4,), "metadata must come from flash_mla_metadata for this batch")
        plan = _plain(metadata).view(batches + 1, 4)
    table = _plain(block_table)
    short_query = (
        q.dtype == torch.bfloat16
        and heads == 4
        and block == 128
        and mode in (0, 3)
        and sinks is None
        and tokens <= 4 * batches
    )
    lengths_checked = plan is not None or (max_kv == -1 and (max_q == -1 or (not packed and max_q == seq)))
    # Each query/head may need two FP32 partials per physical KV page.
    partial_bytes = tokens * heads * table.size(1) * 2 * 512 * 4
    accelerated = short_query and not return_softmax_lse and lengths_checked and _has_triton_ascend()
    # CATLASS reads each paged KV tile into shared L1 for both matrix products.
    # GPU preparation handles ragged packed rows and returns a real contiguous layout.
    catlass = accelerated and 16 <= table.size(1) <= 1024 and batches <= 256
    cube = accelerated and table.size(1) <= 1024 and partial_bytes <= 768 * 1024**2
    grouped_ragged = (batches, table.size(1)) in ((16, 8), (4, 32)) and tokens in (batches, 4 * batches)
    grouped = accelerated and (
        grouped_ragged
        or (
            batches == 1
            and 1 < table.size(1) <= (128 if tokens > 1 else 32)
            and not _single_mtp_tail(batches, tokens, table.size(1), mode == 3)
        )
    )
    if grouped:
        from .group_decode import group_decode

        out = group_decode(
            q,
            cache,
            table,
            kv,
            tokens // batches,
            scale,
            mode == 3,
            layout_out == "NTD",
            plan=plan,
            cu_seqlens_q=cu,
            seqused_q=used,
            packed=packed,
            ragged=grouped_ragged,
        )
        lse = q.new_empty((0,), dtype=torch.float32)
    elif catlass or cube:
        if catlass:
            from .catlass_decode import catlass_decode
            from .prepared_decode import catlass_ops

            catlass_ops()
            compute = catlass_decode
        else:
            from .cube_decode import cube_decode

            compute = cube_decode
        out = compute(q, cache, table, kv, cu, used, plan, scale, mode == 3, layout_out == "NTD")
        lse = q.new_empty((0,), dtype=torch.float32)
    elif short_query and not packed and used is None and plan is None and max_kv == -1:
        # The dense layout guarantees a common S; split-KV can directly read
        # lengths, including when LSE is requested or Triton is unavailable.
        out, lse = ops.decode(q, cache, table, kv, seq, scale, mode == 3, bool(return_softmax_lse))
    else:
        if plan is None:
            plan = ops.plan(kv, cu, used, max_q, max_kv)
        bounds = ops.bounds(plan, tokens, packed, mode, win_left, win_right, capacity)
        out, lse = ops.forward(q, cache, table, bounds, sinks, block, scale, bool(return_softmax_lse))
    if packed:
        if layout_out == "NTD" and not (grouped or catlass or cube):
            out = out.transpose(0, 1).contiguous()
        if return_softmax_lse:
            lse = lse.view(tokens, heads).transpose(0, 1).contiguous()
    else:
        out = out.view(batches, seq, heads, 512)
        if layout_out == "BNSD":
            out = out.transpose(1, 2).contiguous()
        if return_softmax_lse:
            lse = lse.view(batches, seq, heads).transpose(1, 2).contiguous()
    return out, lse
