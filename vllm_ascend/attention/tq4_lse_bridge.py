#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ctypes bridge: TQ4 (quant_mode=3) SFA with softmax LSE via the aclnn V2 op.

Backed by aclnnKvQuantSparseFlashAttentionV2 in libopapi.so (CANN 9.3.0).
Kernel-level verification on this box (910B3, TQ4 slots, PA_BSND + TND):
5/5 cases ALL_PASS with ~1e-8 accuracy, empty-row sentinel max=-2e38/sum=1,
sinks sentinel max=sinks/sum=1 (see /root/tq4dcp_ws/tq4_v2_lse_aclnn.py).

Output contract matches what vllm-ascend's DCP decode merge expects from
`_C_ascend.npu_kv_quant_sparse_flash_attention_vllm(return_softmax_lse=True)`:
- attention_out: [T, N1, D_nope] bf16 (TND)
- softmax_max / softmax_sum: [N2=1, T, N1] fp32 (csrc infershape: dim0=N2=1,
  dim1=T, dim2=N1) — sfa_cp.py permutes (1,0,2) then merges via dcp_a2a_fused.
"""
import ctypes as ct
import os

import torch

CANN = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")
# V2 lives in the built-in opapi (checked: symbol exported by libopapi.so),
# the transformer-specific opapi also ships it.
OPAPI_CANDIDATES = (
    f"{CANN}/lib64/libopapi.so",
    f"{CANN}/lib64/libopapi_transformer.so",
    "/usr/local/Ascend/cann-9.3.0/lib64/libopapi.so",
)

ACL_FLOAT, ACL_FLOAT16, ACL_INT8, ACL_INT32, ACL_UINT8, ACL_BF16 = 0, 1, 2, 3, 4, 27
ACL_FORMAT_ND = 2
_ACL_DTYPES = {
    torch.float32: ACL_FLOAT,
    torch.float16: ACL_FLOAT16,
    torch.int8: ACL_INT8,
    torch.int32: ACL_INT32,
    torch.uint8: ACL_UINT8,
    torch.bfloat16: ACL_BF16,
}

_libcache: dict = {}


def _load():
    if _libcache:
        return _libcache
    nn = ct.CDLL(f"{CANN}/lib64/libnnopbase.so", mode=ct.RTLD_GLOBAL)
    nn.aclCreateTensor.restype = ct.c_void_p
    nn.aclCreateTensor.argtypes = [
        ct.POINTER(ct.c_int64), ct.c_uint64, ct.c_int,
        ct.POINTER(ct.c_int64), ct.c_int64, ct.c_int,
        ct.POINTER(ct.c_int64), ct.c_uint64, ct.c_void_p,
    ]
    nn.aclDestroyTensor.restype = ct.c_int
    nn.aclDestroyTensor.argtypes = [ct.c_void_p]
    last_err = None
    lib = None
    for path in OPAPI_CANDIDATES:
        try:
            lib = ct.CDLL(path, mode=ct.RTLD_GLOBAL)
            getattr(lib, "aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize")
            getattr(lib, "aclnnKvQuantSparseFlashAttentionV2")
            break
        except (OSError, AttributeError) as e:
            last_err = e
            lib = None
    if lib is None:
        raise RuntimeError(f"aclnnKvQuantSparseFlashAttentionV2 not found: {last_err}")
    ws = lib.aclnnKvQuantSparseFlashAttentionV2GetWorkspaceSize
    ws.restype = ct.c_int
    ws.argtypes = (
        [ct.c_void_p] * 10  # q, k, v, indices, k_scale, v_scale, block_table, aq, akv, sinks
        + [ct.c_double, ct.c_int64, ct.c_int64, ct.c_int64]  # scale, kqm, vqm, sbs
        + [ct.c_char_p, ct.c_char_p]  # layout_query, layout_kv
        + [ct.c_int64] * 7  # sparse_mode, pre_tokens, next_tokens, attention_mode, qsrm, tile_size, rope_head_dim
        + [ct.c_bool]  # return_softmax_lse
        + [ct.c_void_p] * 3  # attention_out, softmax_max, softmax_sum
        + [ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_void_p)]
    )
    run = lib.aclnnKvQuantSparseFlashAttentionV2
    run.restype = ct.c_int
    run.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_void_p]
    _libcache.update(nn=nn, ws=ws, run=run)
    return _libcache


def _make_acl_tensor(t: torch.Tensor):
    nn = _libcache["nn"]
    dims = list(t.shape)
    n = len(dims)
    strides, acc = [], 1
    for d in reversed(dims):
        strides.append(acc)
        acc *= d
    strides.reverse()
    darr = (ct.c_int64 * n)(*dims)
    sarr = (ct.c_int64 * n)(*strides)
    return nn.aclCreateTensor(darr, n, _ACL_DTYPES[t.dtype], sarr, 0,
                              ACL_FORMAT_ND, darr, n, ct.c_void_p(t.data_ptr()))


def tq4_sfa_with_lse(
    query: torch.Tensor,
    kv: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    *,
    scale: float,
    sparse_mode: int,
    rope_head_dim: int,
    tile_size: int = 128,
):
    """TQ4 sparse flash attention returning (out, softmax_max, softmax_sum).

    Inputs follow the torch_npu npu_kv_quant_sparse_flash_attention calling
    convention used in sfa_v1._turboquant_sfa: TND query [T, N1, 576] bf16,
    PA_BSND packed KV cache int8 [blocks, 1, slot_bytes] (or the page-strided
    view thereof), TND sparse_indices int32, quant_mode=3, attention_mode=2.
    """
    libs = _load()
    query = query.contiguous()
    kv = kv.contiguous() if not kv.is_contiguous() else kv
    topk_indices = topk_indices.contiguous()
    block_table = block_table.contiguous() if block_table is not None else None
    actual_seq_lengths_query = actual_seq_lengths_query.contiguous()
    actual_seq_lengths_kv = actual_seq_lengths_kv.contiguous()

    n_heads = query.shape[1]
    n_tokens = query.shape[0]
    out_dim = query.shape[2] - rope_head_dim

    attention_out = torch.empty((n_tokens, n_heads, out_dim),
                                dtype=query.dtype, device=query.device)
    # V2 LSE shapes (TND + PA_BSND): [N2, T, N1]; N2=1 for MLA (single KV head).
    softmax_max = torch.empty((1, n_tokens, n_heads), dtype=torch.float32,
                              device=query.device)
    softmax_sum = torch.empty((1, n_tokens, n_heads), dtype=torch.float32,
                              device=query.device)

    handles = []

    def T(x):
        if x is None:
            return None
        h = _make_acl_tensor(x)
        handles.append(h)
        return h

    try:
        hs = [T(query), T(kv), T(kv), T(topk_indices), T(None), T(None),
              T(block_table), T(actual_seq_lengths_query),
              T(actual_seq_lengths_kv), T(None)]
        oh, mh, sh = T(attention_out), T(softmax_max), T(softmax_sum)
        ws_size = ct.c_uint64(0)
        executor = ct.c_void_p()
        rc = libs["ws"](*hs,
                        ct.c_double(scale), ct.c_int64(3), ct.c_int64(3),
                        ct.c_int64(1),  # sparse_block_size
                        b"TND", b"PA_BSND",
                        ct.c_int64(sparse_mode),
                        ct.c_int64((1 << 63) - 1),  # pre_tokens
                        ct.c_int64((1 << 63) - 1),  # next_tokens
                        ct.c_int64(2),  # attention_mode
                        ct.c_int64(1),  # quant_scale_repo_mode
                        ct.c_int64(tile_size),
                        ct.c_int64(rope_head_dim),
                        ct.c_bool(True),  # return_softmax_lse
                        oh, mh, sh,
                        ct.byref(ws_size), ct.byref(executor))
        if rc != 0:
            raise RuntimeError(f"V2 GetWorkspaceSize rc={rc}")
        ws_ptr = None
        ws_t = None
        if ws_size.value > 0:
            ws_t = torch.empty(ws_size.value, dtype=torch.uint8, device=query.device)
            ws_ptr = ct.c_void_p(ws_t.data_ptr())
        stream = ct.c_void_p(torch.npu.current_stream().npu_stream)
        rc = libs["run"](ws_ptr, ws_size.value, executor, stream)
        if rc != 0:
            raise RuntimeError(f"aclnnKvQuantSparseFlashAttentionV2 launch rc={rc}")
    finally:
        for h in handles:
            libs["nn"].aclDestroyTensor(ct.c_void_p(h))

    return attention_out, softmax_max, softmax_sum
