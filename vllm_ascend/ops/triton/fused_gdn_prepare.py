# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from triton.runtime import driver
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

# One program per vector core walks row chunks; Q/K/V sections are loaded as
# whole [BLOCK_T, H, D] blocks (sections are contiguous inside mixed_qkv).
_MAX_BLOCK_T = 16
_TARGET_SECTION_ELEMS = 4096
_NUM_STAGES = 2
_PLAN_CACHE: dict = {}


@triton.jit(do_not_specialize=["T", "NUM_CHUNKS"])
def _fused_gdn_prepare_kernel(
    # outputs
    q_out,  # [1, T, NK, DK] contiguous
    k_out,  # [1, T, NK, DK] contiguous
    v_out,  # [1, T, NV, DV] contiguous
    g_out,  # [1, T, NV] float32
    beta_out,  # [1, T, NV]
    # inputs
    mixed_qkv,  # [T, QKV_DIM] contiguous (one conv output subset)
    a_in,  # [T_ALL, NV] full batch; rows picked via INDEX when HAS_INDEX
    b_in,  # [T_ALL, NV] full batch
    INDEX,  # [T] original token id per output row (unused when HAS_INDEX=False)
    A_log,  # [NV]
    dt_bias,  # [NV]
    # sizes
    T,
    NUM_CHUNKS,
    QKV_DIM: tl.constexpr,
    Q_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    Q_OFFSET: tl.constexpr,
    K_OFFSET: tl.constexpr,
    V_OFFSET: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_VD: tl.constexpr,
    BLOCK_NV: tl.constexpr,
    HAS_INDEX: tl.constexpr,
    EPS: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_t = tl.arange(0, BLOCK_T)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, DK)
    offs_vd = tl.arange(0, BLOCK_VD)
    offs_nv = tl.arange(0, BLOCK_NV)

    h_mask = offs_h < NK
    vd_mask = offs_vd < V_DIM
    nv_mask = offs_nv < NV
    a_log = tl.load(A_log + offs_nv, mask=nv_mask, other=0.0).to(tl.float32)
    dt_val = tl.load(dt_bias + offs_nv, mask=nv_mask, other=0.0).to(tl.float32)
    neg_exp_alog = -tl.exp(a_log)

    base = pid * (NUM_CHUNKS * BLOCK_T)
    for chunk in tl.range(NUM_CHUNKS, num_stages=NUM_STAGES):
        toks = base + chunk * BLOCK_T + offs_t
        tmask = toks < T
        m3 = tmask[:, None, None] & h_mask[None, :, None]

        # Q section: [BLOCK_T, BLOCK_H, DK], l2norm over head dim
        q_vec = tl.load(
            mixed_qkv + toks[:, None, None] * QKV_DIM + Q_OFFSET + offs_h[None, :, None] * DK + offs_d[None, None, :],
            mask=m3,
            other=0.0,
        ).to(tl.float32)
        q_n = q_vec * tl.rsqrt(tl.sum(q_vec * q_vec, 2) + EPS)[:, :, None]
        tl.store(
            q_out + toks[:, None, None] * Q_DIM + offs_h[None, :, None] * DK + offs_d[None, None, :],
            q_n.to(q_out.dtype.element_ty),
            mask=m3,
        )

        # K section
        k_vec = tl.load(
            mixed_qkv + toks[:, None, None] * QKV_DIM + K_OFFSET + offs_h[None, :, None] * DK + offs_d[None, None, :],
            mask=m3,
            other=0.0,
        ).to(tl.float32)
        k_n = k_vec * tl.rsqrt(tl.sum(k_vec * k_vec, 2) + EPS)[:, :, None]
        tl.store(
            k_out + toks[:, None, None] * Q_DIM + offs_h[None, :, None] * DK + offs_d[None, None, :],
            k_n.to(k_out.dtype.element_ty),
            mask=m3,
        )

        # V section: flat contiguous copy [BLOCK_T, NV*DV]
        m2 = tmask[:, None] & vd_mask[None, :]
        v_vec = tl.load(mixed_qkv + toks[:, None] * QKV_DIM + V_OFFSET + offs_vd[None, :], mask=m2, other=0.0)
        tl.store(v_out + toks[:, None] * V_DIM + offs_vd[None, :], v_vec, mask=m2)

        # gating: [BLOCK_T, NV]. a/b stay whole: rows are picked via INDEX
        # (indirect addressing) when the token subset was gathered before
        # conv, replacing 4 host-side index_select dispatches.
        if HAS_INDEX:
            toks_src = tl.load(INDEX + toks, mask=tmask, other=0)
        else:
            toks_src = toks
        gm = tmask[:, None] & nv_mask[None, :]
        a_val = tl.load(a_in + toks_src[:, None] * NV + offs_nv[None, :], mask=gm, other=0.0).to(tl.float32)
        b_val = tl.load(b_in + toks_src[:, None] * NV + offs_nv[None, :], mask=gm, other=0.0).to(tl.float32)
        x = a_val + dt_val[None, :]
        softplus_x = tl.where(BETA * x <= THRESHOLD, (1.0 / BETA) * tl.log(1.0 + tl.exp(BETA * x)), x)
        tl.store(
            g_out + toks[:, None] * NV + offs_nv[None, :],
            (neg_exp_alog[None, :] * softplus_x).to(g_out.dtype.element_ty),
            mask=gm,
        )
        tl.store(
            beta_out + toks[:, None] * NV + offs_nv[None, :],
            tl.sigmoid(b_val).to(beta_out.dtype.element_ty),
            mask=gm,
        )


def _pick_block_t(T: int, section_elems: int, num_core: int) -> int:
    # Two tiers only (T <= num_core: 1 row/program; otherwise section-sized
    # blocks). Each distinct BLOCK_T value compiles once (~5s, then disk
    # cached), so keeping the tier count minimal avoids mid-serving compile
    # stalls when T varies across requests; small-T kernels are launch-bound
    # and insensitive to BLOCK_T anyway.
    if num_core >= T:
        return 1
    return min(_MAX_BLOCK_T, max(1, triton.next_power_of_2(_TARGET_SECTION_ELEMS // section_elems)))


def _precompile_tiers(constexprs: dict, dtypes: tuple, dev, index_dtype) -> None:
    """Compile every BLOCK_T tier for this config on first use.

    vllm's startup covers both tiers for the usual configs (profile_run uses
    T=max_num_batched_tokens, cudagraph capture uses T<=max_num_seqs), but
    that is configuration-dependent (e.g. enforce_eager with large
    max_num_seqs leaves the small tier cold). Compiling all tiers on the
    first call makes the guarantee unconditional; the binaries are disk
    cached afterwards. Skipped under stream capture: the dummy launches
    would be recorded into the graph.
    """
    try:
        if torch.npu.is_current_stream_capturing():
            return
    except Exception:
        return

    num_core = get_vectorcore_num()
    section_cap = _pick_block_t(num_core + 1, constexprs["NK"] * constexprs["DK"], num_core)
    current_bt = constexprs["BLOCK_T"]

    nv = constexprs["NV"]
    qkv_dim, q_dim, v_dim = constexprs["QKV_DIM"], constexprs["Q_DIM"], constexprs["V_DIM"]
    dummy_qkv = torch.zeros(1, qkv_dim, dtype=dtypes[0], device=dev)
    dummy_a = torch.zeros(1, nv, dtype=dtypes[1], device=dev)
    dummy_b = torch.zeros(1, nv, dtype=dtypes[2], device=dev)
    dummy_index = torch.zeros(1, dtype=index_dtype, device=dev)
    dummy_alog = torch.zeros(nv, dtype=dtypes[3], device=dev)
    dummy_dt = torch.zeros(nv, dtype=dtypes[4], device=dev)
    dummy_q = torch.empty(1, q_dim, dtype=dtypes[0], device=dev)
    dummy_k = torch.empty(1, q_dim, dtype=dtypes[0], device=dev)
    dummy_v = torch.empty(1, v_dim, dtype=dtypes[0], device=dev)
    dummy_g = torch.empty(1, nv, dtype=torch.float32, device=dev)
    dummy_beta = torch.empty(1, nv, dtype=dtypes[2], device=dev)

    for bt in {1, section_cap}:
        if bt == current_bt:
            continue
        tier_constexprs = dict(constexprs, BLOCK_T=bt)
        _fused_gdn_prepare_kernel[(1,)](
            dummy_q,
            dummy_k,
            dummy_v,
            dummy_g,
            dummy_beta,
            dummy_qkv,
            dummy_a,
            dummy_b,
            dummy_index,
            dummy_alog,
            dummy_dt,
            1,
            1,
            **tier_constexprs,
        )


def _launch_fused_gdn_prepare(args, plan):
    """Launch the kernel, bypassing JITFunction dispatch on cache hits.

    The standard triton launch path costs >100us of CPU per call on this stack,
    which dominates the kernel time (~40us); a cached CompiledKernel.run call
    cuts it to ~30-40us. Falls back to the standard path whenever it is unsafe
    to reuse the cached binary (unaligned inputs, active profiler hooks).
    Outputs are fresh torch.empty tensors (always 16B-aligned by the caching
    allocator), so only the 6 input tensors need the alignment check.
    """
    ck = plan["ck"]
    if ck is not None:
        if (
            triton.compiler.CompiledKernel.launch_enter_hook is None
            and triton.compiler.CompiledKernel.launch_exit_hook is None
            and args[5].data_ptr() % 16 == 0
            and args[6].data_ptr() % 16 == 0
            and args[7].data_ptr() % 16 == 0
            and args[8].data_ptr() % 16 == 0
            and args[9].data_ptr() % 16 == 0
            and args[10].data_ptr() % 16 == 0
        ):
            dev = driver.active.get_current_device()
            stream = driver.active.get_current_stream(dev)
            grid = plan["grid"]
            ck.run(grid[0], 1, 1, stream, ck.function, ck.packed_metadata, None, None, None, *args)
            return

    ck = _fused_gdn_prepare_kernel[plan["grid"]](*args, **plan["constexprs"])
    plan["ck"] = ck


def fused_gdn_prepare_impl(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    key_dim: int,
    value_dim: int,
    tp_size: int,
    index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse split_qkv + l2norm + gating into a single triton kernel.

    Args:
        index: original token id per output row ([T] int tensor), required when
            ``mixed_qkv`` is a gathered token subset (spec path) but ``a``/``b``
            are still the full batch: the kernel gathers the a/b rows inline,
            replacing 4 host-side index_select dispatches. None when a/b are
            already aligned with the mixed_qkv rows.

    Returns (batched convention, matching rearrange_mixed_qkv/fused_gdn_gating):
        q_norm: [1, T, Nk, Dk] (l2-normalized)
        k_norm: [1, T, Nk, Dk] (l2-normalized)
        v:      [1, T, Nv, Dv] (raw copy)
        g:      [1, T, Nv] (float32, gating)
        beta:   [1, T, Nv] (sigmoid of b)
    """
    T = mixed_qkv.shape[0]
    QKV_DIM = mixed_qkv.shape[1]

    NK = num_k_heads // tp_size
    NV = num_v_heads // tp_size
    DK = head_k_dim
    DV = head_v_dim

    Q_DIM = NK * DK
    V_DIM = NV * DV

    Q_OFFSET = 0
    K_OFFSET = key_dim // tp_size
    V_OFFSET = key_dim * 2 // tp_size

    q_out = torch.empty(1, T, NK, DK, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    k_out = torch.empty(1, T, NK, DK, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    v_out = torch.empty(1, T, NV, DV, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    g = torch.empty(1, T, NV, dtype=torch.float32, device=mixed_qkv.device)
    beta_out = torch.empty(1, T, NV, dtype=b.dtype, device=mixed_qkv.device)

    if T == 0:
        return q_out, k_out, v_out, g, beta_out

    # per-(config, shape, dtype, device) launch plan cache: grid/constexprs are
    # recomputed only when the model config, T or tensor dtypes change
    pkey = (
        T,
        torch.npu.current_device(),
        NK,
        NV,
        DK,
        DV,
        QKV_DIM,
        Q_OFFSET,
        K_OFFSET,
        V_OFFSET,
        mixed_qkv.dtype,
        a.dtype,
        b.dtype,
        A_log.dtype,
        dt_bias.dtype,
        index is not None,
        index.dtype if index is not None else None,
    )
    plan = _PLAN_CACHE.get(pkey)
    if plan is None:
        init_device_properties_triton()
        num_core = get_vectorcore_num()
        block_t = _pick_block_t(T, NK * DK, num_core)
        grid = (min(num_core, triton.cdiv(T, block_t)),)
        num_chunks = triton.cdiv(triton.cdiv(T, grid[0]), block_t)
        constexprs = {
            "QKV_DIM": QKV_DIM,
            "Q_DIM": Q_DIM,
            "V_DIM": V_DIM,
            "NK": NK,
            "NV": NV,
            "DK": DK,
            "DV": DV,
            "Q_OFFSET": Q_OFFSET,
            "K_OFFSET": K_OFFSET,
            "V_OFFSET": V_OFFSET,
            "BLOCK_T": block_t,
            "BLOCK_H": triton.next_power_of_2(NK),
            "BLOCK_VD": triton.next_power_of_2(V_DIM),
            "BLOCK_NV": triton.next_power_of_2(NV),
            "HAS_INDEX": index is not None,
            "EPS": 1e-6,
            "BETA": 1.0,
            "THRESHOLD": 20.0,
            "NUM_STAGES": _NUM_STAGES,
        }
        plan = {"grid": grid, "num_chunks": num_chunks, "constexprs": constexprs, "ck": None}
        _PLAN_CACHE[pkey] = plan
        _precompile_tiers(
            constexprs,
            (mixed_qkv.dtype, a.dtype, b.dtype, A_log.dtype, dt_bias.dtype),
            mixed_qkv.device,
            # HAS_INDEX=False passes `a` as the unused INDEX dummy pointer.
            index.dtype if index is not None else a.dtype,
        )

    args = (
        q_out,
        k_out,
        v_out,
        g,
        beta_out,
        mixed_qkv,
        a,
        b,
        index if index is not None else a,  # unused dummy pointer when HAS_INDEX=False
        A_log,
        dt_bias,
        T,
        plan["num_chunks"],
    )
    _launch_fused_gdn_prepare(args, plan)

    return q_out, k_out, v_out, g, beta_out


def fused_gdn_prepare_fake(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    key_dim,
    value_dim,
    tp_size,
    index=None,
):
    T = mixed_qkv.shape[0]
    NK = num_k_heads // tp_size
    NV = num_v_heads // tp_size
    q = mixed_qkv.new_empty(1, T, NK, head_k_dim)
    k = mixed_qkv.new_empty(1, T, NK, head_k_dim)
    v = mixed_qkv.new_empty(1, T, NV, head_v_dim)
    g = mixed_qkv.new_empty(1, T, NV, dtype=torch.float32)
    beta = mixed_qkv.new_empty(1, T, NV, dtype=b.dtype)
    return q, k, v, g, beta
