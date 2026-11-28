import gc
import math
import random

import numpy as np
import pytest
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def cdiv_torch(a, b):
    return (a + b - 1) // b


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), cdiv_torch(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def forward_h_trans_cpu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor = None,
):
    dtype_ = k.dtype
    state_type_ = initial_state.dtype if initial_state is not None else torch.float32

    k = k.to(torch.float32)
    w = w.to(torch.float32)
    u = u.to(torch.float32)
    g = g.to(torch.float32)

    B, HK, T, K = k.shape[0], k.shape[1], k.shape[2], k.shape[3]
    HV, V = u.shape[1], u.shape[3]

    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) // BT, None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    if initial_state is not None:
        initial_state = initial_state.reshape([N, HV, K, V]).contiguous().to(torch.float32)

    S = torch.zeros((B, HV, NT, K, V), device=k.device, dtype=torch.float32)
    v_new_output = torch.zeros((B, HV, T, V), device=k.device, dtype=torch.float32)
    final_state = torch.zeros((N, HV, K, V), device=k.device, dtype=torch.float32)

    head_ratio = HV // HK
    for n in range(N):
        if cu_seqlens is None:
            bos = 0
            eos = T
            T_inner = T
            NT_inner = NT
            boh = 0
        else:
            bos = cu_seqlens[n]
            eos = cu_seqlens[n + 1]
            T_inner = eos - bos
            NT_inner = (T_inner + BT - 1) // BT
            assert chunk_offsets is not None
            boh = chunk_offsets[n]

        for h in range(HV):
            for i in range(NT_inner):
                k_sel = torch.zeros((BT, k.shape[-1]), device=k.device, dtype=k.dtype)
                w_sel = torch.zeros((BT, w.shape[-1]), device=w.device, dtype=w.dtype)
                u_sel = torch.zeros((BT, u.shape[-1]), device=u.device, dtype=u.dtype)
                g_sel = torch.zeros((BT), device=g.device, dtype=g.dtype)
                actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

                if cu_seqlens is None:
                    k_sel[:actual_len, :] = k[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel[:actual_len, :] = w[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel[:actual_len, :] = u[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[n, h, bos + i * BT : bos + i * BT + actual_len]
                    if initial_state is not None and i == 0:
                        S[n, h, boh + i] = initial_state[n, h]
                    v_new = u_sel - w_sel @ S[n, h, boh + i]
                    if i != NT_inner - 1:
                        S[n, h, boh + i + 1] = S[n, h, boh + i] * g_sel[
                            actual_len - 1, None, None
                        ].exp() + k_sel.transpose(-1, -2) @ (
                            v_new * (g_sel[actual_len - 1, None] - g_sel).exp()[..., None]
                        )
                    else:
                        final_state[n, h] = S[n, h, boh + i] * g_sel[
                            actual_len - 1, None, None
                        ].exp() + k_sel.transpose(-1, -2) @ (
                            v_new * (g_sel[actual_len - 1, None] - g_sel).exp()[..., None]
                        )
                    v_new_output[n, h, bos + i * BT : bos + i * BT + actual_len, :] = v_new[:actual_len, :]
                else:
                    k_sel[:actual_len, :] = k[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel[:actual_len, :] = w[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel[:actual_len, :] = u[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[0, h, bos + i * BT : bos + i * BT + actual_len]
                    if initial_state is not None and i == 0:
                        S[0, h, boh + i] = initial_state[n, h]
                    v_new = u_sel - w_sel @ S[0, h, boh + i]
                    if i != NT_inner - 1:
                        S[0, h, boh + i + 1] = S[0, h, boh + i] * g_sel[
                            actual_len - 1, None, None
                        ].exp() + k_sel.transpose(-1, -2) @ (
                            v_new * (g_sel[actual_len - 1, None] - g_sel).exp()[..., None]
                        )
                    else:
                        final_state[n, h] = S[0, h, boh + i] * g_sel[
                            actual_len - 1, None, None
                        ].exp() + k_sel.transpose(-1, -2) @ (
                            v_new * (g_sel[actual_len - 1, None] - g_sel).exp()[..., None]
                        )
                    v_new_output[0, h, bos + i * BT : bos + i * BT + actual_len, :] = v_new[:actual_len, :]

    S = S.to(dtype_)
    v_new_output = v_new_output.to(dtype_)
    final_state = final_state.to(state_type_) if final_state is not None else None
    return S, v_new_output, final_state


def gen_seqlen(seqlen, is_varied_len, batch):
    if is_varied_len == 0:
        return None
    cu_seqlens = [0]
    avg_len = seqlen // batch
    for i in range(batch - 1):
        diff = random.randint(avg_len // 2, avg_len * 3 // 2)
        cu_seqlens.append(cu_seqlens[-1] + diff)
    cu_seqlens.append(seqlen)
    return torch.Tensor(cu_seqlens).to(torch.int64)


def get_cu_offsets(chunk_size, cu_seqlens):
    if cu_seqlens is None:
        return None, None
    cu_seqlens = cu_seqlens.to(torch.int64)
    num_chunks = 0
    curr_token = 0
    for seq in cu_seqlens:
        num_chunks += math.ceil((seq - curr_token) / chunk_size)
        curr_token = seq
    return cu_seqlens, torch.zeros([num_chunks, 2]).to(cu_seqlens.dtype)


def gen_decay_data(shape_batch, v_num_head, seqlen, chunk_size, cu_seqlens):
    base = torch.randint(-15, -5, [v_num_head])
    bias = torch.empty([shape_batch, v_num_head, seqlen]).uniform_(-2, 0)
    g = base[:, None] + bias

    token_batch = len(cu_seqlens) - 1 if cu_seqlens is not None else 1
    for shape_batch_idx in range(shape_batch):
        for v_head_idx in range(v_num_head):
            for token_batch_idx in range(token_batch):
                batch_token_start = cu_seqlens[token_batch_idx] if cu_seqlens is not None else 0
                batch_token_end = cu_seqlens[token_batch_idx + 1] if cu_seqlens is not None else seqlen
                batch_tokens = batch_token_end - batch_token_start
                batch_chunks = math.ceil(batch_tokens / chunk_size)
                for chunk_id in range(batch_chunks):
                    chunk_start_token = batch_token_start + chunk_size * chunk_id
                    chunk_end_token = min(chunk_start_token + chunk_size, batch_token_end)
                    g[shape_batch_idx, v_head_idx, chunk_start_token:chunk_end_token] = g[
                        shape_batch_idx, v_head_idx, chunk_start_token:chunk_end_token
                    ].cumsum(0)
    return g


def gen_input_data(
    batch,
    seqlen,
    k_num_head,
    v_num_head,
    k_head_dim,
    v_head_dim,
    is_varied_len,
    chunk_size,
    dtype,
    use_initial_state,
    state_dtype,
):
    if is_varied_len:
        shape_batch = 1
        token_batch = batch
    else:
        shape_batch = batch
        token_batch = 1

    cu_seqlens = gen_seqlen(seqlen, is_varied_len, token_batch)
    cu_seqlens, chunk_offsets = get_cu_offsets(chunk_size, cu_seqlens)

    w = torch.randn([shape_batch, v_num_head, seqlen, k_head_dim], dtype=dtype)
    u = torch.randn([shape_batch, v_num_head, seqlen, v_head_dim], dtype=dtype)
    k = torch.randn([shape_batch, k_num_head, seqlen, k_head_dim], dtype=dtype)
    g = gen_decay_data(shape_batch, v_num_head, seqlen, chunk_size, cu_seqlens)

    if use_initial_state:
        initial_state = torch.randn([shape_batch, v_num_head, token_batch, k_head_dim, v_head_dim], dtype=state_dtype)
    else:
        initial_state = None

    return k, w, u, g, cu_seqlens, chunk_offsets, initial_state


def _as_int_list(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.cpu().tolist()
    return list(x)


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize("k_num_head,v_num_head", [(2, 2)])
@pytest.mark.parametrize("k_head_dim,v_head_dim", [(128, 128)])
@pytest.mark.parametrize("is_varied_len", [0, 1])
@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("store_final_state", [False, True])
@pytest.mark.parametrize("g_dtype,state_dtype", [(torch.float32, torch.float32)])
def test_chunk_gated_delta_rule_fwd_h(
    batch,
    seqlen,
    k_num_head,
    v_num_head,
    k_head_dim,
    v_head_dim,
    is_varied_len,
    chunk_size,
    dtype,
    use_initial_state,
    store_final_state,
    g_dtype,
    state_dtype,
):
    torch.manual_seed(seed)

    k, w, u, g, cu_seqlens, chunk_offsets, initial_state = gen_input_data(
        batch=batch,
        seqlen=seqlen,
        k_num_head=k_num_head,
        v_num_head=v_num_head,
        k_head_dim=k_head_dim,
        v_head_dim=v_head_dim,
        is_varied_len=is_varied_len,
        chunk_size=chunk_size,
        dtype=dtype,
        use_initial_state=use_initial_state,
        state_dtype=state_dtype,
    )

    g = g.to(g_dtype)

    ref_h, ref_v_new, ref_final_state = forward_h_trans_cpu(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=store_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )

    torch.npu.synchronize()
    result = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k.npu(),
        w.npu(),
        u.npu(),
        g=g.npu(),
        initial_state=(initial_state.npu() if initial_state is not None else None),
        output_final_state=store_final_state,
        chunk_size=chunk_size,
        cu_seqlens=_as_int_list(cu_seqlens),
        chunk_indices=_as_int_list(chunk_offsets),
    )
    torch.npu.synchronize()

    npu_h = result[0].cpu().to(torch.float32)
    npu_v_new = result[1].cpu().to(torch.float32)
    npu_final_state = result[2].cpu().to(torch.float32) if result[2] is not None else None

    ref_h = ref_h.to(torch.float32)
    ref_v_new = ref_v_new.to(torch.float32)
    ref_final_state = ref_final_state.to(torch.float32) if ref_final_state is not None else None

    torch.testing.assert_close(npu_h, ref_h, rtol=1e-2, atol=1e-2, equal_nan=True)
    torch.testing.assert_close(npu_v_new, ref_v_new, rtol=1e-2, atol=1e-2, equal_nan=True)
    if store_final_state:
        torch.testing.assert_close(npu_final_state, ref_final_state, rtol=1e-2, atol=1e-2, equal_nan=True)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
