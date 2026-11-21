import gc
import math
import random

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)

seed = 1
random.seed(seed)
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


def golden_chunk_fwd_o(q, k, v, hidden_state, g, gk, scale, chunk_size, cu_seqlens, chunk_indices):
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    hidden_state = hidden_state.to(torch.float32)
    g = g.to(torch.float32)

    B, HK, T, _ = k.shape[0], k.shape[1], k.shape[2], k.shape[3]
    HV, V = v.shape[1], v.shape[3]

    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) // BT, None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    o_output = torch.zeros((B, HV, T, V), dtype=torch.float32)
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
                q_sel = torch.zeros((BT, q.shape[-1]), dtype=q.dtype)
                k_sel = torch.zeros((BT, k.shape[-1]), dtype=k.dtype)
                v_sel = torch.zeros((BT, v.shape[-1]), dtype=v.dtype)
                g_sel = torch.zeros((BT), dtype=g.dtype)
                actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

                if cu_seqlens is None:
                    q_sel[:actual_len, :] = q[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    k_sel[:actual_len, :] = k[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    v_sel[:actual_len, :] = v[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[n, h, bos + i * BT : bos + i * BT + actual_len]
                    hidden_state_sel = hidden_state[n, h, boh + i]
                    attn = q_sel @ k_sel.transpose(-1, -2)
                    L_mask = (g_sel.unsqueeze(-1) - g_sel.unsqueeze(-2)).exp()
                    attn = attn * L_mask
                    del L_mask
                    attn = torch.tril(attn, 0)
                    o_inter = (q_sel * g_sel.exp()[:, None]) @ hidden_state_sel
                    o = (o_inter + attn @ v_sel) * scale
                    o_output[n, h, bos + i * BT : bos + i * BT + actual_len, :] = o[:actual_len, :]
                else:
                    q_sel[:actual_len, :] = q[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    k_sel[:actual_len, :] = k[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    v_sel[:actual_len, :] = v[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[0, h, bos + i * BT : bos + i * BT + actual_len]
                    hidden_state_sel = hidden_state[0, h, boh + i]
                    attn = q_sel @ k_sel.transpose(-1, -2)
                    L_mask = (g_sel.unsqueeze(-1) - g_sel.unsqueeze(-2)).exp()
                    attn = attn * L_mask
                    del L_mask
                    attn = torch.tril(attn, 0)
                    o_inter = (q_sel * g_sel.exp()[:, None]) @ hidden_state_sel
                    o = (o_inter + attn @ v_sel) * scale
                    o_output[0, h, bos + i * BT : bos + i * BT + actual_len, :] = o[:actual_len, :]
    o_output = o_output.to(torch.bfloat16)
    return o_output


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


def get_cu_offsets(chunk_size, token_batch, cu_seqlens):
    if cu_seqlens is None:
        return None, None
    cu_seqlens = cu_seqlens.to(torch.int64)
    num_chunks = 0
    chunk_offsets = []
    for tb in range(token_batch):
        curr_chunks = math.ceil((cu_seqlens[tb + 1] - cu_seqlens[tb]) / chunk_size)
        num_chunks += curr_chunks
        for c in range(curr_chunks):
            chunk_offsets.append([tb, c])
    return cu_seqlens, torch.Tensor(chunk_offsets).to(cu_seqlens.dtype)


def gen_input_data(
    shape_batch,
    seqlen,
    k_num_head,
    v_num_head,
    k_head_dim,
    v_head_dim,
    is_varied_len,
    token_batch,
    chunk_size,
    dtype,
    g_dtype,
):
    cu_seqlens = gen_seqlen(seqlen, is_varied_len, token_batch)
    cu_seqlens, chunk_offsets = get_cu_offsets(chunk_size, token_batch, cu_seqlens)
    num_chunks = chunk_offsets.shape[0] if chunk_offsets is not None else math.ceil(seqlen / chunk_size)

    q = torch.randn([shape_batch, k_num_head, seqlen, k_head_dim], dtype=dtype)
    k = torch.randn([shape_batch, k_num_head, seqlen, k_head_dim], dtype=dtype)
    v = torch.randn([shape_batch, v_num_head, seqlen, v_head_dim], dtype=dtype)
    h = torch.randn([shape_batch, v_num_head, num_chunks, k_head_dim, v_head_dim], dtype=dtype)
    g = torch.randn([shape_batch, v_num_head, seqlen], dtype=g_dtype)
    return q, k, v, h, g, cu_seqlens, chunk_offsets


@pytest.mark.parametrize("shape_batch", [1])
@pytest.mark.parametrize("seqlen", [128, 256])
@pytest.mark.parametrize("k_num_head,v_num_head", [(2, 2)])
@pytest.mark.parametrize("k_head_dim,v_head_dim", [(128, 128)])
@pytest.mark.parametrize("is_varied_len", [0, 1])
@pytest.mark.parametrize("token_batch", [1])
@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("scale", [0.08838834764831845])
@pytest.mark.parametrize("dtype,g_dtype", [(torch.bfloat16, torch.float32)])
def test_chunk_fwd_o(
    shape_batch,
    seqlen,
    k_num_head,
    v_num_head,
    k_head_dim,
    v_head_dim,
    is_varied_len,
    token_batch,
    chunk_size,
    scale,
    dtype,
    g_dtype,
):
    torch.manual_seed(seed)

    if is_varied_len:
        shape_batch = 1
        token_batch = shape_batch
    else:
        token_batch = 1

    q, k, v, h, g, cu_seqlens, chunk_offsets = gen_input_data(
        shape_batch=shape_batch,
        seqlen=seqlen,
        k_num_head=k_num_head,
        v_num_head=v_num_head,
        k_head_dim=k_head_dim,
        v_head_dim=v_head_dim,
        is_varied_len=is_varied_len,
        token_batch=token_batch,
        chunk_size=chunk_size,
        dtype=dtype,
        g_dtype=g_dtype,
    )

    ref_o = golden_chunk_fwd_o(
        q,
        k,
        v,
        h,
        g,
        None,
        scale,
        chunk_size,
        cu_seqlens,
        None,
    )

    torch.npu.synchronize()
    result = torch.ops._C_ascend.chunk_fwd_o(
        q.npu(),
        k.npu(),
        v.npu(),
        h.npu(),
        scale,
        g=g.npu(),
        g_gamma=None,
        cu_seqlens=cu_seqlens.tolist() if cu_seqlens is not None else None,
        chunk_indices=chunk_offsets.flatten().tolist() if chunk_offsets is not None else None,
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )
    torch.npu.synchronize()

    npu_o = result.cpu().to(torch.float32)
    ref_o = ref_o.cpu().to(torch.float32)

    torch.testing.assert_close(npu_o, ref_o, rtol=1e-2, atol=1e-2, equal_nan=True)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
