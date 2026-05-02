import os
import sys
import logging
import numpy as np
import random
import ml_dtypes
import torch
import torch_npu
from ml_dtypes import bfloat16
from dataclasses import dataclass
import math
np.random.seed(1)
torch.manual_seed(1)
torch.npu.set_device(2)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
from typing import Optional
def forward_h_trans_cpu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
):
    # 典型场景 HQ=HK=16 HV=32
    # assert HV >= HK 并且可以整除
    # assert 变长场景下，B==1
    # assert K==V==D
    dtype_ = k.dtype

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
    final_state = None

    S = torch.zeros((B, HV, NT, K, V), device=k.device, dtype=torch.float32)
    v_new_output = torch.zeros((B, HV, T, V), device=k.device, dtype=torch.float32)

    head_ratio = HV // HK
    for n in range(N):
        if cu_seqlens is None: # 定长
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
            boh = chunk_offsets[n]

        for h in range(HV):
            # B H T D ->
            for i in range(NT_inner):
                k_sel = torch.zeros((BT, k.shape[-1]), device=k.device, dtype=k.dtype)
                w_sel = torch.zeros((BT, w.shape[-1]), device=w.device, dtype=w.dtype)
                u_sel = torch.zeros((BT, u.shape[-1]), device=u.device, dtype=u.dtype)
                g_sel = torch.zeros((BT), device=g.device, dtype=g.dtype)
                actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

                if cu_seqlens is None: # 定长
                    k_sel[:actual_len, :] = k[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel[:actual_len, :] = w[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel[:actual_len, :] = u[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[n, h, bos + i * BT : bos + i * BT + actual_len]
                    v_new = u_sel - w_sel @ S[n, h, boh+i]
                    if i != NT_inner-1:
                        # S[n, h, boh+i+1] = S[n, h, boh+i] * g_sel[actual_len-1, None, None].exp() + (k_sel * (g_sel[actual_len-1, None] - g_sel).exp()[..., None]).transpose(-1, -2) @ v_new
                        S[n, h, boh+i+1] = S[n, h, boh+i] * g_sel[actual_len-1, None, None].exp() + k_sel.transpose(-1, -2) @ (v_new * (g_sel[actual_len-1, None] - g_sel).exp()[..., None])
                    v_new_output[n, h, bos + i * BT: bos + i * BT + actual_len, :] = v_new[:actual_len, :]

                else:
                    k_sel[:actual_len, :] = k[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel[:actual_len, :] = w[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel[:actual_len, :] = u[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[0, h, bos + i * BT : bos + i * BT + actual_len]
                    v_new = u_sel - w_sel @ S[0, h, boh+i]
                    if i != NT_inner-1:
                        # S[0, h, boh+i+1] = S[0, h, boh+i] * g_sel[actual_len-1, None, None].exp() + (k_sel * (g_sel[actual_len-1, None] - g_sel).exp()[..., None]).transpose(-1, -2) @ v_new
                        S[0, h, boh+i+1] = S[0, h, boh+i] * g_sel[actual_len-1, None, None].exp() + k_sel.transpose(-1, -2) @ (v_new * (g_sel[actual_len-1, None] - g_sel).exp()[..., None])
                    v_new_output[0, h ,bos + i * BT: bos + i * BT + actual_len, :] = v_new[:actual_len, :]


    #S = S.to(torch.bfloat16)
    #v_new_output = v_new_output.to(torch.bfloat16)
    S = S.to(dtype_)
    v_new_output = v_new_output.to(dtype_)
    return S, v_new_output, None

class GDNFwdHInput:
    def __init__(self):
        self.batch = int(sys.argv[1])
        self.seqlen = int(sys.argv[2])
        self.k_num_head = int(sys.argv[3])
        self.v_num_head = int(sys.argv[4])
        self.k_head_dim = int(sys.argv[5])
        self.v_head_dim = int(sys.argv[6])
        self.is_varied_len = int(sys.argv[7])
        self.chunk_size = int(sys.argv[8])
        self.use_initial_state = int(sys.argv[9])
        self.store_final_state = int(sys.argv[10])
        self.str_dtype = str(sys.argv[11])
        self.use_actual_input = int(sys.argv[12])
        self.use_actual_output = int(sys.argv[13])
        self.data_path = str(sys.argv[14])

        if self.is_varied_len:
            self.shape_batch = 1
            self.token_batch = self.batch
        else:
            self.shape_batch = self.batch
            self.token_batch = 1
        if self.str_dtype == "half" or self.str_dtype == "fp16" or self.str_dtype == "float16":
            self.dtype = torch.float16
        elif self.str_dtype == "bf16" or self.str_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            logging("[ERROR] dtype must be half or bf16")
            sys.exit()

class GDNFwdHInputTensor:
    def __init__(self, k, w, u, g, cu_seqlens, chunk_offsets, initial_state):
        self.k = k
        self.w = w
        self.u = u
        self.g = g
        self.cu_seqlens = cu_seqlens
        self.chunk_offsets = chunk_offsets
        self.initial_state = initial_state

class GDNFwdHOutputTensor:
    def __init__(self, h, v_new, final_state = None):
        self.h = h
        self.v_new = v_new
        self.final_state = final_state

def parse_actual_input(h_input):
    actual_data = torch.load(h_input.data_path, map_location='cpu')
    k = actual_data['k'][:, :, :h_input.k_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    w = actual_data['w'][:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    u = actual_data['u'][:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    g = actual_data['g'][:, :, :h_input.v_num_head].transpose(1, 2).contiguous()
    cu_seqlens, chunk_offsets = get_cu_offsets(h_input, actual_data.get('cu_seqlens'))
    initial_state = None
    h_input.num_tokens = cu_seqlens[-1] if h_input.is_varied_len else h_input.seqlen
    return GDNFwdHInputTensor(k, w, u, g, cu_seqlens, chunk_offsets, initial_state)

def parse_actual_output(h_input):
    actual_data = torch.load(h_input.data_path, map_location='cpu')
    h = actual_data['h'] if 'h' in actual_data.keys() else actual_data['ref_h']
    v = actual_data['v_new'] if 'v_new' in actual_data.keys() else actual_data['ref_v_new']
    h = h[:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    v = v[:, :, :h_input.v_num_head].to(h_input.dtype).transpose(1, 2).contiguous()
    return GDNFwdHOutputTensor(h, v)

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

def gen_wu_data():
    pass

def get_cu_offsets(h_input, cu_seqlens):
    if cu_seqlens is None:
        return None, None
    cu_seqlens = cu_seqlens.to(torch.int64)
    num_chunks = 0
    curr_token = 0
    for seq in cu_seqlens:
        num_chunks += math.ceil((seq - curr_token) / h_input.chunk_size)
        curr_token = seq
    return cu_seqlens.npu(), torch.zeros([num_chunks, 2]).to(cu_seqlens.dtype).npu()

def gen_decay_data(h_input, cu_seqlens, chunk_offsets):
    base = torch.randint(-15, -5, [h_input.v_num_head])
    bias = torch.empty([h_input.shape_batch, h_input.v_num_head, h_input.seqlen]).uniform_(-2, 0)
    g = base[:, None] + bias

    for shape_batch_idx in range(h_input.shape_batch):
        for v_head_idx in range(h_input.v_num_head):
            for token_batch_idx in range(h_input.token_batch):
                batch_token_start, batch_token_end = cu_seqlens[token_batch_idx], cu_seqlens[token_batch_idx+1]
                batch_tokens = batch_token_end - batch_token_start
                batch_chunks = math.ceil(batch_tokens / h_input.chunk_size)
                for chunk_id in range(batch_chunks):
                    chunk_start_token = batch_token_start + h_input.chunk_size * chunk_id
                    chunk_end_token = min(chunk_start_token + h_input.chunk_size, batch_token_end)
                    g[shape_batch_idx, v_head_idx, chunk_start_token:chunk_end_token] = g[shape_batch_idx, v_head_idx, chunk_start_token:chunk_end_token].cumsum(0)
    return g

def gen_input_data(h_input, rand_wu = True):
    cu_seqlens = gen_seqlen(h_input.seqlen, h_input.is_varied_len, h_input.token_batch)
    cu_seqlens, chunk_offsets = get_cu_offsets(h_input, cu_seqlens)
    if rand_wu:
        w = torch.randn([h_input.shape_batch, h_input.k_num_head, h_input.seqlen, h_input.k_head_dim], dtype=h_input.dtype)
        u = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.seqlen, h_input.v_head_dim], dtype=h_input.dtype)
    else:
        w, u = gen_wu_data()
    k = torch.randn([h_input.shape_batch, h_input.k_num_head, h_input.seqlen, h_input.k_head_dim], dtype=h_input.dtype)
    g = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.seqlen], dtype=torch.float)
    # g = gen_decay_data(h_input, cu_seqlens, chunk_offsets)
    if h_input.use_initial_state:
        initial_state = torch.randn([h_input.shape_batch, h_input.v_num_head, h_input.token_batch, h_input.k_head_dim, h_input.v_head_dim], dtype=h_input.dtype)
    else:
        initial_state = None
    return GDNFwdHInputTensor(k, w, u, g, cu_seqlens, chunk_offsets, initial_state)

def gen_ref_data(h_input, input_tensor):
    h, v, _ = forward_h_trans_cpu(k=input_tensor.k, w=input_tensor.w, u=input_tensor.u, g=input_tensor.g)
    return GDNFwdHOutputTensor(h, v)

def save_data(input_tensor, output_tensor):
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)
    input_tensor.k.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "k.bin"))
    input_tensor.w.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "w.bin"))
    input_tensor.u.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "u.bin"))
    input_tensor.g.numpy().tofile(os.path.join(WORKSPACE, "data", "g.bin"))
    if input_tensor.cu_seqlens is not None:
        np.array(input_tensor.cu_seqlens.cpu()).astype(np.int64).tofile(os.path.join(WORKSPACE, "data", "cu_seqlens.bin"))

    if input_tensor.initial_state is not None:
        input_tensor.initial_state.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "initial_state.bin"))

    output_tensor.h.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "h_ref.bin"))
    output_tensor.v_new.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "v_ref.bin"))
    if output_tensor.final_state is not None:
        output_tensor.final_state.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "final_state_ref.bin"))

if __name__ == "__main__":

    gdn_fwd_h_input = GDNFwdHInput()

    if gdn_fwd_h_input.use_actual_input:
        input_tensor = parse_actual_input(gdn_fwd_h_input)
    else:
        input_tensor = gen_input_data(gdn_fwd_h_input)

    if gdn_fwd_h_input.use_actual_output:
        output_tensor = parse_actual_output(gdn_fwd_h_input)
    else:
        output_tensor = gen_ref_data(gdn_fwd_h_input, input_tensor)
    torch.npu.synchronize()
    result = torch_npu.npu_chunk_gated_delta_rule_fwd_h(
        input_tensor.k.npu(),
        input_tensor.w.npu(),
        input_tensor.u.npu(),
        input_tensor.g.npu(),
        None,
        input_tensor.cu_seqlens,
        input_tensor.chunk_offsets,
        False,
        gdn_fwd_h_input.chunk_size
    )
    torch.npu.synchronize()
    save_data(input_tensor, output_tensor)
    result[0].cpu().view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "h_npu.bin"))
    result[1].cpu().view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "v_npu.bin"))
