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
torch.npu.set_device(7)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))

def cdiv_torch(a, b):
    return (a + b - 1) // b

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), cdiv_torch(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1) 

def forward_o_trans_cpu(
    q,k,v,hidden_state,g,gk,
    scale,
    chunk_size,
    cu_seqlens,chunk_indices
):
    q = q.to(torch.float32).npu()
    k = k.to(torch.float32).npu()
    v = v.to(torch.float32).npu()
    hidden_state = hidden_state.to(torch.float32).npu()
    g = g.to(torch.float32).npu()

    B, HK, T, K = k.shape[0], k.shape[1], k.shape[2], k.shape[3]
    HV, V = v.shape[1], v.shape[3] 

    BT = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) // BT, None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    o_output = torch.zeros((B, HV, T, V), device=k.device, dtype=torch.float32)
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
                q_sel = torch.zeros((BT, q.shape[-1]), device=q.device, dtype=q.dtype)
                k_sel = torch.zeros((BT, k.shape[-1]), device=k.device, dtype=k.dtype)
                v_sel = torch.zeros((BT, v.shape[-1]), device=v.device, dtype=v.dtype)
                g_sel = torch.zeros((BT), device=g.device, dtype=g.dtype)
                actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

                if cu_seqlens is None: 
                    q_sel[:actual_len, :] = q[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    k_sel[:actual_len, :] = k[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    v_sel[:actual_len, :] = v[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[n, h, bos + i * BT : bos + i * BT + actual_len]
                    hidden_state_sel = hidden_state[n, h, boh+i]
                    attn = q_sel @ k_sel.transpose(-1, -2) 
                    L_mask = (g_sel.unsqueeze(-1) - g_sel.unsqueeze(-2)).exp() 
                    attn = attn*L_mask    
                    del L_mask
                    attn = torch.tril(attn, 0) 
                    o_inter = (q_sel * g_sel.to(torch.float16).exp()[:, None]) @ hidden_state_sel
                    o = (o_inter + attn @ v_sel) * scale
                    o_output[n, h, bos + i * BT: bos + i * BT + actual_len,:] = o[:actual_len, :]
                else:
                    q_sel[:actual_len, :] = q[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    k_sel[:actual_len, :] = k[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    v_sel[:actual_len, :] = v[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel[:actual_len] = g[0, h, bos + i * BT : bos + i * BT + actual_len]
                    hidden_state_sel = hidden_state[0, h, boh+i]
                    attn = q_sel @ k_sel.transpose(-1, -2) 
                    L_mask = (g_sel.unsqueeze(-1) - g_sel.unsqueeze(-2)).exp() 
                    attn = attn*L_mask           
                    del L_mask
                    attn = torch.tril(attn, 0) 
                    o_inter = (q_sel * g_sel.exp()[:, None]) @ hidden_state_sel
                    o = (o_inter + attn @ v_sel) * scale
                    o_output[0, h, bos + i * BT: bos + i * BT + actual_len,:] = o[:actual_len, :]
    o_output = o_output.to(torch.bfloat16)
    return o_output

class GDNFwdOInput:
    def __init__(self):
        self.shape_batch = int(sys.argv[1])
        self.seqlen = int(sys.argv[2])
        self.k_num_head = int(sys.argv[3])
        self.v_num_head = int(sys.argv[4])
        self.k_head_dim = int(sys.argv[5])
        self.v_head_dim = int(sys.argv[6])
        self.is_varied_len = int(sys.argv[7])
        self.token_batch = int(sys.argv[8])
        self.chunk_size = int(sys.argv[9])
        self.scale = float(sys.argv[10])
        self.str_dtype = str(sys.argv[11])
        self.use_actual_input = int(sys.argv[12])
        self.use_actual_output = int(sys.argv[13])
        self.data_path = str(sys.argv[14])

        if self.str_dtype == "half" or self.str_dtype == "fp16" or self.str_dtype == "float16":
            self.dtype = torch.float16
        elif self.str_dtype == "bf16" or self.str_dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            logging("[ERROR] dtype must be half or bf16")
            sys.exit()

class GDNFwdOInputTensor:
    def __init__(self, q, k, v, h, g, cu_seqlens, chunk_offsets):
        self.q = q
        self.k = k
        self.v = v
        self.h = h
        self.g = g
        self.cu_seqlens = cu_seqlens
        self.chunk_offsets = chunk_offsets

class GDNFwdOOutputTensor:
    def __init__(self, o):
        self.o = o

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

def get_cu_offsets(o_input, cu_seqlens):
    if cu_seqlens is None:
        return None, None
    cu_seqlens = cu_seqlens.to(torch.int64)
    num_chunks = 0
    chunk_offsets = []
    for tb in range(o_input.token_batch):
        curr_chunks = math.ceil((cu_seqlens[tb + 1] - cu_seqlens[tb]) / o_input.chunk_size)
        num_chunks += curr_chunks
        for c in range(curr_chunks):
            chunk_offsets.append([tb, c])
    return cu_seqlens.npu(), torch.Tensor(chunk_offsets).to(cu_seqlens.dtype).npu()

def gen_decay_data(o_input, cu_seqlens, chunk_offsets):
    base = torch.randint(-15, -5, [o_input.v_num_head])
    bias = torch.empty([o_input.shape_batch, o_input.v_num_head, o_input.seqlen]).uniform_(-2, 0)
    g = base[:, None] + bias

    for shape_batch_idx in range(o_input.shape_batch):
        for v_head_idx in range(o_input.v_num_head):
            for token_batch_idx in range(o_input.token_batch):
                batch_token_start, batch_token_end = cu_seqlens[token_batch_idx], cu_seqlens[token_batch_idx+1]
                batch_tokens = batch_token_end - batch_token_start
                batch_chunks = math.ceil(batch_tokens / o_input.chunk_size)
                for chunk_id in range(batch_chunks):
                    chunk_start_token = batch_token_start + o_input.chunk_size * chunk_id
                    chunk_end_token = min(chunk_start_token + o_input.chunk_size, batch_token_end)
                    g[shape_batch_idx, v_head_idx, chunk_start_token:chunk_end_token] = g[shape_batch_idx, v_head_idx, chunk_start_token:chunk_end_token].cumsum(0)
    return g

def gen_input_data(o_input):
    cu_seqlens = gen_seqlen(o_input.seqlen, o_input.is_varied_len, o_input.token_batch)
    cu_seqlens, chunk_offsets = get_cu_offsets(o_input, cu_seqlens)
    num_chunks = chunk_offsets.shape[0] if chunk_offsets is not None else (math.ceil(o_input.seqlen / o_input.chunk_size))
    q = torch.randn([o_input.shape_batch, o_input.k_num_head, o_input.seqlen, o_input.k_head_dim], dtype=o_input.dtype)
    k = torch.randn([o_input.shape_batch, o_input.k_num_head, o_input.seqlen, o_input.k_head_dim], dtype=o_input.dtype)
    v = torch.randn([o_input.shape_batch, o_input.v_num_head, o_input.seqlen, o_input.v_head_dim], dtype=o_input.dtype)
    h = torch.randn([o_input.shape_batch, o_input.v_num_head, num_chunks, o_input.k_head_dim, o_input.v_head_dim], dtype=o_input.dtype)
    g = torch.randn([o_input.shape_batch, o_input.v_num_head, o_input.seqlen], dtype=torch.float)
    # g = gen_decay_data(o_input, cu_seqlens, chunk_offsets)
    return GDNFwdOInputTensor(q, k, v, h, g, cu_seqlens, chunk_offsets)

def parse_actual_input(o_input):
    actual_data = torch.load(o_input.data_path, map_location='cpu')
    q = actual_data['q'].to(o_input.dtype).transpose(1, 2).contiguous()
    k = actual_data['k'].to(o_input.dtype).transpose(1, 2).contiguous()
    v = actual_data['v'].to(o_input.dtype).transpose(1, 2).contiguous()
    h = actual_data['h'].to(o_input.dtype).transpose(1, 2).contiguous()
    g = actual_data['g'].transpose(1, 2).contiguous()
    cu_seqlens, chunk_offsets = get_cu_offsets(o_input, actual_data.get('cu_seqlens'))
    return GDNFwdOInputTensor(q, k, v, h, g, cu_seqlens, chunk_offsets)

def gen_ref_data(o_input, input_tensor):
    o = forward_o_trans_cpu(
        input_tensor.q, input_tensor.k, input_tensor.v,
        input_tensor.h, input_tensor.g, None,
        o_input.scale, o_input.chunk_size,
        input_tensor.cu_seqlens,
        None
    )
    return GDNFwdOOutputTensor(o)

def parse_actual_output(o_input):
    actual_data = torch.load(o_input.data_path, map_location='cpu')
    o = actual_data['o'].to(o_input.dtype).transpose(1, 2).contiguous()
    return GDNFwdOOutputTensor(o)

def save_data(input_tensor, output_tensor):
    os.makedirs(os.path.join(WORKSPACE, "data"), exist_ok=True)
    input_tensor.q.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "q.bin"))
    input_tensor.k.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "k.bin"))
    input_tensor.v.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "v.bin"))
    input_tensor.h.view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "h.bin"))
    input_tensor.g.numpy().tofile(os.path.join(WORKSPACE, "data", "g.bin"))
    if input_tensor.cu_seqlens is not None:
        np.array(input_tensor.cu_seqlens.cpu()).astype(np.int64).tofile(os.path.join(WORKSPACE, "data", "cu_seqlens.bin"))

    output_tensor.o.cpu().to(input_tensor.q.dtype).view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "o_ref.bin"))

if __name__ == "__main__":
    gdn_fwd_o_input = GDNFwdOInput()

    if gdn_fwd_o_input.use_actual_input:
        input_tensor = parse_actual_input(gdn_fwd_o_input)
    else:
        input_tensor = gen_input_data(gdn_fwd_o_input)
    
    if gdn_fwd_o_input.use_actual_output:
        output_tensor = parse_actual_output(gdn_fwd_o_input)
    else:
        output_tensor = gen_ref_data(gdn_fwd_o_input, input_tensor)

    torch.npu.synchronize()
    result = torch_npu.npu_chunk_fwd_o(
        input_tensor.q.npu(),
        input_tensor.k.npu(),
        input_tensor.v.npu(),
        input_tensor.h.npu(),
        input_tensor.g.npu(),
        input_tensor.cu_seqlens,
        input_tensor.chunk_offsets,
        gdn_fwd_o_input.scale,
        gdn_fwd_o_input.chunk_size
    )
    torch.npu.synchronize()
    save_data(input_tensor, output_tensor)
    result.cpu().view(torch.int16).numpy().tofile(os.path.join(WORKSPACE, "data", "o_npu.bin"))