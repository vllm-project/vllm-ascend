import math
import torch
import torch_npu
import triton
import triton.language as tl
form vllm_ascend.ops.triton.triton_utils import extract_slice,get_element,get_vectorcore_num,insert_slice

@triton.jit
def prepare_chunk_kernel(cu_seqlens,
                        chunk_size,
                        output_indices,
                        output_offsets,
                        genseq,
                        n_elements:tl.constexpr,
                        MAX_CHUNK:tl.constexpr,
                        BLOCK_SIZE:tl.constexpr,
                        ):
    offsets0 = tl.arange(0,BLOCK_SIZE)
    mask0 = offsets0 < n_elements
    seq = tl.load(cu_seqlens + offsets0,mask=mask0)
    seq_0 = extract_slice(seq, [0], [n_elements-1], [1])
    seq_next = extract_slice(seq, [1], [n_elements-1],[1])
    chunk_num = tl.cdiv(seq_next - seq_0, chunk_size)
    csum = tl.cumsum(chunk_num)
    offs = MAX_CHUNK
    mask0 = offsets0 < BLOCK_SIZE
    gseq = tl.load(genseq + offsets0, mask=mask0)
    for pos in tl.range(0, n_elements-1):
        len = get_element(chunk_num, (pos,))
        mask_gen = offsets0 < len
        tl.store(output_indices + offs + offsets0, gseq, mask=mask_gen)
        offs = offs + tl.cast(len, tl.int32)
    indoffsets = tl.arange(0, MAX_CHUNK)
    indmask = indoffsets < offs
    indeces = tl.load(output_indices + MAX_CHUNK + indoffsets, mask=indmask)
    cmp = indeces == 0
    outind = tl.cumsum(tl.cast(cmp, tl.int32)) - 1
    tl.store(output_indices + indoffsets, outind, mask = indmask)

    offsets0 = tl.arange(0, n_elements - 1)
    mask0 = offsets0 < n_elements - 1
    tl.store(output_offsets + 1 + offsets0, csum, mask0)

def prepare_chunk(seqlen, chunk_size):
    output = torch.zeros_like(seqlen)
    genseq = torch.arange(start=0, end=512).npu()
    n_elements = output.numel()
    max_chunk = 2048
    out_indices = torch.empty(2, max_chunk, device='npu')
    prepare_chunk_kernel[(1,)](seqlen, chunk_size, out_indices, output, genseq, n_elements, max_chunk, 512)
    out_indices = out_indices[:, 0:output[-1]]
    out_indices = out_indices.t()
    return out_indices, output