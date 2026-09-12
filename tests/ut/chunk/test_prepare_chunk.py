from dataclasses import dataclass
from typing import List, Tuple
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from vllm_ascend.ops.triton.triton_utils import extract_slice, get_element


@dataclass
class TestCase:
    cu_seqlens: List[int]
    chunk_size: int
    name: str = ""


def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices_ref(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def prepare_chunk_offsets_ref(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)

@triton.jit
def prepare_chunk_kernel(
    cu_seqlens, chunk_size, output_indices, output_offsets, genseq,
    n_elements: tl.constexpr, MAX_CHUNK: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    offsets0 = tl.arange(0, BLOCK_SIZE)
    mask0 = offsets0 < n_elements
    seq = tl.load(cu_seqlens + offsets0, mask=mask0)
    seq_0 = extract_slice(seq, [0], [n_elements-1], [1])
    seq_next = extract_slice(seq, [1], [n_elements-1], [1])
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
    tl.store(output_indices + indoffsets, outind, mask=indmask)

    offsets0 = tl.arange(0, n_elements - 1)
    mask0 = offsets0 < n_elements - 1
    tl.store(output_offsets + 1 + offsets0, csum, mask0)


def prepare_chunk_fused(cu_seqlens: torch.Tensor, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros_like(cu_seqlens)
    genseq = torch.arange(start=0, end=512, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    n_elements = output.numel()
    max_chunk = 2048
    out_indices = torch.empty(2, max_chunk, device=cu_seqlens.device)
    BLOCK_SIZE = 512
    grid = (1,)
    prepare_chunk_kernel[grid](
        cu_seqlens, chunk_size, out_indices, output, genseq,
        n_elements, max_chunk, BLOCK_SIZE
    )
    out_indices = out_indices[:, 0:output[-1]]
    out_indices = out_indices.t()
    return out_indices, output

def _assert_tensors_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, msg: str = "") -> None:
    assert tensor1.shape == tensor2.shape, f"{msg} shape mismatch: {tensor1.shape} != {tensor2.shape}"
    assert torch.allclose(tensor1.cpu().float(), tensor2.cpu().float()), f"{msg} value mismatch"

TEST_CASES = [
    TestCase(cu_seqlens=[0, 8], chunk_size=4, name="single_exact"),
    TestCase(cu_seqlens=[0, 10], chunk_size=4, name="single_not_exact"),
    TestCase(cu_seqlens=[0, 5, 13, 16], chunk_size=4, name="multi_mixed"),
    TestCase(cu_seqlens=[0, 3], chunk_size=4, name="shorter_than_chunk"),
    TestCase(cu_seqlens=[0, 0, 5, 5], chunk_size=4, name="with_empty"),
    TestCase(cu_seqlens=[0, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864], chunk_size=128, name="large"),
]

@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_prepare_chunk_basic(test_case: TestCase) -> None:
    device = torch.device("npu")
    cu_seqlens_tensor = torch.tensor(test_case.cu_seqlens, device=device)
    
    ref_indices = prepare_chunk_indices_ref(cu_seqlens_tensor, test_case.chunk_size)
    ref_offsets = prepare_chunk_offsets_ref(cu_seqlens_tensor, test_case.chunk_size)
    
    fused_indices, fused_offsets = prepare_chunk_fused(cu_seqlens_tensor, test_case.chunk_size)
    
    _assert_tensors_equal(fused_indices, ref_indices, "indices")
    _assert_tensors_equal(fused_offsets, ref_offsets, "offsets")

@pytest.mark.parametrize("chunk_size", [1, 4, 16, 64])
def test_prepare_chunk_varying_sizes(chunk_size: int) -> None:
    device = torch.device("npu")
    cu_seqlens = [0, 20, 35, 50, 100]
    cu_seqlens_tensor = torch.tensor(cu_seqlens, device=device)
    
    ref_indices = prepare_chunk_indices_ref(cu_seqlens_tensor, chunk_size)
    ref_offsets = prepare_chunk_offsets_ref(cu_seqlens_tensor, chunk_size)
    
    fused_indices, fused_offsets = prepare_chunk_fused(cu_seqlens_tensor, chunk_size)
    
    _assert_tensors_equal(fused_indices, ref_indices, "indices")
    _assert_tensors_equal(fused_offsets, ref_offsets, "offsets")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])