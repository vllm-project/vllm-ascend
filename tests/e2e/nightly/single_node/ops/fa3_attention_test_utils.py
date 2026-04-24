import gc
import math
from dataclasses import dataclass

import torch
import torch_npu

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


@dataclass
class FA3DecodeSyntheticCase:
    q_nope_quant: torch.Tensor
    q_nope_deq: torch.Tensor
    q_pe_input: torch.Tensor
    q_pe_deq: torch.Tensor
    kv_base_quant_per_req: list[torch.Tensor]
    k_nope_deq_per_req: list[torch.Tensor]
    v_nope_deq_per_req: list[torch.Tensor]
    k_pe_cache_per_req: list[torch.Tensor]
    k_pe_deq_per_req: list[torch.Tensor]
    query_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    seq_lens: list[int]
    block_size: int
    num_heads: int
    num_kv_heads: int
    latent_dim: int
    rope_dim: int
    softmax_scale: float
    device: torch.device


@dataclass
class FA3DecodeBatch:
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    k_nope_cache: torch.Tensor
    k_pe_cache: torch.Tensor
    block_table: torch.Tensor
    query_scale: torch.Tensor
    actual_seq_qlen: list[int]
    actual_seq_kvlen: list[int]
    golden_output: torch.Tensor


def require_a5():
    if get_ascend_device_type() != AscendDeviceType.A5:
        import pytest

        pytest.skip("FA3 A5 decode tests only support Ascend A5.")


def cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _quantize_to_float8(tensor: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scaled = (tensor / scale).clamp(-240.0, 240.0)
    quant = scaled.to(torch.float8_e4m3fn)
    dequant = quant.float() * scale
    return quant.contiguous(), dequant.contiguous()


def _take_first_dim(tensor: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Avoid advanced indexing on NPU dtypes such as float8."""
    if not indices:
        raise ValueError("indices must not be empty")
    if len(indices) == 1:
        return tensor.narrow(0, indices[0], 1).contiguous()
    return torch.cat([tensor.narrow(0, idx, 1) for idx in indices], dim=0).contiguous()


def make_fa3_decode_case(
    seq_lens: list[int],
    num_heads: int,
    latent_dim: int,
    rope_dim: int,
    block_size: int,
    device: torch.device,
    seed: int = 20260421,
) -> FA3DecodeSyntheticCase:
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if latent_dim <= 0 or rope_dim <= 0:
        raise ValueError(f"latent_dim and rope_dim must be positive, got {latent_dim=} {rope_dim=}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    num_reqs = len(seq_lens)
    num_kv_heads = 1
    softmax_scale = 1.0 / math.sqrt(latent_dim + rope_dim)

    query_scale = torch.rand((num_reqs, num_heads, 1), generator=gen, device=device, dtype=torch.float32) * 0.25 + 0.2
    k_scale = torch.rand((num_kv_heads,), generator=gen, device=device, dtype=torch.float32) * 0.2 + 0.3
    v_scale = torch.rand((num_kv_heads,), generator=gen, device=device, dtype=torch.float32) * 0.25 + 0.35

    q_nope_src = torch.randn((num_reqs, num_heads, latent_dim), generator=gen, device=device, dtype=torch.float32)
    q_nope_quant, q_nope_deq = _quantize_to_float8(q_nope_src, query_scale)

    q_pe_src = torch.randn((num_reqs, num_heads, rope_dim), generator=gen, device=device, dtype=torch.float32)
    q_pe_input = (q_pe_src / query_scale / k_scale.view(1, num_kv_heads, 1)).to(torch.bfloat16).contiguous()
    q_pe_deq = (q_pe_input.float() * query_scale * k_scale.view(1, num_kv_heads, 1)).contiguous()

    kv_base_quant_per_req: list[torch.Tensor] = []
    k_nope_deq_per_req: list[torch.Tensor] = []
    v_nope_deq_per_req: list[torch.Tensor] = []
    k_pe_cache_per_req: list[torch.Tensor] = []
    k_pe_deq_per_req: list[torch.Tensor] = []

    key_scale = k_scale.view(1, num_kv_heads, 1)
    value_scale = v_scale.view(1, num_kv_heads, 1)

    for seq_len in seq_lens:
        kv_base_src = torch.randn((seq_len, num_kv_heads, latent_dim), generator=gen, device=device, dtype=torch.float32)
        kv_base_quant, kv_base_deq = _quantize_to_float8(kv_base_src, key_scale)
        kv_value_deq = (kv_base_quant.float() * value_scale).contiguous()

        k_pe_src = torch.randn((seq_len, num_kv_heads, rope_dim), generator=gen, device=device, dtype=torch.float32)
        k_pe_cache = k_pe_src.to(torch.bfloat16).contiguous()

        kv_base_quant_per_req.append(kv_base_quant)
        k_nope_deq_per_req.append(kv_base_deq)
        v_nope_deq_per_req.append(kv_value_deq)
        k_pe_cache_per_req.append(k_pe_cache)
        k_pe_deq_per_req.append(k_pe_cache.float().contiguous())

    return FA3DecodeSyntheticCase(
        q_nope_quant=q_nope_quant,
        q_nope_deq=q_nope_deq,
        q_pe_input=q_pe_input,
        q_pe_deq=q_pe_deq,
        kv_base_quant_per_req=kv_base_quant_per_req,
        k_nope_deq_per_req=k_nope_deq_per_req,
        v_nope_deq_per_req=v_nope_deq_per_req,
        k_pe_cache_per_req=k_pe_cache_per_req,
        k_pe_deq_per_req=k_pe_deq_per_req,
        query_scale=query_scale,
        k_scale=k_scale.contiguous(),
        v_scale=v_scale.contiguous(),
        seq_lens=seq_lens,
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        softmax_scale=softmax_scale,
        device=device,
    )


def shard_case_by_heads(case: FA3DecodeSyntheticCase, head_start: int, head_end: int) -> FA3DecodeSyntheticCase:
    if not (0 <= head_start < head_end <= case.num_heads):
        raise ValueError(f"Invalid head shard [{head_start}, {head_end}) for {case.num_heads} heads")

    shard = slice(head_start, head_end)
    return FA3DecodeSyntheticCase(
        q_nope_quant=case.q_nope_quant[:, shard, :].contiguous(),
        q_nope_deq=case.q_nope_deq[:, shard, :].contiguous(),
        q_pe_input=case.q_pe_input[:, shard, :].contiguous(),
        q_pe_deq=case.q_pe_deq[:, shard, :].contiguous(),
        kv_base_quant_per_req=[tensor.contiguous() for tensor in case.kv_base_quant_per_req],
        k_nope_deq_per_req=[tensor.contiguous() for tensor in case.k_nope_deq_per_req],
        v_nope_deq_per_req=[tensor.contiguous() for tensor in case.v_nope_deq_per_req],
        k_pe_cache_per_req=[tensor.contiguous() for tensor in case.k_pe_cache_per_req],
        k_pe_deq_per_req=[tensor.contiguous() for tensor in case.k_pe_deq_per_req],
        query_scale=case.query_scale[:, shard, :].contiguous(),
        k_scale=case.k_scale.contiguous(),
        v_scale=case.v_scale.contiguous(),
        seq_lens=list(case.seq_lens),
        block_size=case.block_size,
        num_heads=head_end - head_start,
        num_kv_heads=case.num_kv_heads,
        latent_dim=case.latent_dim,
        rope_dim=case.rope_dim,
        softmax_scale=case.softmax_scale,
        device=case.device,
    )


def compute_request_golden(case: FA3DecodeSyntheticCase, req_idx: int) -> torch.Tensor:
    q_nope = case.q_nope_deq[req_idx].float()
    q_pe = case.q_pe_deq[req_idx].float()
    k_nope = case.k_nope_deq_per_req[req_idx].float()
    v_nope = case.v_nope_deq_per_req[req_idx].float()
    k_pe = case.k_pe_deq_per_req[req_idx].float()

    score_nope = torch.einsum("hd,tgd->hgt", q_nope, k_nope).squeeze(1)
    score_rope = torch.einsum("hd,tgd->hgt", q_pe, k_pe).squeeze(1)
    scores = (score_nope + score_rope) * case.softmax_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,tgd->hgd", probs, v_nope).squeeze(1).contiguous()


def make_fa3_batch(case: FA3DecodeSyntheticCase, req_ids: list[int]) -> FA3DecodeBatch:
    batch_size = len(req_ids)
    max_blocks = max(math.ceil(case.seq_lens[req_id] / case.block_size) for req_id in req_ids)
    total_blocks = sum(math.ceil(case.seq_lens[req_id] / case.block_size) for req_id in req_ids)

    q_nope = _take_first_dim(case.q_nope_quant, req_ids).unsqueeze(2).contiguous()
    q_pe = _take_first_dim(case.q_pe_input, req_ids).unsqueeze(2).contiguous()
    query_scale = _take_first_dim(case.query_scale, req_ids)

    k_nope_cache = torch.zeros(
        (total_blocks, case.num_kv_heads, case.block_size, case.latent_dim),
        dtype=torch.float8_e4m3fn,
        device=case.device,
    )
    k_pe_cache = torch.zeros(
        (total_blocks, case.num_kv_heads, case.block_size, case.rope_dim),
        dtype=torch.bfloat16,
        device=case.device,
    )
    block_table = torch.zeros((batch_size, max_blocks), dtype=torch.int32, device=case.device)

    current_block = 0
    golden_outputs = []
    actual_seq_qlen = []
    actual_seq_kvlen = []
    cumulative_q = 0

    for batch_idx, req_id in enumerate(req_ids):
        seq_len = case.seq_lens[req_id]
        num_blocks = math.ceil(seq_len / case.block_size)
        golden_outputs.append(compute_request_golden(case, req_id))
        actual_seq_kvlen.append(seq_len)

        cumulative_q += 1
        actual_seq_qlen.append(cumulative_q)

        kv_base_quant = case.kv_base_quant_per_req[req_id]
        k_pe_req = case.k_pe_cache_per_req[req_id]

        for block_offset in range(num_blocks):
            block_id = current_block + block_offset
            token_start = block_offset * case.block_size
            token_end = min(token_start + case.block_size, seq_len)
            token_count = token_end - token_start
            block_table[batch_idx, block_offset] = block_id
            k_nope_cache[block_id, :, :token_count, :] = kv_base_quant[token_start:token_end].permute(1, 0, 2)
            k_pe_cache[block_id, :, :token_count, :] = k_pe_req[token_start:token_end].permute(1, 0, 2)

        current_block += num_blocks

    golden_output = torch.stack(golden_outputs, dim=0).contiguous()
    return FA3DecodeBatch(
        q_nope=q_nope,
        q_pe=q_pe,
        k_nope_cache=k_nope_cache,
        k_pe_cache=k_pe_cache,
        block_table=block_table,
        query_scale=query_scale,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        golden_output=golden_output,
    )


def canonicalize_fa3_output(output: torch.Tensor, batch_size: int, num_heads: int) -> torch.Tensor:
    if output.dim() == 4:
        if output.shape[0] == batch_size and output.shape[1] == num_heads:
            return output.squeeze(2).contiguous()
        if output.shape[0] == num_heads and output.shape[1] == batch_size:
            return output.permute(1, 0, 2, 3).squeeze(2).contiguous()
    if output.dim() == 3:
        if output.shape[0] == batch_size and output.shape[1] == num_heads:
            return output.contiguous()
        if output.shape[0] == num_heads and output.shape[1] == batch_size:
            return output.permute(1, 0, 2).contiguous()
    raise ValueError(
        f"Unexpected FA3 output shape {tuple(output.shape)} for batch_size={batch_size}, num_heads={num_heads}"
    )


def run_fa3_operator(batch: FA3DecodeBatch, case: FA3DecodeSyntheticCase) -> torch.Tensor:
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        batch.q_nope,
        batch.k_nope_cache,
        batch.k_nope_cache,
        query_rope=batch.q_pe,
        key_rope=batch.k_pe_cache,
        num_query_heads=case.num_heads,
        num_key_value_heads=case.num_kv_heads,
        input_layout="BNSD",
        atten_mask=None,
        sparse_mode=0,
        softmax_scale=case.softmax_scale,
        query_quant_mode=3,
        key_quant_mode=0,
        value_quant_mode=0,
        dequant_scale_query=batch.query_scale,
        dequant_scale_key=case.k_scale,
        dequant_scale_value=case.v_scale,
        block_table=batch.block_table,
        block_size=case.block_size,
        actual_seq_qlen=batch.actual_seq_qlen,
        actual_seq_kvlen=batch.actual_seq_kvlen,
        return_softmax_lse=True,
    )
    return canonicalize_fa3_output(output, batch.q_nope.shape[0], case.num_heads)


def cosine_similarity_per_req(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual_flat = actual.reshape(actual.shape[0], -1).float()
    expected_flat = expected.reshape(expected.shape[0], -1).float()
    return torch.nn.functional.cosine_similarity(actual_flat, expected_flat, dim=-1)
