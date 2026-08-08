import random

import pytest
import torch
import torch_npu

from vllm_ascend._310p.ops.custom_fused_infer_attention import (
    custom_fused_infer_attention_v310,
)
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(autouse=True)
def _register_custom_op():
    enable_custom_op()


def _generate_random_block_table(kv_seq_lens, block_size, total_physical_blocks):
    B = len(kv_seq_lens)
    max_blocks_per_seq = int((max(kv_seq_lens) + block_size - 1) // block_size) + 2
    block_table = torch.full([B, max_blocks_per_seq], -1, dtype=torch.int32)
    available_blocks = list(range(total_physical_blocks))

    for b in range(B):
        cur_kv_len = kv_seq_lens[b].item()
        needed_blocks = (cur_kv_len + block_size - 1) // block_size

        if needed_blocks > len(available_blocks):
            raise ValueError(f"Not enough physical blocks: need {needed_blocks}, available {len(available_blocks)}")

        chosen_blocks = random.sample(available_blocks, needed_blocks)
        for cb in chosen_blocks:
            available_blocks.remove(cb)
        block_table[b, :needed_blocks] = torch.tensor(chosen_blocks, dtype=torch.int32)

    return block_table


def _compute_golden_output_cpu(
    query,
    key_cache,
    value_cache,
    num_heads,
    num_key_value_heads,
    head_dim,
    block_size,
    block_table,
    query_lens_abs,
    kv_seq_lens,
    scale,
    layout,
):
    B = len(query_lens_abs)
    out_list = []
    q_offset = 0

    for b in range(B):
        q_len = query_lens_abs[b].item()

        if layout == "TND":
            cur_query = query[q_offset : q_offset + q_len]
            q_offset += q_len
        elif layout == "BSND":
            cur_query = query[b, :q_len]
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        cur_kv_len = kv_seq_lens[b].item()
        cur_block_indices = block_table[b]
        num_blocks_needed = (cur_kv_len + block_size - 1) // block_size

        keys_list = []
        values_list = []
        for i in range(num_blocks_needed):
            block_idx = cur_block_indices[i].item()
            keys_list.append(key_cache[block_idx])
            values_list.append(value_cache[block_idx])

        full_keys = torch.cat(keys_list, dim=0)[:cur_kv_len, :]
        full_values = torch.cat(values_list, dim=0)[:cur_kv_len, :]

        full_keys = full_keys.view(cur_kv_len, num_key_value_heads, head_dim).permute(1, 0, 2)
        full_values = full_values.view(cur_kv_len, num_key_value_heads, head_dim).permute(1, 0, 2)

        if num_key_value_heads != num_heads:
            group = num_heads // num_key_value_heads
            full_keys = full_keys.repeat_interleave(group, dim=0)
            full_values = full_values.repeat_interleave(group, dim=0)

        q_trans = cur_query.transpose(0, 1).to(torch.float32)
        k_trans = full_keys.transpose(-1, -2).to(torch.float32)
        attn_scores = torch.matmul(q_trans, k_trans) * scale

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        cur_out = torch.matmul(attn_weights, full_values.to(torch.float32))

        out_list.append(cur_out.transpose(0, 1))

    if layout == "TND":
        return torch.cat(out_list, dim=0).to(query.dtype)
    elif layout == "BSND":
        max_q_len = query.shape[1]
        padded_out = torch.zeros([B, max_q_len, num_heads, head_dim], dtype=torch.float32)
        for b in range(B):
            q_len = query_lens_abs[b].item()
            padded_out[b, :q_len] = out_list[b]
        return padded_out.to(query.dtype)


@pytest.mark.parametrize(
    "layout,head_dim,block_size",
    [
        ("TND", 128, 128),
        ("BSND", 128, 128),
    ],
)
def test_custom_fused_infer_attention_v310(layout, head_dim, block_size):
    """Smoke test: one MHA case per layout+head_dim combo, eager mode."""
    random.seed(42)
    torch.manual_seed(42)

    dtype = torch.float16
    atol = 1e-4
    scale = head_dim**-0.5
    num_heads = 4
    num_kv_heads = 4
    B = 2

    query_lens_cpu_abs = torch.tensor([random.randint(1, 128) for _ in range(B)], dtype=torch.int64)
    kv_seq_lens_cpu = torch.tensor([random.randint(1, 256) for _ in range(B)], dtype=torch.int64)

    total_needed_blocks = sum((seq_len.item() + block_size - 1) // block_size for seq_len in kv_seq_lens_cpu)
    block_num = total_needed_blocks + 20
    block_table_cpu = _generate_random_block_table(kv_seq_lens_cpu, block_size, block_num)
    block_table = block_table_cpu.npu()

    key_cache = torch.randn([block_num, block_size, num_kv_heads * head_dim], dtype=dtype).npu()
    value_cache = torch.randn([block_num, block_size, num_kv_heads * head_dim], dtype=dtype).npu()
    kv_seq_lens_npu = kv_seq_lens_cpu.npu()

    if layout == "TND":
        T_q = query_lens_cpu_abs.sum().item()
        query = torch.randn([T_q, num_heads, head_dim], dtype=dtype).npu()
    elif layout == "BSND":
        max_q_len = query_lens_cpu_abs.max().item()
        query = torch.randn([B, max_q_len, num_heads, head_dim], dtype=dtype).npu()
    query_lens_npu = query_lens_cpu_abs.npu()

    # Reshape KV cache to NZ layout expected by the operator
    k_cache = (
        key_cache.reshape(key_cache.shape[0], key_cache.shape[1], -1)
        .reshape(key_cache.shape[0], key_cache.shape[1], -1, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    v_cache = (
        value_cache.reshape(value_cache.shape[0], value_cache.shape[1], -1)
        .reshape(value_cache.shape[0], value_cache.shape[1], -1, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    attention_output_npu = custom_fused_infer_attention_v310(
        query=query,
        key=k_cache,
        value=v_cache,
        actual_seq_lengths_q=query_lens_npu.tolist(),
        actual_seq_lengths_kv=kv_seq_lens_npu.tolist(),
        block_table=block_table,
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        block_size=block_size,
        input_layout=layout,
        scale_value=scale,
    )
    torch_npu.npu.synchronize()

    golden_output_cpu = _compute_golden_output_cpu(
        query=query.detach().cpu(),
        key_cache=key_cache.detach().cpu(),
        value_cache=value_cache.detach().cpu(),
        num_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        block_table=block_table_cpu,
        query_lens_abs=query_lens_cpu_abs,
        kv_seq_lens=kv_seq_lens_cpu,
        scale=scale,
        layout=layout,
    )

    npu_out = attention_output_npu.detach().cpu()
    cpu_out = golden_output_cpu

    if layout == "BSND":
        valid_npu = []
        valid_cpu = []
        for b in range(B):
            q_len = query_lens_cpu_abs[b].item()
            valid_npu.append(npu_out[b, :q_len].flatten())
            valid_cpu.append(cpu_out[b, :q_len].flatten())
        npu_valid = torch.cat(valid_npu)
        cpu_valid = torch.cat(valid_cpu)
        diff_mean = (npu_valid - cpu_valid).abs().mean()
    else:
        diff_mean = (npu_out - cpu_out).abs().mean()

    assert diff_mean <= atol, f"layout={layout} hd={head_dim} bs={block_size}: mean_diff={diff_mean:.6f} > atol={atol}"
