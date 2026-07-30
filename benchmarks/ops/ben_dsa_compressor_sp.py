import argparse
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.attention.context_parallel.compressor_sp import (
    build_compressor_sp_plan,
    collect_boundary_state_row_indices,
    collect_state_row_indices,
    run_compressor_op,
    sync_boundary_state_blocks,
)
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op


def _sync(device):
    if device.type == "npu":
        torch.npu.synchronize()


def _distributed_all_reduce(tensor):
    dist.all_reduce(tensor)
    return tensor


def _scatter(cache, slot_mapping, values, mode):
    if mode == "custom":
        torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, values)
        return
    block_idx = slot_mapping[:, 0].long()
    offset = slot_mapping[:, 1].long()
    cache[block_idx, offset] = values


def _gather_cache_rows(cache, slot_mapping):
    block_idx = slot_mapping[:, 0].long()
    offset = slot_mapping[:, 1].long()
    return cache[block_idx, offset]


def _bench(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000 / iters


def _make_slot_mapping(rows, block_size, device):
    slots = torch.arange(rows, dtype=torch.int32, device=device)
    return torch.stack([slots // block_size, slots % block_size], dim=-1)


def _compressed_row_count(start_pos, tokens, ratio):
    return (start_pos + tokens) // ratio - start_pos // ratio


def _pad_rope_rows(rope, target_rows):
    pad_rows = target_rows - rope.shape[0]
    if pad_rows <= 0:
        return rope
    return torch.cat((rope, rope[:1].expand(pad_rows, -1)), dim=0)


def _make_index(values, index_slice, device):
    if index_slice is not None:
        return None
    return torch.tensor(values, dtype=torch.long, device=device)


def _select_dim0(tensor, indices, index_slice):
    if index_slice is not None:
        start, length = index_slice
        return tensor.narrow(0, start, length)
    return tensor.index_select(0, indices)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepSeek V4 DSA compressor SP local plan."
    )
    parser.add_argument("--ratio", type=int, default=4, choices=[4, 128])
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--rope-head-dim", type=int, default=64)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scatter-mode", choices=["custom", "torch"], default="custom")
    parser.add_argument("--allow-c4-nonaligned", action="store_true")
    parser.add_argument("--allow-c128-nonaligned", action="store_true")
    parser.add_argument(
        "--continuation-tokens",
        type=int,
        default=0,
        help="Run a second chunk against the state produced by the first chunk.",
    )
    parser.add_argument(
        "--sync-boundary-state",
        action="store_true",
        help=(
            "Before the continuation, emulate TP boundary-state synchronization "
            "by copying the first chunk's full boundary block into SP state."
        ),
    )
    parser.add_argument(
        "--distributed-boundary-sync",
        action="store_true",
        help="Use HCCL all-reduce to validate the production boundary-state synchronization path.",
    )
    parser.add_argument(
        "--replay-boundary-state",
        action="store_true",
        help="Recompute C4 chunk-end state from the plan's state-only tail.",
    )
    args = parser.parse_args()

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("This benchmark requires an Ascend NPU runtime.")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl")
        args.rank = dist.get_rank()
        args.tp_size = dist.get_world_size()
    bootstrap_custom_op_env(include_vendor_lib=True)
    if not enable_custom_op():
        raise RuntimeError("Failed to register vllm-ascend custom ops.")
    if not (
        hasattr(torch.ops._C_ascend, "compressor")
        and hasattr(torch.ops._C_ascend, "npu_scatter_nd_update_v2")
    ):
        raise RuntimeError(
            "Required _C_ascend compressor/scatter ops are not registered."
        )

    device = torch.device(f"npu:{int(os.environ.get('LOCAL_RANK', '0'))}")
    torch.manual_seed(args.seed)
    if args.start_pos < 0:
        raise ValueError("--start-pos must be non-negative")
    final_seq_len = args.start_pos + args.tokens
    coff = 2 if args.ratio == 4 else 1
    compressed_rows = _compressed_row_count(args.start_pos, args.tokens, args.ratio)
    rope_rows = min(args.tokens, args.tokens // args.ratio + 1)
    tokens_per_rank = (args.tokens + args.tp_size - 1) // args.tp_size
    local_start = args.rank * tokens_per_rank
    local_end = min(args.tokens, local_start + tokens_per_rank)

    positions = list(range(args.start_pos, final_seq_len))
    plan = build_compressor_sp_plan(
        enabled=True,
        has_prefill=True,
        need_gather_q_kv=True,
        tp_size=args.tp_size,
        compress_ratio=args.ratio,
        is_chunked_prefill=args.continuation_tokens > 0,
        allow_c4_non_aligned=args.allow_c4_nonaligned,
        allow_c128_non_aligned=args.allow_c128_nonaligned,
        input_positions=positions,
        query_start_loc=[0, args.tokens],
        seq_lens=[final_seq_len],
        local_start=local_start,
        local_end=local_end,
    )
    if not plan.enabled:
        raise RuntimeError(f"SP plan fallback: {plan.reason}")

    x = torch.randn(args.tokens, args.hidden_size, dtype=torch.bfloat16, device=device)
    wkv = torch.randn(
        coff * args.head_dim, args.hidden_size, dtype=torch.bfloat16, device=device
    )
    wgate = torch.randn(
        coff * args.head_dim, args.hidden_size, dtype=torch.bfloat16, device=device
    )
    state_block_size = 8 if args.ratio == 4 else 32
    state_blocks = (final_seq_len + state_block_size - 1) // state_block_size
    state_cache_full = torch.zeros(
        state_blocks + 1,
        state_block_size,
        2 * coff * args.head_dim,
        dtype=torch.float32,
        device=device,
    )
    state_cache_sp = torch.zeros_like(state_cache_full)
    ape = torch.randn(
        args.ratio, coff * args.head_dim, dtype=torch.float32, device=device
    )
    norm = torch.ones(args.head_dim, dtype=torch.bfloat16, device=device)
    rope = torch.zeros(
        rope_rows, args.rope_head_dim, dtype=torch.bfloat16, device=device
    )
    state_block_table = torch.arange(
        1, state_blocks + 1, dtype=torch.int32, device=device
    ).view(1, state_blocks)
    cu_seqlens = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)
    start_pos = torch.tensor([args.start_pos], dtype=torch.int32, device=device)
    cache_blocks = (compressed_rows + args.block_size - 1) // args.block_size
    cache_full = torch.empty(
        cache_blocks + 1,
        args.block_size,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    cache_sp = torch.empty_like(cache_full)
    slot_mapping = _make_slot_mapping(compressed_rows, args.block_size, device)

    token_indices = _make_index(plan.token_indices, plan.token_slice, device)
    req_indices = _make_index(plan.req_indices, plan.req_slice, device)
    sp_cu_seqlens = torch.tensor(plan.cu_seqlens, dtype=torch.int32, device=device)
    sp_start_pos = torch.tensor(plan.start_pos, dtype=torch.int32, device=device)
    compressed_row_indices = _make_index(
        plan.compressed_row_indices, plan.compressed_row_slice, device
    )
    output_keep_indices = _make_index(
        plan.output_keep_indices, plan.output_keep_slice, device
    )
    slot_mapping_indices = _make_index(
        plan.slot_mapping_indices, plan.slot_mapping_slice, device
    )
    sp_x = _select_dim0(x, token_indices, plan.token_slice)
    sp_state_block_table = _select_dim0(state_block_table, req_indices, plan.req_slice)
    sp_rope_rows = min(
        len(plan.token_indices),
        len(plan.token_indices) // args.ratio + len(plan.req_indices),
    )
    sp_rope = _pad_rope_rows(
        _select_dim0(rope, compressed_row_indices, plan.compressed_row_slice),
        sp_rope_rows,
    )
    sp_output_slots = _select_dim0(
        slot_mapping, slot_mapping_indices, plan.slot_mapping_slice
    )

    def replay_boundary_state(target_state_cache, target_state_block_table):
        if not args.replay_boundary_state:
            return
        if not plan.supports_boundary_state_replay:
            raise RuntimeError("SP plan has no C4 boundary replay metadata")

        replay_token_indices = _make_index(
            plan.boundary_replay_token_indices,
            plan.boundary_replay_token_slice,
            device,
        )
        replay_req_indices = _make_index(
            plan.boundary_replay_req_indices,
            plan.boundary_replay_req_slice,
            device,
        )
        replay_row_indices = _make_index(
            plan.boundary_replay_compressed_row_indices,
            plan.boundary_replay_compressed_row_slice,
            device,
        )
        replay_x = _select_dim0(
            x, replay_token_indices, plan.boundary_replay_token_slice
        )
        replay_rope_rows = min(
            len(plan.boundary_replay_token_indices),
            len(plan.boundary_replay_token_indices) // args.ratio
            + len(plan.boundary_replay_req_indices),
        )
        replay_rope = _pad_rope_rows(
            _select_dim0(
                rope,
                replay_row_indices,
                plan.boundary_replay_compressed_row_slice,
            ),
            replay_rope_rows,
        )
        run_compressor_op(
            replay_x,
            wkv,
            wgate,
            target_state_cache,
            ape,
            norm,
            replay_rope,
            replay_rope,
            state_block_table=_select_dim0(
                target_state_block_table,
                replay_req_indices,
                plan.boundary_replay_req_slice,
            ),
            cu_seqlens=torch.tensor(
                plan.boundary_replay_cu_seqlens,
                dtype=torch.int32,
                device=device,
            ),
            seqused=None,
            start_pos=torch.tensor(
                plan.boundary_replay_start_pos,
                dtype=torch.int32,
                device=device,
            ),
            rope_head_dim=args.rope_head_dim,
            cmp_ratio=args.ratio,
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )

    def full_compressor():
        return run_compressor_op(
            x,
            wkv,
            wgate,
            state_cache_full,
            ape,
            norm,
            rope,
            rope,
            state_block_table=state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=args.rope_head_dim,
            cmp_ratio=args.ratio,
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )

    def sp_compressor():
        out = run_compressor_op(
            sp_x,
            wkv,
            wgate,
            state_cache_sp,
            ape,
            norm,
            sp_rope,
            sp_rope,
            state_block_table=sp_state_block_table,
            cu_seqlens=sp_cu_seqlens,
            seqused=None,
            start_pos=sp_start_pos,
            rope_head_dim=args.rope_head_dim,
            cmp_ratio=args.ratio,
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )
        replay_boundary_state(state_cache_sp, state_block_table)
        return out

    def full_path():
        out = full_compressor()
        _scatter(cache_full, slot_mapping, out[:compressed_rows], args.scatter_mode)

    def sp_path():
        out = sp_compressor()
        _scatter(
            cache_sp,
            sp_output_slots,
            _select_dim0(out, output_keep_indices, plan.output_keep_slice),
            args.scatter_mode,
        )

    cache_full.zero_()
    cache_sp.zero_()
    state_cache_full.zero_()
    state_cache_sp.zero_()
    full_path()
    sp_path()
    _sync(device)
    compared_slots = sp_output_slots
    full_rows = _gather_cache_rows(cache_full, compared_slots)
    sp_rows = _gather_cache_rows(cache_sp, compared_slots)
    cache_diff = (full_rows.float() - sp_rows.float()).abs()
    max_error = float(cache_diff.max().item()) if cache_diff.numel() else 0.0
    mean_error = float(cache_diff.mean().item()) if cache_diff.numel() else 0.0
    state_block_size = 8 if args.ratio == 4 else 32
    state_rows = torch.tensor(
        collect_state_row_indices(
            token_positions=_select_dim0(
                torch.tensor(positions, dtype=torch.long, device=device),
                token_indices,
                plan.token_slice,
            ),
            req_block_table=_select_dim0(
                state_block_table, req_indices, plan.req_slice
            ),
            cu_seqlens=plan.cu_seqlens,
            state_block_size=state_block_size,
        ),
        dtype=torch.long,
        device=device,
    )
    state_diff = (
        state_cache_full.index_select(0, state_rows).float()
        - state_cache_sp.index_select(0, state_rows).float()
    ).abs()
    state_max_error = float(state_diff.max().item()) if state_diff.numel() else 0.0
    state_mean_error = float(state_diff.mean().item()) if state_diff.numel() else 0.0
    state_mismatch_rows = []
    if state_diff.numel():
        state_row_max = state_diff.reshape(state_diff.shape[0], -1).max(dim=1).values
        state_mismatch_rows = state_rows[state_row_max > 1e-2].cpu().tolist()

    continuation_result = {}
    if args.continuation_tokens > 0:
        next_tokens = args.continuation_tokens
        next_start_pos_value = final_seq_len
        next_final_seq_len = next_start_pos_value + next_tokens
        next_positions = list(range(next_start_pos_value, next_final_seq_len))
        next_tokens_per_rank = (next_tokens + args.tp_size - 1) // args.tp_size
        next_local_start = args.rank * next_tokens_per_rank
        next_local_end = min(next_tokens, next_local_start + next_tokens_per_rank)
        next_plan = build_compressor_sp_plan(
            enabled=True,
            has_prefill=True,
            need_gather_q_kv=True,
            tp_size=args.tp_size,
            compress_ratio=args.ratio,
            is_chunked_prefill=True,
            allow_c4_non_aligned=args.allow_c4_nonaligned,
            allow_c128_non_aligned=args.allow_c128_nonaligned,
            input_positions=next_positions,
            query_start_loc=[0, next_tokens],
            seq_lens=[next_final_seq_len],
            local_start=next_local_start,
            local_end=next_local_end,
        )
        chain_state_blocks = (
            next_final_seq_len + state_block_size - 1
        ) // state_block_size
        chain_state_full = torch.zeros(
            chain_state_blocks + 1,
            state_block_size,
            2 * coff * args.head_dim,
            dtype=torch.float32,
            device=device,
        )
        chain_state_sp = torch.zeros_like(chain_state_full)
        chain_state_block_table = torch.arange(
            1, chain_state_blocks + 1, dtype=torch.int32, device=device
        ).view(1, chain_state_blocks)

        run_compressor_op(
            x,
            wkv,
            wgate,
            chain_state_full,
            ape,
            norm,
            rope,
            rope,
            state_block_table=chain_state_block_table,
            cu_seqlens=cu_seqlens,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=args.rope_head_dim,
            cmp_ratio=args.ratio,
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )
        run_compressor_op(
            sp_x,
            wkv,
            wgate,
            chain_state_sp,
            ape,
            norm,
            sp_rope,
            sp_rope,
            state_block_table=_select_dim0(
                chain_state_block_table, req_indices, plan.req_slice
            ),
            cu_seqlens=sp_cu_seqlens,
            seqused=None,
            start_pos=sp_start_pos,
            rope_head_dim=args.rope_head_dim,
            cmp_ratio=args.ratio,
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )
        replay_boundary_state(chain_state_sp, chain_state_block_table)

        boundary_state_rows = collect_boundary_state_row_indices(
            boundary_positions=[final_seq_len - 1],
            req_block_table=chain_state_block_table,
            state_block_size=state_block_size,
        )
        if args.distributed_boundary_sync:
            boundary_state_rows_tensor = sync_boundary_state_blocks(
                state_cache=chain_state_sp,
                state_block_table=chain_state_block_table,
                boundary_req_indices=torch.tensor(
                    plan.boundary_req_indices, dtype=torch.long, device=device
                ),
                boundary_positions=torch.tensor(
                    plan.boundary_positions, dtype=torch.int32, device=device
                ),
                boundary_owner_mask=torch.tensor(
                    plan.boundary_owner_mask, dtype=torch.bool, device=device
                ),
                all_reduce=_distributed_all_reduce,
            )
            boundary_state_rows = tuple(boundary_state_rows_tensor.cpu().tolist())
        elif args.sync_boundary_state and boundary_state_rows:
            boundary_state_index = torch.tensor(
                boundary_state_rows, dtype=torch.long, device=device
            )
            chain_state_sp.index_copy_(
                0,
                boundary_state_index,
                chain_state_full.index_select(0, boundary_state_index),
            )

        boundary_replay_max_error = None
        if boundary_state_rows:
            boundary_state_index = torch.tensor(
                boundary_state_rows, dtype=torch.long, device=device
            )
            boundary_replay_diff = (
                chain_state_full.index_select(0, boundary_state_index).float()
                - chain_state_sp.index_select(0, boundary_state_index).float()
            ).abs()
            boundary_replay_max_error = float(
                boundary_replay_diff.max().item()
            )

        x_next = torch.randn(
            next_tokens, args.hidden_size, dtype=torch.bfloat16, device=device
        )
        next_rope_rows = min(next_tokens, next_tokens // args.ratio + 1)
        rope_next = torch.zeros(
            next_rope_rows, args.rope_head_dim, dtype=torch.bfloat16, device=device
        )
        next_cu_seqlens = torch.tensor(
            [0, next_tokens], dtype=torch.int32, device=device
        )
        next_start_pos = torch.tensor(
            [next_start_pos_value], dtype=torch.int32, device=device
        )
        next_full_out = run_compressor_op(
            x_next,
            wkv,
            wgate,
            chain_state_full,
            ape,
            norm,
            rope_next,
            rope_next,
            state_block_table=chain_state_block_table,
            cu_seqlens=next_cu_seqlens,
            seqused=None,
            start_pos=next_start_pos,
            rope_head_dim=args.rope_head_dim,
            cmp_ratio=args.ratio,
            coff=coff,
            norm_eps=1e-6,
            rotary_mode=2,
            cache_mode=1,
        )

        if next_plan.enabled:
            next_token_indices = _make_index(
                next_plan.token_indices, next_plan.token_slice, device
            )
            next_req_indices = _make_index(
                next_plan.req_indices, next_plan.req_slice, device
            )
            next_compressed_row_indices = _make_index(
                next_plan.compressed_row_indices, next_plan.compressed_row_slice, device
            )
            next_output_keep_indices = _make_index(
                next_plan.output_keep_indices, next_plan.output_keep_slice, device
            )
            next_full_row_indices = torch.tensor(
                next_plan.local_keep_to_full_row_indices,
                dtype=torch.long,
                device=device,
            )
            next_sp_x = _select_dim0(x_next, next_token_indices, next_plan.token_slice)
            next_sp_cu_seqlens = torch.tensor(
                next_plan.cu_seqlens, dtype=torch.int32, device=device
            )
            next_sp_start_pos = torch.tensor(
                next_plan.start_pos, dtype=torch.int32, device=device
            )
            next_sp_rope_rows = min(
                len(next_plan.token_indices),
                len(next_plan.token_indices) // args.ratio + len(next_plan.req_indices),
            )
            next_sp_rope = _pad_rope_rows(
                _select_dim0(
                    rope_next,
                    next_compressed_row_indices,
                    next_plan.compressed_row_slice,
                ),
                next_sp_rope_rows,
            )
            next_local_out = run_compressor_op(
                next_sp_x,
                wkv,
                wgate,
                chain_state_sp,
                ape,
                norm,
                next_sp_rope,
                next_sp_rope,
                state_block_table=_select_dim0(
                    chain_state_block_table, next_req_indices, next_plan.req_slice
                ),
                cu_seqlens=next_sp_cu_seqlens,
                seqused=None,
                start_pos=next_sp_start_pos,
                rope_head_dim=args.rope_head_dim,
                cmp_ratio=args.ratio,
                coff=coff,
                norm_eps=1e-6,
                rotary_mode=2,
                cache_mode=1,
            )
            next_full_rows = next_full_out.index_select(0, next_full_row_indices)
            next_local_rows = _select_dim0(
                next_local_out, next_output_keep_indices, next_plan.output_keep_slice
            )
            state_token_positions = _select_dim0(
                torch.tensor(next_positions, dtype=torch.long, device=device),
                next_token_indices,
                next_plan.token_slice,
            )
            state_req_block_table = _select_dim0(
                chain_state_block_table, next_req_indices, next_plan.req_slice
            )
            state_cu_seqlens = next_plan.cu_seqlens
        else:
            next_local_out = run_compressor_op(
                x_next,
                wkv,
                wgate,
                chain_state_sp,
                ape,
                norm,
                rope_next,
                rope_next,
                state_block_table=chain_state_block_table,
                cu_seqlens=next_cu_seqlens,
                seqused=None,
                start_pos=next_start_pos,
                rope_head_dim=args.rope_head_dim,
                cmp_ratio=args.ratio,
                coff=coff,
                norm_eps=1e-6,
                rotary_mode=2,
                cache_mode=1,
            )
            # The full Compressor may return a padding row, but this chunk has
            # no valid compressed slot. Production falls back for state update
            # only, so there are no output rows to compare or scatter.
            next_full_rows = next_full_out[:0]
            next_local_rows = next_local_out[:0]
            state_token_positions = torch.tensor(
                next_positions, dtype=torch.long, device=device
            )
            state_req_block_table = chain_state_block_table
            state_cu_seqlens = (0, next_tokens)

        if args.distributed_boundary_sync and next_plan.requires_boundary_state_sync:
            sync_boundary_state_blocks(
                state_cache=chain_state_sp,
                state_block_table=chain_state_block_table,
                boundary_req_indices=torch.tensor(
                    next_plan.boundary_req_indices, dtype=torch.long, device=device
                ),
                boundary_positions=torch.tensor(
                    next_plan.boundary_positions, dtype=torch.int32, device=device
                ),
                boundary_owner_mask=torch.tensor(
                    next_plan.boundary_owner_mask, dtype=torch.bool, device=device
                ),
                all_reduce=_distributed_all_reduce,
            )

        next_row_diff = (next_full_rows.float() - next_local_rows.float()).abs()
        next_cache_diff = next_row_diff
        if next_full_rows.numel() > 0:
            next_slots = _make_slot_mapping(
                next_full_rows.shape[0], args.block_size, device
            )
            next_cache_full = torch.zeros(
                1,
                args.block_size,
                next_full_rows.shape[-1],
                dtype=next_full_rows.dtype,
                device=device,
            )
            next_cache_local = torch.zeros_like(next_cache_full)
            _scatter(next_cache_full, next_slots, next_full_rows, args.scatter_mode)
            _scatter(next_cache_local, next_slots, next_local_rows, args.scatter_mode)
            next_cache_diff = (
                _gather_cache_rows(next_cache_full, next_slots).float()
                - _gather_cache_rows(next_cache_local, next_slots).float()
            ).abs()

        next_state_rows = torch.tensor(
            collect_state_row_indices(
                token_positions=state_token_positions,
                req_block_table=state_req_block_table,
                cu_seqlens=state_cu_seqlens,
                state_block_size=state_block_size,
            ),
            dtype=torch.long,
            device=device,
        )
        next_state_diff = (
            chain_state_full.index_select(0, next_state_rows).float()
            - chain_state_sp.index_select(0, next_state_rows).float()
        ).abs()
        next_state_mismatch_rows = []
        if next_state_diff.numel():
            next_state_row_max = (
                next_state_diff.reshape(next_state_diff.shape[0], -1).max(dim=1).values
            )
            next_state_mismatch_rows = (
                next_state_rows[next_state_row_max > 1e-2].cpu().tolist()
            )
        continuation_result = {
            "continuation_tokens": next_tokens,
            "continuation_start_pos": next_start_pos_value,
            "continuation_boundary_state_rows": boundary_state_rows,
            "continuation_boundary_state_synced": (
                args.sync_boundary_state or args.distributed_boundary_sync
            ),
            "continuation_boundary_state_replayed": args.replay_boundary_state,
            "continuation_boundary_replay_max_error": (
                boundary_replay_max_error
            ),
            "continuation_boundary_state_distributed": args.distributed_boundary_sync,
            "continuation_plan_enabled": next_plan.enabled,
            "continuation_plan_reason": next_plan.reason,
            "continuation_plan_start_pos": next_plan.start_pos,
            "continuation_sp_tokens": len(next_plan.token_indices),
            "continuation_row_max_error": (
                float(next_row_diff.max().item()) if next_row_diff.numel() else 0.0
            ),
            "continuation_cache_max_error": (
                float(next_cache_diff.max().item()) if next_cache_diff.numel() else 0.0
            ),
            "continuation_state_max_error": (
                float(next_state_diff.max().item()) if next_state_diff.numel() else 0.0
            ),
            "continuation_state_mismatch_rows": next_state_mismatch_rows,
        }

    full_compressor_ms = _bench(full_compressor, args.warmup, args.iters, device)
    sp_compressor_ms = _bench(sp_compressor, args.warmup, args.iters, device)
    full_path_ms = _bench(full_path, args.warmup, args.iters, device)
    sp_path_ms = _bench(sp_path, args.warmup, args.iters, device)
    print(
        {
            "ratio": args.ratio,
            "tokens": args.tokens,
            "start_pos": args.start_pos,
            "rank": args.rank,
            "distributed_boundary_sync": args.distributed_boundary_sync,
            "replay_boundary_state": args.replay_boundary_state,
            "full_compressor_ms": full_compressor_ms,
            "sp_compressor_ms": sp_compressor_ms,
            "compressor_speedup": full_compressor_ms / sp_compressor_ms
            if sp_compressor_ms > 0
            else None,
            "compressor_latency_reduction_pct": (
                (full_compressor_ms - sp_compressor_ms) / full_compressor_ms * 100
                if full_compressor_ms > 0
                else None
            ),
            "full_path_ms": full_path_ms,
            "sp_path_ms": sp_path_ms,
            "path_speedup": full_path_ms / sp_path_ms if sp_path_ms > 0 else None,
            "max_error": max_error,
            "mean_error": mean_error,
            "state_max_error": state_max_error,
            "state_mean_error": state_mean_error,
            "state_mismatch_rows": state_mismatch_rows,
            "failed_shape": None,
            "scatter_mode": args.scatter_mode,
            "sp_tokens": len(plan.token_indices),
            "sp_rows": len(plan.slot_mapping_indices),
            "start_pos_zero": plan.start_pos_zero,
            "seq_len_aligned": plan.seq_len_aligned,
            "requires_tail_state_update": plan.requires_tail_state_update,
            "tail_token_ranges": plan.tail_token_ranges,
            "padding_rows": len(plan.padding_row_indices),
            "token_selector": "slice"
            if plan.token_slice is not None
            else "index_select",
            "row_selector": "slice"
            if plan.output_keep_slice is not None
            else "index_select",
            **continuation_result,
        }
    )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
