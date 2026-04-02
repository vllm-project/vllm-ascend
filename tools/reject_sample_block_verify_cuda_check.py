#!/usr/bin/env python3
"""Standalone CUDA correctness check for the block-verify Triton kernel.

This repo targets Ascend/NPU, so importing the original module on CUDA will
hit two portability blockers before we can test the kernel logic itself:

1. `vllm_ascend` package import eventually pulls in `torch_npu`.
2. The Triton kernel depends on Ascend-specific helpers such as `get_element`.

This script keeps the block-verify algorithm identical, but uses a tiny CUDA
compatibility harness so we can validate the logic on NVIDIA GPUs such as H100.

It also applies one safety fix that is required on CUDA Triton:
`cu_num_draft_tokens_ptr + offsets - 1` must be masked with `offsets > 0`,
otherwise request 0 can trigger an out-of-bounds load.
"""

from __future__ import annotations

import argparse
import random
import sys

import torch
import triton
import triton.language as tl


def ref_block_verify(
    output_token_ids: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    recovered_token_ids: torch.Tensor,
    uniform_probs: torch.Tensor,
    is_greedy: torch.Tensor,
    no_draft_probs: bool,
) -> None:
    batch_size, max_spec_len_plus_1 = output_token_ids.shape
    output_token_ids.fill_(-1)

    for req_idx in range(batch_size):
        if bool(is_greedy[req_idx].item()):
            continue

        start_idx = 0 if req_idx == 0 else int(cu_num_draft_tokens[req_idx - 1].item())
        end_idx = int(cu_num_draft_tokens[req_idx].item())
        num_draft_tokens = end_idx - start_idx

        if num_draft_tokens == 0:
            output_token_ids[req_idx, 0] = bonus_token_ids[req_idx]
            continue

        accepted_len = 0
        prefix_prob = 1.0
        for pos in range(num_draft_tokens):
            token_idx = start_idx + pos
            draft_token_id = int(draft_token_ids[token_idx].item())
            target_prob = float(target_probs[token_idx, draft_token_id].item())
            draft_prob = 1.0 if no_draft_probs else float(draft_probs[token_idx, draft_token_id].item())

            if draft_prob > 0:
                prefix_prob = min(prefix_prob * target_prob / draft_prob, 1.0)
            else:
                prefix_prob = 0.0

            if pos == num_draft_tokens - 1:
                h_block = prefix_prob
            else:
                next_token_idx = token_idx + 1
                if no_draft_probs:
                    next_draft_token_id = int(draft_token_ids[next_token_idx].item())
                    next_target_prob = float(target_probs[next_token_idx, next_draft_token_id].item())
                    residual_mass = prefix_prob * (1.0 - next_target_prob)
                else:
                    residual_mass = torch.clamp(
                        prefix_prob * target_probs[next_token_idx] - draft_probs[next_token_idx],
                        min=0.0,
                    ).sum().item()
                denom = residual_mass + 1.0 - prefix_prob
                h_block = (residual_mass / denom) if denom > 0 else 0.0

            if float(uniform_probs[token_idx].item()) <= h_block:
                accepted_len = pos + 1

        for pos in range(accepted_len):
            output_token_ids[req_idx, pos] = draft_token_ids[start_idx + pos]

        if accepted_len == num_draft_tokens:
            output_token_ids[req_idx, num_draft_tokens] = bonus_token_ids[req_idx]
        else:
            output_token_ids[req_idx, accepted_len] = recovered_token_ids[start_idx + accepted_len]

        assert accepted_len < max_spec_len_plus_1


@triton.jit
def get_element_fallback(x, position: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    lanes = tl.arange(0, BLOCK_SIZE)
    return tl.sum(tl.where(lanes == position, x, 0), axis=0)


@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_random_sample_block_verify_kernel_cuda_harness(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    bonus_token_ids_ptr,
    recovered_token_ids_ptr,
    uniform_probs_ptr,
    is_greedy_ptr,
    max_spec_len,
    vocab_size,
    vec_len,
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    block_idx = tl.program_id(0)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < vec_len

    is_greedy = tl.load(is_greedy_ptr + offsets, mask, other=1)
    not_greedy_mask = is_greedy == 0

    # CUDA Triton needs the previous load to be masked by offsets > 0;
    # otherwise request 0 can read cu_num_draft_tokens[-1].
    prev_mask = not_greedy_mask & (offsets > 0)
    prev_ends = tl.load(cu_num_draft_tokens_ptr + offsets - 1, prev_mask, other=0)
    start_idxs = tl.where(offsets == 0, 0, prev_ends)
    end_idxs = tl.load(cu_num_draft_tokens_ptr + offsets, not_greedy_mask, other=0)
    n_num_draft_tokens = end_idxs - start_idxs
    loop = (vocab_size + SUB_BLOCK - 1) // SUB_BLOCK

    for req_i in range(BLOCK_SIZE):
        not_greedy = get_element_fallback(not_greedy_mask, req_i, BLOCK_SIZE)
        if not_greedy:
            start_idx = get_element_fallback(start_idxs, req_i, BLOCK_SIZE)
            req_idx = block_idx * BLOCK_SIZE + req_i
            num_draft_tokens = get_element_fallback(n_num_draft_tokens, req_i, BLOCK_SIZE)

            if num_draft_tokens == 0:
                bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1), bonus_token_id)
            else:
                accepted_len = 0
                prefix_prob = 1.0
                for pos in range(num_draft_tokens):
                    token_idx = start_idx + pos
                    draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
                    target_prob = tl.load(target_probs_ptr + token_idx * vocab_size + draft_token_id)

                    if NO_DRAFT_PROBS:
                        draft_prob = 1.0
                    else:
                        draft_prob = tl.load(draft_probs_ptr + token_idx * vocab_size + draft_token_id)

                    if draft_prob > 0:
                        prefix_prob = min(prefix_prob * target_prob / draft_prob, 1.0)
                    else:
                        prefix_prob = 0.0

                    if pos == num_draft_tokens - 1:
                        h_block = prefix_prob
                    else:
                        next_token_idx = token_idx + 1
                        if NO_DRAFT_PROBS:
                            next_draft_token_id = tl.load(draft_token_ids_ptr + next_token_idx)
                            next_target_prob = tl.load(
                                target_probs_ptr + next_token_idx * vocab_size + next_draft_token_id
                            )
                            residual_mass = prefix_prob * (1.0 - next_target_prob)
                        else:
                            residual_mass = 0.0
                            for loop_i in range(loop):
                                vocab_start = loop_i * SUB_BLOCK
                                vocab_offset = vocab_start + tl.arange(0, SUB_BLOCK)
                                next_draft_prob = tl.load(
                                    draft_probs_ptr + next_token_idx * vocab_size + vocab_offset,
                                    mask=vocab_offset < vocab_size,
                                    other=0,
                                )
                                next_target_prob = tl.load(
                                    target_probs_ptr + next_token_idx * vocab_size + vocab_offset,
                                    mask=vocab_offset < vocab_size,
                                    other=0,
                                )
                                residual_prob = tl.maximum(prefix_prob * next_target_prob - next_draft_prob, 0.0)
                                residual_mass += tl.sum(residual_prob, axis=0)

                        denom = residual_mass + 1.0 - prefix_prob
                        h_block = residual_mass / denom if denom > 0 else 0.0

                    uniform_prob = tl.load(uniform_probs_ptr + token_idx)
                    if uniform_prob <= h_block:
                        accepted_len = pos + 1

                for pos in range(accepted_len):
                    token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
                    tl.store(output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id)

                if accepted_len == num_draft_tokens:
                    bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
                    tl.store(
                        output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
                        bonus_token_id,
                    )
                else:
                    recovered_token_id = tl.load(recovered_token_ids_ptr + start_idx + accepted_len)
                    tl.store(
                        output_token_ids_ptr + req_idx * (max_spec_len + 1) + accepted_len,
                        recovered_token_id,
                    )


def make_probs(num_tokens: int, vocab_size: int, device: str) -> torch.Tensor:
    x = torch.rand((num_tokens, vocab_size), device=device, dtype=torch.float32)
    return (x / x.sum(dim=-1, keepdim=True).clamp(min=1e-6)).contiguous()


def run_one_case(
    case_name: str,
    batch_size: int,
    max_spec_len: int,
    vocab_size: int,
    no_draft_probs: bool,
    block_sizes: list[int],
) -> None:
    device = "cuda"
    num_draft_tokens = torch.randint(0, max_spec_len + 1, (batch_size,), device=device, dtype=torch.int32)
    num_draft_tokens[0] = max_spec_len
    cu_num_draft_tokens = torch.cumsum(num_draft_tokens, dim=0, dtype=torch.int32)
    num_tokens = int(cu_num_draft_tokens[-1].item())

    draft_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
    target_probs = make_probs(num_tokens, vocab_size, device)
    draft_probs = None if no_draft_probs else make_probs(num_tokens, vocab_size, device)
    if (not no_draft_probs) and num_tokens > 0:
        zero_mask = torch.rand((num_tokens,), device=device) < 0.2
        rows = torch.arange(num_tokens, device=device)
        draft_probs[rows[zero_mask], draft_token_ids[zero_mask]] = 0.0

    bonus_token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)
    recovered_token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int64)
    uniform_probs = torch.rand((num_tokens,), device=device, dtype=torch.float32)
    is_greedy = torch.rand((batch_size,), device=device) < 0.25
    is_greedy[0] = False

    out_ref = torch.full((batch_size, max_spec_len + 1), -1, device=device, dtype=torch.int64)
    ref_block_verify(
        out_ref,
        cu_num_draft_tokens,
        draft_token_ids,
        draft_probs,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        no_draft_probs,
    )

    for block_size in block_sizes:
        out = torch.full((batch_size, max_spec_len + 1), -1, device=device, dtype=torch.int64)
        grid = (triton.cdiv(batch_size, block_size),)
        rejection_random_sample_block_verify_kernel_cuda_harness[grid](
            out,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            batch_size,
            NO_DRAFT_PROBS=no_draft_probs,
            BLOCK_SIZE=block_size,
            SUB_BLOCK=min(256, max(1, triton.next_power_of_2(vocab_size))),
        )
        torch.cuda.synchronize()

        if not torch.equal(out_ref, out):
            print(f"MISMATCH {case_name} block_size={block_size}")
            print("num_draft_tokens", num_draft_tokens.cpu().tolist())
            print("is_greedy", is_greedy.cpu().tolist())
            print("out_ref", out_ref.cpu().tolist())
            print("out", out.cpu().tolist())
            raise SystemExit(1)


def run_repo_fixed_case() -> None:
    device = "cuda"
    batch_size = 7
    max_spec_len = 3
    vocab_size = 5

    cu_num_draft_tokens = torch.tensor([2, 2, 5, 8, 11, 14, 15], dtype=torch.int32, device=device)
    draft_token_ids = torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=torch.int64, device=device)
    target_probs = torch.tensor(
        [
            [0.4, 0.3, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.2, 0.4, 0.1],
            [0.1, 0.4, 0.1, 0.1, 0.3],
            [0.2, 0.1, 0.4, 0.1, 0.2],
            [0.4, 0.2, 0.1, 0.2, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.3, 0.1],
            [0.4, 0.2, 0.1, 0.2, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.3, 0.1],
            [0.4, 0.4, 0.1, 0.0, 0.1],
            [0.4, 0.3, 0.1, 0.1, 0.1],
            [0.4, 0.0, 0.5, 0.0, 0.1],
            [0.4, 0.1, 0.3, 0.1, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    )
    uniform_probs = torch.tensor(
        [0.9, 0.0, 0.9, 0.7, 0.8, 0.5, 0.45, 1.0, 0.5, 0.45, 1.0, 0.39, 0.4, 0.1, 0.3],
        dtype=torch.float32,
        device=device,
    )
    bonus_token_ids = torch.full((batch_size,), max_spec_len + 1, dtype=torch.int64, device=device)
    recovered_token_ids = torch.full((draft_token_ids.shape[0],), max_spec_len, dtype=torch.int64, device=device)
    is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
    is_greedy[4] = True

    out_ref = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64, device=device)
    ref_block_verify(
        out_ref,
        cu_num_draft_tokens,
        draft_token_ids,
        None,
        target_probs,
        bonus_token_ids,
        recovered_token_ids,
        uniform_probs,
        is_greedy,
        True,
    )

    expected = [[0, 1, 4, -1], [4, -1, -1, -1],
        [3, -1, -1, -1], [3, -1, -1, -1],
        [-1, -1, -1, -1], [3, -1, -1, -1],
        [0, 4, -1, -1]]
    assert out_ref.cpu().tolist() == expected

    for block_size in [1, 2, 4, 8]:
        out = torch.full((batch_size, max_spec_len + 1), -1, dtype=torch.int64, device=device)
        grid = (triton.cdiv(batch_size, block_size),)
        rejection_random_sample_block_verify_kernel_cuda_harness[grid](
            out,
            cu_num_draft_tokens,
            draft_token_ids,
            None,
            target_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            is_greedy,
            max_spec_len,
            vocab_size,
            batch_size,
            NO_DRAFT_PROBS=True,
            BLOCK_SIZE=block_size,
            SUB_BLOCK=8,
        )
        torch.cuda.synchronize()
        if not torch.equal(out_ref, out):
            print(f"FIXED CASE MISMATCH block_size={block_size}")
            print("out_ref", out_ref.cpu().tolist())
            print("out", out.cpu().tolist())
            raise SystemExit(1)

    print("PASS fixed_case", expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3, help="Random trials per shape.")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="full",
        help="Quick runs fewer shapes; full matches the H100 validation sweep.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available in this environment.", file=sys.stderr)
        return 2

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_repo_fixed_case()

    if args.mode == "quick":
        batch_sizes = [1, 7]
        max_spec_lens = [1, 3]
        vocab_sizes = [5, 67]
    else:
        batch_sizes = [1, 2, 3, 7, 16]
        max_spec_lens = [1, 2, 3, 5]
        vocab_sizes = [5, 17, 67]

    case_id = 0
    for no_draft_probs in [True, False]:
        for batch_size in batch_sizes:
            for max_spec_len in max_spec_lens:
                for vocab_size in vocab_sizes:
                    for _ in range(args.trials):
                        case_id += 1
                        run_one_case(
                            case_name=(
                                f"case_{case_id}_ndp_{int(no_draft_probs)}"
                                f"_b{batch_size}_m{max_spec_len}_v{vocab_size}"
                            ),
                            batch_size=batch_size,
                            max_spec_len=max_spec_len,
                            vocab_size=vocab_size,
                            no_draft_probs=no_draft_probs,
                            block_sizes=[1, 2, 4, 8],
                        )

    print(f"ALL PASS ({case_id} random cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
