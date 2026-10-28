# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""[Role] MRv2 logit-bias kernel with Ascend-compatible stop-token stores.
[Invariants] Preserve upstream bias/min-token semantics and wrapper signature.
[Timeouts & Locks] No Host synchronization; restoration avoids unordered locks.
[Side-Effect Policy] Updates only the supplied logits on the caller's stream.

Keep finite-value selection in tl.where instead of the scatter-store mask.
This avoids the unordered-lock initialization race in older Triton-Ascend.
Remove this override after the deployed Triton launcher fixes PR2817.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    # Allowed token IDs.
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    # Logit bias.
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    # Min tokens.
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    restore_when_all_masked_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
    CHECK_ALL_MASKED_ROWS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)

    block = tl.arange(0, BLOCK_SIZE)

    # Allowed token IDs.
    num_allowed_token_ids = tl.load(num_allowed_token_ids_ptr + req_state_idx)
    if num_allowed_token_ids > 0:
        block = tl.arange(0, BLOCK_SIZE)
        mask = block < num_allowed_token_ids

        # Save logits for allowed token IDs.
        allowed_token_ids = tl.load(
            allowed_token_ids_ptr + req_state_idx * allowed_token_ids_stride + block,
            mask=mask,
        )
        logits = tl.load(logits_ptr + token_idx * logits_stride + allowed_token_ids, mask=mask)

        tl.debug_barrier()  # save must read original logits before the -inf overwrite

        # Set logits to -inf for all tokens.
        for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
            offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
            tl.store(
                logits_ptr + token_idx * logits_stride + offset,
                -float("inf"),
                mask=offset < vocab_size,
            )

        tl.debug_barrier()  # -inf overwrite must finish before restoring saved logits

        # Restore logits for allowed token IDs.
        tl.store(
            logits_ptr + token_idx * logits_stride + allowed_token_ids,
            logits,
            mask=mask,
        )

    # Logit bias.
    num_logit_bias = tl.load(num_logit_bias_ptr + req_state_idx)
    if num_logit_bias > 0:
        mask = block < num_logit_bias
        token_ids = tl.load(
            bias_token_ids_ptr + req_state_idx * bias_token_ids_stride + block,
            mask=mask,
        )
        bias = tl.load(bias_ptr + req_state_idx * bias_stride + block, mask=mask)
        logits = tl.load(logits_ptr + token_idx * logits_stride + token_ids, mask=mask)
        logits += bias
        tl.store(logits_ptr + token_idx * logits_stride + token_ids, logits, mask=mask)

    # Apply min tokens.
    num_stop_token_ids = tl.load(num_stop_token_ids_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx)
    min_len = tl.load(min_lens_ptr + req_state_idx)
    if num_stop_token_ids > 0 and pos + 1 < min_len:
        mask = block < num_stop_token_ids
        stop_token_ids = tl.load(
            stop_token_ids_ptr + req_state_idx * stop_token_ids_stride + block,
            mask=mask,
        )
        if CHECK_ALL_MASKED_ROWS:
            should_restore_stop_logits = tl.load(restore_when_all_masked_ptr + req_state_idx)
            if should_restore_stop_logits:
                stop_logits = tl.load(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    mask=mask,
                    other=-float("inf"),
                )

                # Save must read original logits before the -inf overwrite.
                tl.debug_barrier()

                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )

                # The -inf overwrite must be visible to the row scan below.
                tl.debug_barrier()

                row_max = tl.full((), -float("inf"), tl.float32)
                for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
                    offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
                    logits = tl.load(
                        logits_ptr + token_idx * logits_stride + offset,
                        mask=offset < vocab_size,
                        other=-float("inf"),
                    )
                    row_max = tl.maximum(row_max, tl.max(logits, axis=0))

                if row_max == -float("inf"):
                    # Non-finite saved values retain the preceding -inf write.
                    finite = (stop_logits > -float("inf")) & (stop_logits < float("inf"))
                    restored = tl.where(finite, stop_logits, -float("inf"))
                    tl.store(
                        logits_ptr + token_idx * logits_stride + stop_token_ids,
                        restored,
                        mask=mask,
                    )
            else:
                tl.store(
                    logits_ptr + token_idx * logits_stride + stop_token_ids,
                    -float("inf"),
                    mask=mask,
                )
        else:
            tl.store(
                logits_ptr + token_idx * logits_stride + stop_token_ids,
                -float("inf"),
                mask=mask,
            )
