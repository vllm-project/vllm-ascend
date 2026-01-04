import torch

from vllm.triton_utils import triton, tl

# =============================================================================
# Batch-Invariant Flash Attention with KV Cache
# =============================================================================
#
# This section implements a batch-invariant Flash Attention operator using
# Triton for Ascend NPU. The key design principle is that each (batch, head)
# pair is processed independently to ensure batch invariance.
#
# Grid Design: (num_q_blocks, batch_size, num_heads)
# - This ensures Request A's computation never depends on Request B
#
# Block Sizes: Conservative sizes to avoid NPU UB overflow (192KB limit)
# - BLOCK_M = 64 (query block size)
# - BLOCK_N = 64 (key/value block size, adjusted based on HEAD_DIM)
# =============================================================================


def _get_block_sizes(head_dim: int) -> tuple:
    """
    Compute conservative block sizes based on head dimension to avoid NPU UB overflow.

    Approximate UB usage per block:
    - Q_block: BLOCK_M * HEAD_DIM * 2 bytes
    - K_block: BLOCK_N * HEAD_DIM * 2 bytes
    - V_block: BLOCK_N * HEAD_DIM * 2 bytes
    - O_block: BLOCK_M * HEAD_DIM * 2 bytes
    - Float32 accumulators: BLOCK_M * BLOCK_N * 4 bytes

    Args:
        head_dim: The head dimension (e.g., 64, 128, 192, 256)

    Returns:
        Tuple of (BLOCK_M, BLOCK_N)
    """
    BLOCK_M = 64  # Conservative query block size

    # Adjust BLOCK_N based on head dimension to stay within UB limits
    if head_dim <= 64:
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_N = 64
    elif head_dim <= 192:
        BLOCK_N = 32
    else:  # head_dim > 192
        BLOCK_N = 32

    return BLOCK_M, BLOCK_N


@triton.jit
def _flash_attn_with_kvcache_kernel(
    # Q, K, V pointers
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    # Softmax LSE output (optional)
    LSE_ptr,
    # Dimensions
    batch_size,
    seqlen_q,
    seqlen_k,
    num_heads_q,
    num_heads_k,
    head_dim,
    # Strides for Q: (batch, seqlen, heads, head_dim)
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    # Strides for K: (batch, seqlen, heads, head_dim)
    stride_kb,
    stride_ks,
    stride_kh,
    stride_kd,
    # Strides for V: (batch, seqlen, heads, head_dim)
    stride_vb,
    stride_vs,
    stride_vh,
    stride_vd,
    # Strides for O: (batch, seqlen, heads, head_dim)
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    # Stride for LSE: (batch, heads, seqlen)
    stride_lseb,
    stride_lseh,
    stride_lses,
    # Attention parameters
    softmax_scale,
    # Causal masking
    is_causal: tl.constexpr,
    # Softcap (0.0 means disabled)
    softcap,
    # GQA ratio (num_heads_q // num_heads_k)
    gqa_ratio: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Whether to output LSE
    WRITE_LSE: tl.constexpr,
):
    """
    Batch-invariant Flash Attention forward kernel.

    Grid: (num_q_blocks, batch_size, num_heads_q)

    This kernel processes one block of queries per program, ensuring that
    each (batch, head) pair is processed independently for batch invariance.
    """
    # Get program IDs - this is the key to batch invariance
    # Each (batch, head) pair gets its own independent computation
    pid_m = tl.program_id(0)  # Query block index
    pid_b = tl.program_id(1)  # Batch index
    pid_h = tl.program_id(2)  # Head index (query head)

    # Compute KV head index for GQA/MQA
    # For standard attention: gqa_ratio = 1
    # For GQA: gqa_ratio = num_heads_q // num_heads_k > 1
    kv_head_idx = pid_h // gqa_ratio

    # Compute starting positions
    q_start = pid_m * BLOCK_M

    # Create offset arrays
    offs_m = q_start + tl.arange(0, BLOCK_M)  # Query positions [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)  # Key/Value positions [BLOCK_N]
    offs_d = tl.arange(0, HEAD_DIM)  # Head dimension [HEAD_DIM]

    # Compute Q pointer for this (batch, head, q_block)
    Q_block_ptr = (Q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qs +
                   pid_h * stride_qh + offs_d[None, :] * stride_qd)

    # Load Q block with masking
    q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim)
    q = tl.load(Q_block_ptr, mask=q_mask, other=0.0).to(tl.float32)

    # Initialize output accumulator and log-sum-exp
    # Using float32 for numerical stability
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Sum of exp(scores)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # Max score

    # Iterate over K/V blocks
    # For causal masking, we only need to iterate up to the diagonal
    # Note: Causal mask is aligned to bottom-right corner of attention matrix
    # This means query at position i can attend to keys at positions <= i + (seqlen_k - seqlen_q)
    causal_offset = seqlen_k - seqlen_q  # Offset for bottom-right alignment

    if is_causal:
        # For causal attention, the last valid K position for query at position q_pos
        # is q_pos + causal_offset (aligned to bottom-right corner of attention matrix)
        kv_len = tl.minimum(seqlen_k, q_start + BLOCK_M + causal_offset)
    else:
        kv_len = seqlen_k

    num_kv_blocks = tl.cdiv(kv_len, BLOCK_N)

    for kv_block_idx in range(num_kv_blocks):
        k_start = kv_block_idx * BLOCK_N
        offs_kv = k_start + offs_n

        # Compute K pointer for this (batch, kv_head, kv_block)
        K_block_ptr = (K_ptr + pid_b * stride_kb +
                       offs_kv[None, :] * stride_ks + kv_head_idx * stride_kh +
                       offs_d[:, None] * stride_kd)

        # Load K block: [HEAD_DIM, BLOCK_N]
        k_mask = (offs_kv[None, :] < seqlen_k) & (offs_d[:, None] < head_dim)
        k = tl.load(K_block_ptr, mask=k_mask, other=0.0).to(tl.float32)

        # Compute attention scores: Q @ K^T -> [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k, allow_tf32=False)
        scores = scores * softmax_scale

        # Apply softcap if enabled
        if softcap > 0.0:
            scores = softcap * tl.math.tanh(scores / softcap)

        # Apply causal mask if needed
        if is_causal:
            # Causal mask: query at position i can only attend to keys at positions <= i + offset
            # This aligns the mask to the bottom-right corner of the attention matrix
            # Example: if seqlen_q=2, seqlen_k=5, offset=3
            #   Q0 can attend to K0,K1,K2,K3 (positions <= 0+3)
            #   Q1 can attend to K0,K1,K2,K3,K4 (positions <= 1+3)
            causal_mask = (offs_m[:, None] + causal_offset) >= offs_kv[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        # Apply boundary mask for keys beyond seqlen_k
        boundary_mask = offs_kv[None, :] < seqlen_k
        scores = tl.where(boundary_mask, scores, float("-inf"))

        # Online softmax update (numerically stable)
        # m_ij = max of current block scores
        m_ij = tl.max(scores, axis=1)
        # New max
        m_new = tl.maximum(m_i, m_ij)
        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)
        # Compute exp(scores - m_new)
        p = tl.exp(scores - m_new[:, None])
        # Update sum of exp
        l_new = alpha * l_i + tl.sum(p, axis=1)

        # Load V block: [BLOCK_N, HEAD_DIM]
        V_block_ptr = (V_ptr + pid_b * stride_vb +
                       offs_kv[:, None] * stride_vs + kv_head_idx * stride_vh +
                       offs_d[None, :] * stride_vd)

        v_mask = (offs_kv[:, None] < seqlen_k) & (offs_d[None, :] < head_dim)
        v = tl.load(V_block_ptr, mask=v_mask, other=0.0).to(tl.float32)

        # Update output accumulator
        # acc = alpha * acc + P @ V
        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v, allow_tf32=False)

        # Update running statistics
        m_i = m_new
        l_i = l_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Write output
    O_block_ptr = (O_ptr + pid_b * stride_ob + offs_m[:, None] * stride_os +
                   pid_h * stride_oh + offs_d[None, :] * stride_od)

    o_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < head_dim)
    tl.store(O_block_ptr, acc.to(O_block_ptr.dtype.element_ty), mask=o_mask)

    # Write log-sum-exp if requested
    if WRITE_LSE:
        lse = m_i + tl.log(l_i)
        LSE_block_ptr = (LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh +
                         offs_m * stride_lses)
        lse_mask = offs_m < seqlen_q
        tl.store(LSE_block_ptr, lse, mask=lse_mask)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    page_table=None,
    cu_seqlens_q=None,
    cu_seqlens_k_new=None,
    max_seqlen_q=None,
    rotary_seqlens=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
):
    """
    Batch-invariant Flash Attention with KV Cache.

    This function provides a drop-in replacement for flash_attn_with_kvcache
    from flash-attention v3, ensuring batch-invariant computation on Ascend NPU.

    The key batch invariance guarantee: the output for sequence i is
    mathematically identical regardless of:
    - Its position in the batch
    - The presence or absence of other sequences in the batch
    - The sequence lengths of other sequences

    Arguments:
        q: (batch_size, seqlen_q, num_heads, head_dim)
        k_cache: (batch_size, seqlen_k, num_heads_k, head_dim) - KV cache for keys
        v_cache: (batch_size, seqlen_k, num_heads_k, head_dim) - KV cache for values
        k: Optional new keys to append (not yet supported in batch-invariant version)
        v: Optional new values to append (not yet supported in batch-invariant version)
        softmax_scale: Scale factor for attention scores. Default: 1/sqrt(head_dim)
        causal: Whether to apply causal masking
        softcap: Softcap value (0.0 means disabled)
        return_softmax_lse: Whether to return log-sum-exp values

    Returns:
        out: (batch_size, seqlen_q, num_heads, head_dim)
        softmax_lse: (batch_size, num_heads, seqlen_q) if return_softmax_lse=True

    Note:
        - This implementation currently does not support:
          - Appending new K/V to cache (k, v parameters)
          - Rotary embeddings (rotary_cos, rotary_sin)
          - Paged KV cache (page_table)
          - Variable length sequences (cu_seqlens_q)
          - Sliding window attention (window_size)
          - FP8 quantization (q_descale, k_descale, v_descale)
        - These features may be added in future versions
    """
    # Input validation
    assert q.dim(
    ) == 4, f"q must be 4D (batch, seqlen, heads, head_dim), got {q.dim()}D"
    assert k_cache.dim() == 4, f"k_cache must be 4D, got {k_cache.dim()}D"
    assert v_cache.dim() == 4, f"v_cache must be 4D, got {v_cache.dim()}D"

    # Feature limitation warnings/assertions
    if k is not None or v is not None:
        raise NotImplementedError(
            "Appending new K/V to cache is not yet supported in batch-invariant mode. "
            "Please update the cache manually before calling this function.")

    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError(
            "Rotary embeddings are not yet supported in batch-invariant mode. "
            "Please apply rotary embeddings to Q/K before calling this function."
        )

    if page_table is not None:
        raise NotImplementedError(
            "Paged KV cache is not yet supported in batch-invariant mode.")

    if cu_seqlens_q is not None:
        raise NotImplementedError(
            "Variable length sequences (varlen mode) are not yet supported "
            "in batch-invariant mode.")

    if window_size != (-1, -1):
        raise NotImplementedError(
            "Sliding window attention is not yet supported in batch-invariant mode."
        )

    # Extract dimensions
    batch_size, seqlen_q, num_heads_q, head_dim = q.shape
    _, seqlen_k, num_heads_k, _ = k_cache.shape

    # Handle cache_seqlens (actual sequence lengths in cache)
    # cache_seqlens can be:
    # - None: use full seqlen_k for all sequences
    # - int: all sequences have the same length
    # - Tensor: each sequence has its own length
    if cache_seqlens is not None:
        if isinstance(cache_seqlens, int):
            # All sequences have the same length
            cache_seqlens_tensor = None
            effective_seqlen_k = cache_seqlens
            variable_seqlens = False
        elif isinstance(cache_seqlens, torch.Tensor):
            # Variable lengths per sequence
            cache_seqlens_tensor = cache_seqlens.to(torch.int32)
            effective_seqlen_k = seqlen_k  # Will be overridden per-sequence
            variable_seqlens = True
        else:
            raise TypeError(
                f"cache_seqlens must be int or Tensor, got {type(cache_seqlens)}"
            )
    else:
        cache_seqlens_tensor = None
        effective_seqlen_k = seqlen_k
        variable_seqlens = False

    # Validate GQA/MQA configuration
    assert num_heads_q % num_heads_k == 0, (
        f"num_heads_q ({num_heads_q}) must be divisible by num_heads_k ({num_heads_k})"
    )
    gqa_ratio = num_heads_q // num_heads_k

    # Set default softmax scale
    if softmax_scale is None:
        softmax_scale = head_dim**(-0.5)

    # Ensure inputs are contiguous
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()

    # Allocate output tensor
    out = torch.empty_like(q)

    # Allocate LSE tensor if requested
    if return_softmax_lse:
        softmax_lse = torch.empty((batch_size, num_heads_q, seqlen_q),
                                  dtype=torch.float32,
                                  device=q.device)
    else:
        # Create dummy tensor (won't be written to)
        softmax_lse = torch.empty(0, device=q.device)

    # Get conservative block sizes for NPU
    BLOCK_M, BLOCK_N = _get_block_sizes(head_dim)

    # Round up head_dim to power of 2 for efficiency
    HEAD_DIM_PADDED = triton.next_power_of_2(head_dim)

    # Handle variable sequence lengths by processing each batch element separately
    # This ensures batch invariance when sequences have different lengths
    if variable_seqlens:
        for b in range(batch_size):
            seq_len_k = cache_seqlens_tensor[b].item()

            # Compute grid dimensions for this single sequence
            num_q_blocks = triton.cdiv(seqlen_q, BLOCK_M)
            grid = (num_q_blocks, 1, num_heads_q)

            # Launch kernel for this sequence
            _flash_attn_with_kvcache_kernel[grid](
                # Pointers - offset by batch index
                q[b:b + 1],
                k_cache[b:b + 1],
                v_cache[b:b + 1],
                out[b:b + 1],
                softmax_lse[b:b + 1] if return_softmax_lse else softmax_lse,
                # Dimensions
                1,
                seqlen_q,
                seq_len_k,
                num_heads_q,
                num_heads_k,
                head_dim,
                # Q strides (for batch size 1)
                q[b:b + 1].stride(0),
                q[b:b + 1].stride(1),
                q[b:b + 1].stride(2),
                q[b:b + 1].stride(3),
                # K strides
                k_cache[b:b + 1].stride(0),
                k_cache[b:b + 1].stride(1),
                k_cache[b:b + 1].stride(2),
                k_cache[b:b + 1].stride(3),
                # V strides
                v_cache[b:b + 1].stride(0),
                v_cache[b:b + 1].stride(1),
                v_cache[b:b + 1].stride(2),
                v_cache[b:b + 1].stride(3),
                # O strides
                out[b:b + 1].stride(0),
                out[b:b + 1].stride(1),
                out[b:b + 1].stride(2),
                out[b:b + 1].stride(3),
                # LSE strides
                softmax_lse[b:b + 1].stride(0) if return_softmax_lse else 0,
                softmax_lse[b:b + 1].stride(1) if return_softmax_lse else 0,
                softmax_lse[b:b + 1].stride(2) if return_softmax_lse else 1,
                # Attention parameters
                softmax_scale,
                causal,
                softcap,
                gqa_ratio,
                # Block sizes
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                HEAD_DIM=HEAD_DIM_PADDED,
                WRITE_LSE=return_softmax_lse,
            )
    else:
        # All sequences have the same length - can process in parallel
        # Compute grid dimensions
        # Grid: (num_q_blocks, batch_size, num_heads_q)
        # This ensures batch invariance: each (batch, head) pair is independent
        num_q_blocks = triton.cdiv(seqlen_q, BLOCK_M)
        grid = (num_q_blocks, batch_size, num_heads_q)

        # Launch kernel
        _flash_attn_with_kvcache_kernel[grid](
            # Pointers
            q,
            k_cache,
            v_cache,
            out,
            softmax_lse,
            # Dimensions
            batch_size,
            seqlen_q,
            effective_seqlen_k,
            num_heads_q,
            num_heads_k,
            head_dim,
            # Q strides
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            # K strides
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            # V strides
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            # O strides
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            # LSE strides
            softmax_lse.stride(0) if return_softmax_lse else 0,
            softmax_lse.stride(1) if return_softmax_lse else 0,
            softmax_lse.stride(2) if return_softmax_lse else 1,
            # Attention parameters
            softmax_scale,
            causal,
            softcap,
            gqa_ratio,
            # Block sizes
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM_PADDED,
            WRITE_LSE=return_softmax_lse,
        )

    if return_softmax_lse:
        return out, softmax_lse
    return out


# Alias for consistency with the spec naming
flash_attn_with_kvcache_batch_invariant = flash_attn_with_kvcache

# =============================================================================
# Integration Functions
# =============================================================================
# The following functions provide mechanisms to integrate the batch-invariant
# flash attention into vllm-ascend runtime.
# =============================================================================

_original_flash_attn_with_kvcache = None


def get_flash_attn_with_kvcache():
    """
    Get the batch-invariant flash attention function.

    This function returns the batch-invariant implementation of
    flash_attn_with_kvcache that can be used as a drop-in replacement
    for the standard flash-attention v3 implementation.

    Returns:
        The flash_attn_with_kvcache function

    Example:
        >>> from vllm_ascend.batch_invariant import get_flash_attn_with_kvcache
        >>> flash_attn = get_flash_attn_with_kvcache()
        >>> output = flash_attn(q, k_cache, v_cache, causal=True)
    """
    return flash_attn_with_kvcache
