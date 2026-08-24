# MsaIndexScore

## Product Support

| Product                                                               | Supported |
| --------------------------------------------------------------------- | :-------: |
| <term>Atlas A2 Training Series/Atlas 800I A2 Inference Product</term> |     √     |
| <term>Atlas A3 Training Series/Atlas A3 Inference Series</term>       |     √     |
| <term>Ascend 950PR/Ascend 950DT</term>                                |     ×     |

## Function Description

- **Operator function**: Computes block scores for the Index Branch of the MSA (MiniMax Sparse Attention) module. For each query token and each sparse KV block, the operator applies a "matmul + max-pool" operation to $Q_{idx}$ and $K_{idx}$ (with optional int8 dequantization) over all causally visible tokens in that block. The resulting per-block importance score, `score`, is used as the input to the subsequent TopK operation in the Index Branch. Prefill and decode share the same interface.

- **Formulas**:

    - Non-quantized case:

    $$
    score = Maxpool[ Q_{idx}@K_{idx}^{T} ]
    $$

    - Int8-quantized case:

    $$
    score = Maxpool[ scale \cdot Q_{idx}@K_{idx}^{T} ]
    $$

    Complete formula:

    $$
    score = Maxpool[(scale \cdot) Q_{idx}@K_{idx}^{T} + atten\_mask] + local\_mask
    $$

    Maxpool takes the maximum along the KV-token dimension within each sparse block of length $block\_size$. `start_loc`, `init_blocks`, and `local_blocks` jointly generate $local\_mask$. High scores are written to the leading blocks and to blocks near the current query so that they are always selected by the subsequent TopK operation. Set `init_blocks` and `local_blocks` to 0 to disable $local\_mask$ and match the Triton raw-score kernel.

## Parameters

> **Notes:**
>
> - B (Batch Size) is the number of input samples.
> - S (Sequence Length) is the sequence length; $S1$ is for the query side and $S2$ is for the key side.
> - T is the sum of sequence lengths across the batch; $T1$ is for the query side and $T2$ is for the key side.
> - N (Head Num) is the number of heads; $N1$ is for the query side and $N2$ is for the key side.
> - D (Head Dim) is the dimension of one attention head.
> - In PageAttention, $block\_num$ is the total number of physical blocks, $block\_size$ is the number of tokens per block, $maxBlockNumPerSeq$ is the maximum number of logical blocks per batch entry (typically $\ge\lceil S2/block\_size\rceil$), and $M_b=\lceil S2/block\_size\rceil$ is the total number of logical blocks.

| Parameter | Input/Output/Attribute | Description | Data Type | Format |
| --------- | ---------------------- | ----------- | --------- | ------ |
| query | Input | $Q_{idx}$ in the formula. Only TND is currently supported, with shape $[T1, N1, D]$. | BFLOAT16, FLOAT16 | ND |
| key | Input | $K_{idx}$ in the formula. Supports TND (`[T2, N2, D]`), BNBD (`[block_num, N2, block_size, D]`), and BBND (`[block_num, block_size, N2, D]`). | BFLOAT16, FLOAT16, INT8 | ND, NZ |
| block_table | Optional input | PageAttention logical-block-to-physical-page mapping. Required for PageAttention. It must be two-dimensional, and its second dimension must be at least $maxBlockNumPerSeq$; shape: $[B, S2/block\_size]$. | INT32 | ND |
| scale | Optional input | Dequantization coefficient $scale$ in the formula. It must be empty for non-quantized inputs and is required for quantized inputs. Shape for PageAttention: $[block\_num, N2, block\_size]$ or $[block\_num, block\_size, N2]$; shape for TND: $[T2, N2]$. | FLOAT | ND, NZ |
| atten_mask | Optional input | Mask controlling causal visibility. Used only when `sparse_mode=3`. A value of 1 excludes a position from computation, while 0 includes it; shape: $[2048, 2048]$. | INT8 | ND |
| actual_seq_qlen | Optional input | Number of valid query tokens in each batch entry. Required when query uses TND. It is a non-decreasing prefix sum with shape $[B+1]$. | INT32 | ND |
| actual_seq_klen | Optional input | Number of valid key tokens in each batch entry. For a TND key, it is a required prefix sum. For PageAttention, it contains the visible $S2$ of each request; shape: $[B]$. | INT32 | ND |
| start_loc | Input | Logical-block index containing the current query, rather than a token prefix. Used to generate $local\_mask$; shape: $[B]$. | INT32 | ND |
| layout_key | Attribute | Key layout: `"TND"`, `"BBND"`, or `"BNBD"`. The aclnn parameter is named `layoutKeyOptional` and defaults to `"BBND"` when omitted. | STRING | - |
| sparse_mode | Attribute | Sparse mode. 0: defaultMask (`atten_mask` is empty); 3: rightDownCausal (requires an `atten_mask` of shape $[2048, 2048]$). | INT64 | - |
| init_blocks | Attribute | Number of leading blocks forced by $local\_mask$. Logical blocks in $[0, init\_blocks)$ receive the high score $1\mathrm{e}30$. Optional; default: $0$. | INT64 | - |
| local_blocks | Attribute | Length of the local window forced by $local\_mask$. The window is $[max(0, start\_loc+1-local\_blocks), start\_loc]$ and receives the high score $1\mathrm{e}29$, overriding `init_blocks` at overlapping positions. Optional; default: $1$ to match MiniMax HF. Set it to $0$ to match the Triton raw score. | INT64 | - |
| score | Output | Per-block importance score $score$ in the formula; shape: $[N1, T1, RoundUp(maxBlockNumPerSeq, 16)]$. | FLOAT | ND |

## Constraints

- Only a $block\_size$ of 128 is currently supported.
- `layout_key` must be explicitly set to `"BBND"`, `"BNBD"`, or `"TND"` and must match the actual shape of `key`.
- In PageAttention (`layout_key` is `"BBND"` or `"BNBD"`), `block_table` is required. For a TND key, `block_table` must be omitted and `actual_seq_klen` must be a `[B+1]` prefix sum.
- In the non-quantized case, `key` must have the same dtype as `query` (currently BFLOAT16 or FLOAT16), and `scale` must be empty. Only INT8 quantization is supported. For quantized inputs, `scale` is required: its PageAttention shape is $[block\_num, N2, block\_size]$ or $[block\_num, block\_size, N2]$, and its TND shape is $[T2, N2]$ with dtype FLOAT. FP8 and <term>Ascend 950PR/Ascend 950DT</term> are not currently supported.
- `sparse_mode` currently supports only 0 and 3:
    - 0 selects defaultMask mode, and `atten_mask` must be empty.
    - 3 selects rightDownCausal mode. `atten_mask` is required with shape $[2048, 2048]$; 1 excludes a position from computation, while 0 includes it.
- `init_blocks` and `local_blocks` must be $\ge 0$ and must not exceed the number of logical blocks (the second dimension of `block_table` for PageAttention, or the aligned final score dimension for TND). When both are 0, $local\_mask$ is skipped.
- The operator outputs block scores only; it does **not** perform TopK.

## Examples

| Invocation | Example | Description |
| ---------- | ------- | ----------- |
| Standalone aclnn operator | [test_aclnn_msa_index_score.cpp](./examples/test_aclnn_msa_index_score.cpp) | End-to-end accuracy self-check with a built-in CPU golden implementation |
| Interface documentation | [aclnnMsaIndexScore.md](./docs/aclnnMsaIndexScore.md) | Two-stage interface documentation |
| Test documentation | [tests/README.md](./tests/README.md) | Test matrix and execution instructions |

Build and run:

```bash
bash build.sh --pkg --soc=ascend910b --ops=msa_index_score -j32
./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/tmp/msa_opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/msa_opp/vendors/custom_transformer
bash build.sh --run_example msa_index_score eager cust --vendor_name=custom
```

> **Implementation Notes (A2/A3)**
>
> - The key layout is selected by the `layout_key` attribute (`layoutKeyOptional` in aclnn). PageAttention **BBND** and **BNBD**, as well as packed **TND**, are supported. TND does not use `block_table`, and `actual_seq_klen` is a `[B+1]` prefix sum. The default layout is `"BBND"`.
> - For `sparse_mode=3`, the host requires `atten_mask[2048,2048]`. The device interprets the visible window according to rightDownCausal semantics, consistent with LightningIndexer, without loading the mask template element by element.
> - `start_loc` is a logical-block index. Together with the `init_blocks` attribute (default: 0) and `local_blocks` attribute (default: 1), it applies `local_mask` after Maxpool.
> - Complete formula: `score = Maxpool[(scale·)Q@Kᵀ + atten_mask] + local_mask`.
