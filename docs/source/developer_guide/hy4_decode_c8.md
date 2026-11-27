# HY4 compact C8 Decode and paged IndexPool

HY4's sparse-attention layers select history with IndexPool. Its sliding-window
layers attend to the latest 512 tokens. These optimizations reduce the
intermediates needed by those two paths on Ascend 950DT with CANN 9.3.

## Configuration and supported paths

The following `additional-config` booleans default to false. The external-DP
Decode launch examples enable all three, alongside the existing Draft graph
option. Prefill retains its current implementation.

```json
{
  "hy4flash_dfly_dp_graph": true,
  "hy4flash_cann_indexer_decode": true,
  "hy4flash_cann_c8_decode": true,
  "hy4flash_cann_c8_swa_decode": true
}
```

Merge these fields into the deployment's existing configuration. The validated
layout is P TP4 and D TP1/DP4/EP4, FP8 KV, FULL_DECODE_ONLY, DFly 5, and
PCP1. Each target verification step has six query slots. The C8 sparse path
requires 32 local heads and 2,048 candidates. The sliding path requires 64
local heads and a 512-token window. Paged IndexPool uses 64-token cache blocks,
16 indexer heads, ratio 4, budget 511 and supported rank batches 2, 4 and 8.
Other shapes retain their existing implementation; enabling a flag does not
make every batch use that path.

## Paged IndexPool selection

The packed KV cache also carries pooled IndexPool keys. Previously the batched
path wrote a full FP32 score tensor for every query and history group before
selecting TopK. With eight requests, six verification slots and a 262,144-token
capacity, that explicit score tensor is `8 * 6 * 65536 * 4 = 12 MiB` per layer.

The optional path maintains a derived NPU BF16 layout:
`[physical_blocks, 16, 1, 128]`. Each physical KV block covers 64 tokens, hence
16 pooled keys at ratio four. This costs 4 KiB per physical block per layer.
The CANN LightningIndexerV2 operator consumes this layout with the scheduler's
block table and returns 511 selected group IDs. It does the scoring and TopK
internally, avoiding the explicit Python-visible full score tensor. The operator
may still use its own workspace.

The derived pages have a request lifetime contract:

1. An eager request refreshes its pages from the current packed cache after KV
   transfer or a prefix hit. Reused physical page IDs never imply reusable
   derived contents.
2. Each Decode step mirrors newly completed groups into both layouts. It keeps
   the current IndexPool encoding and RoPE implementation.
3. Group IDs expand to four token IDs each. A partial current group supplies up
   to four additional tokens, giving a fixed width of 2,048 candidates.

Short rows keep all visible history. The decision is per query, so short, long
and entirely padded rows can coexist in one captured graph. Padding remains
`-1`. CANN receives a sufficient metadata extent for TopK, but the dense
short-row result masks out unused history. The derived pages add memory and
request-start refresh work in exchange for less repeated selection work.

## Compact C8 sparse attention

A C8 KV row contains 512 bytes of FP8 latent values, 128 bytes of BF16 RoPE
coordinates and 16 bytes of FP32 scales: 656 bytes total. A materialized FP32
row with 576 coordinates occupies 2,304 bytes.

The optimized path gathers selected rows in their existing packed C8 format.
It reinterprets the bytes for the CANN C8 ABI and calls
`kv_quant_sparse_flash_attention`. It does not convert the whole row to a new
FP8 representation. Dequantization, attention scores, sink normalization and
the value reduction run inside the operator. The output still goes through
the existing value/output projections.

This removes the explicit FP32 KV materialization and its HBM traffic. It does
not remove the packed gather, all temporary storage, or the attention work
itself. The learned sink participates in the softmax denominator; it is not
discarded or approximated by a post-processing correction.

## Compact 512-token sliding attention

The sliding layers use the same compact C8 operator without IndexPool. For a
query at visible position `p`, the start is `max(p + 1 - 512, 0)` and the valid
length is `min(max(p + 1, 0), 512)`. Gathered valid rows form a prefix, followed
by padding. An entirely padded query has length zero and its output is masked
to zero.

This ordering matters: passing only the count of valid rows would be incorrect
if negative leading positions remained in the packed buffer. Fixed-size
buffers and device lengths let graph replay handle short contexts, full
windows and padding without copying sequence lengths to the CPU.

## Validation and limits

The candidate based on `1bec7f497ccd5b884dda47e3b7f9f4ec068e06a4`, with all
three flags enabled, completed AISBench AIME 2026: **30/30 (100%)**, 30 natural
stops, zero generation failures and zero length-limit stops. The longest
response was 92,607 tokens. All four D ranks loaded the candidate and replayed
the target and Draft graphs; supported target batches replayed the paged path.

The run used the official `aime2026_gen_0_shot_chat_prompt` and math scorer,
dataset `evalscope/aime26@12ae93b463353acd4d272dbbc7dbecbcfb902642`, concurrency
16, temperature 0.9, top-p 1, top-k -1, seed 1024, reasoning effort high and an
output limit of 131,072. Service limits were 262,144 context, 128 sequences and
4,096 batched tokens, with BF16 execution and FP8 KV. The paired vLLM revision
was `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe` plus the bundled companion patch.

Hardware component checks covered graph replay with mixed lengths, padding,
page reuse and incremental writes. Paged candidate sets matched the reference;
C8 attention relative L2 error against FP32 attention was about 0.2%. This is
not bitwise equivalence or a guarantee for other workloads.

Historical same-environment AISBench tests on `edc7c505` measured mean stable
ITL of 103.47 ms for baseline and 75.50 ms for the combination, at 133,120 input
tokens, 1,024 output tokens and concurrency 32. Those are historical results,
not a performance measurement of the current precision-fixed PR. Current
performance requires a new comparison on one pinned revision.

Run the NPU regression file with the paired runtime and CANN environment:

```bash
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_hy4_decode_c8.py -q
```

These optimizations are local Decode computation changes. They do not replace
the PD proxy, Mooncake transfer, external DP balancing or DFly acceptance logic.
