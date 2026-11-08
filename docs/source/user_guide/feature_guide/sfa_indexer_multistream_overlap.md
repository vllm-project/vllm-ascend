# Multi-Stream Overlapped SFA Indexer Pre-Processing

## Overview

Sparse Flash Attention (SFA) layers run a Lightning Indexer before the
attention kernel to select the top-k KV blocks. The indexer pre-processing has
two independent strands:

* **k strand** – `wk_weights_proj` (cube), k normalization + RoPE (vector),
  and the C8 quantization of `k_li` (vector);
* **q strand** – `quantize(q_c)` (vector) followed by the `wq_b` quant matmul
  (cube).

By default both strands are issued back-to-back on the current stream, so the
q strand is fully exposed on the critical path even though its cube and vector
phases could run on different engines.

`multistream_sfa_indexer_overlap` moves the q strand to the attention
calculation stream and interleaves the two strands so that each engine (AIC /
AIV) is kept busy.

## Design

```text
main stream:  wk_weights_proj(AIC) -> k vector part(AIV) -> k c8 quant(AIV) -> LightningIndexer
aux  stream:          quantize q_c(AIV) -> q quant matmul(AIC)
```

Four synchronization points keep the strands independent while avoiding
cube contention:

1. `aux.wait_stream(main)` – the auxiliary stream may start `quantize(q_c)`
   as soon as `q_c` is produced; it overlaps with the cube-heavy
   `wk_weights_proj` on the main stream.
2. `aux.wait_stream(main)` – before the q quant matmul, the auxiliary stream
   waits for the main stream cube phase (`wk_weights_proj`), so the two cube
   phases never compete for AIC.
3. The main stream runs the k vector part while the aux stream runs the q
   quant matmul.
4. `main.wait_stream(aux)` – the main stream waits for the aux stream only
   once, right before the k c8 quantization, i.e. before the indexer kernel
   actually needs the projected `q_li`.

All waits use `wait_stream()` (event-reuse friendly) instead of an explicit
`record_event()` / `wait_event()` pair.

`forward` then hands `q_li` over to `indexer_select_post_process(q_li=...)`,
which skips the q projection entirely.

Both the native pre-processing branch (`fused_qkv_a_proj` + `q_a_layernorm`)
and the fused branches (`mul`/`PROLOG_V3` and `MLAPO`) are supported. In the
fused branches the indexer pre-processing is computed after the preprocess
kernel, because `q_c` is only available once that kernel has been issued.

## Enabling

The feature is **disabled by default**. Enable it through `--additional-config`:

```shell
vllm serve <model> \
  --additional-config '{"multistream_sfa_indexer_overlap": true}'
```

## Compatibility and fallback

| Condition | Behaviour |
| --- | --- |
| `multistream_sfa_indexer_overlap = false` (default) | Original serial code path, unchanged kernel order |
| `wq_b` is a static W8A8 linear | q projection split into `quantize` + `npu_quant_matmul`, fully overlapped |
| `q_c` is already `int8` or a `(tensor, scale)` tuple | Quantization is skipped, only the matmul is moved to the aux stream |
| `wq_b` uses any other quantization method | Falls back to the serial path for that layer |

The transformation only changes where each kernel is *issued*; the kernels
themselves, their arguments and their order within a strand are unchanged, so
the numerics are identical to the serial path.

## Performance

Measured with GLM-5.2-w8a8c8 on Ascend 910C. Prefill uses
DP1 / TP16 / PP2 / EP32 and decode uses DP16 / TP2 / EP32.

| Concurrency | Metric | Disabled | Enabled |
| --- | --- | --- | --- |
| 1 | ITL | 62.3 ms | **61.9 ms** |
| 1 | TPOT | 17.80 ms | **17.68 ms** |
| 16 | ITL | 86.1 ms | **83.8 ms** |
| 16 | TPOT | 24.60 ms | **23.94 ms** |

The overlap covers all indexer calls on the decode path. The remaining gap is
the tail of the q quant matmul on the auxiliary stream, which the main stream
still has to wait for before the indexer kernel.

## Accuracy

No accuracy regression: GPQA Diamond stays at ~90.9 with the feature enabled.
The transformation only changes where each kernel is issued, so the numerics
are identical to the serial path.

## Limitations

* The overlap relies on the auxiliary stream being free; it shares the stream
  with the other attention calculations of the same device.
* Long prefill (`num_input_tokens > MLAPO_MAX_SUPPORTED_TOKENS`) falls back to
  the native pre-processing branch, which is supported but has not been
  profiled yet.
