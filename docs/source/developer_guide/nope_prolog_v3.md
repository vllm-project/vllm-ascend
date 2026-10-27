# NoPE PROLOG_V3 integration

NoPE SFA layers, including GLM-5.3-Flash, can opt in to fused preprocessing
after a Dr=0-capable operator is installed and callable through
`torch_npu.npu_mla_prolog_v3`. The same-named function in an older installation
does not establish Dr=0 support.

The current contract requires A2/A3 (DAV_2201), W8A8 dynamic MLA projections,
INT8 per-tile C8 KV storage, D=256, KV rank 512, Q rank 1536 or 2048, tile size
128, and PCP/DCP sizes of one. An indexer's query projection may use the
matching W8A8 dynamic scheme or an unquantized FLOAT linear (normally BF16).
The packed cache width is 528 bytes per slot.
MXFP8 MLA weights and other hardware families remain outside this opt-in path.

W8A8 checkpoints can keep `indexer.wq_b` in floating point. This does not
require requantizing the checkpoint or a separate C8 checkpoint. For this
consumer, the adapter multiplies Prolog's INT8 `q_c` by its FP32 per-token
scale, then casts to the hidden-state dtype before the existing indexer linear.
Quantized indexers keep the prepared-input path without this conversion.
`query_quant_mode=0` controls the main attention query, not the low-rank `q_c`.
Dequantization preserves the returned quantized values, not the original
pre-quantization query; compare end-to-end accuracy and performance with native
preprocessing on the target NPU, including graph replay and MTP.

## Validate and enable

Run the standalone smoke check inside the target container, after activating
the custom operator environment. It uses synthetic weights and private cache
storage, through the same Python entry point as the framework:

```bash
python tools/nope_prolog/smoke_nope_prolog_npu.py --device 0
```

The check covers single-token and four-token inputs, zero-width RoPE output,
and preservation of unmapped cache slots when a `-1` padding index is present.
It does not establish model accuracy, performance, or graph capture support.

Merge these options into the existing server command and restart the server
so the decision is made before weight conversion:

```bash
--kv-cache-dtype int8 --additional-config '{"enable_nope_prolog": true}'
```

Preserve other keys already present in `--additional-config`. Start with
`--enforce-eager`, compare against native preprocessing, then validate FULL
capture/replay, different valid row counts within one bucket, and MTP.

Successful selection logs `SFA NoPE preprocessing: PROLOG_V3`. A static
configuration mismatch logs `PROLOG_V3 rejected` and selects native before
weight conversion. Selection alone does not prove that a request executed the
operator; confirm actual calls with a real request and profiling.
FLOAT indexers additionally log `SFA NoPE indexer: dequantize Prolog INT8 q_c
for floating-point wq_b.` at initialization.

## CPU contract tests

The normal unit tests are in `tests/ut/attention/test_nope_prolog_preparation.py`.
For development without vLLM or torch_npu installed, run:

```bash
python tools/nope_prolog/run_cpu_contract_tests.py .
```

This source-isolated runner requires PyTorch and pytest. It executes production
function bodies and the hardware capability matrix, while replacing surrounding
infrastructure and NPU operations. Its results are not full-module, hardware,
graph, or end-to-end validation.
