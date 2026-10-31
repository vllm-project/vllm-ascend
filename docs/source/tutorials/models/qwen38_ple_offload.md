# Qwen3.8-Next PLE CPU offload

Qwen4Exp BF16 ngram embedding tables can stay in CANN-registered host memory.
The NPU gathers the selected rows directly; projections and convolution states
remain on the NPU. This feature is opt-in and keeps the checkpoint format unchanged.

## Enable offload

Add the following to a Model Runner V2 deployment:

```bash
--engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
```

Use `--engram-config '{"cpu_offload":false}'` for a resident-table baseline.
DP, PP, PCP, and DCP must all be 1, and ETP must equal TP. Shared tables, THP,
elastic EP, and ubatching are not supported. The CANN runtime must support host
registration and device mapping. Reserve enough host memory for each rank's
BF16 vocabulary shard and for checkpoint loading.

The backend supports BF16 PLE tables only. Quantized model weights can still be
used when the PLE embedding tables themselves remain BF16.

## Sequence parallelism and resource lifetime

The implementation preserves vocabulary row sharding and checkpoint loading.
Each rank returns zeros for indices outside its vocabulary shard. Regular TP
reduces the partial results; HC sequence parallelism uses the outer reduce-scatter.
The PLE and MTP paths use graph-safe dynamic padding for token counts that are
not divisible by TP. Non-SP Ascend MoE outputs are already reduced, so HC shards
these outputs without summing them again.

Model Runner V2 retains host owners across graph teardown and releases them
after its parent shutdown completes. Failed unregister/free operations retain
ownership for retry. Successful release invalidates the embedding parameter.
Forced process termination is not a substitute for testing graceful shutdown.

## Validation

```bash
pytest -q tests/ut/models/test_ple_offload.py
pytest -q tests/e2e/pull_request/one_card/test_ple_host_lookup.py
torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 \
  --master-port=29628 tests/e2e/ple_host_sp.py
```

Tests cover 160-wide rows, invalid IDs, vocabulary and pointer-segment boundaries,
changing graph replay inputs, weight loading, and retryable cleanup.
Real-model graph/MTP/SP validation and matched performance comparisons are also
required when changing the vLLM, CANN, or hardware combination.

Earlier Ascend 950 TP4/EP experiments on a development branch measured 23.84 GiB
of host tables per rank. Model memory decreased from 58.07 to 34.22 GiB per rank,
and the KV budget increased from 20.36 to 44.20 GiB. These historical measurements
are not a benchmark of every subsequent upstream revision.

Compare model-loading memory and the available KV budget, since the service can
reuse the released memory for KV cache. Compare latency at the same actual batch
before testing higher concurrency. Random host gathers can add latency; contiguous
H2D bandwidth does not predict decode TPOT.

The earlier model-level HC SP checks used TP4/EP, MTP3, and FULL_DECODE_ONLY.
FlashComm1 and MoE SP are not included in the validated combination. This change
does not add the independent MRV2 early-queue or PyTorch PLE convolution features.
