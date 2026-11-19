# GLM5Next KDA verify benchmark

Compare the unfused, original fused and batch-tuned implementations on Ascend950DT_9581.
Install torch-npu, fla-npu, vllm-ascend, CANN 9.2.1 and TileLang with the Ascend mixed Cube/Vector backend.
Run from the repository root.

```sh
for seed in 51400 63700; do
    python benchmarks/kernels/kda_verify/benchmark.py \
        --batches 1,2,3,4,5,6,7,8,9,10 --seed "$seed" --out "timing-$seed.json"
done

python benchmarks/kernels/kda_verify/benchmark.py --batches 1:80 --out full-sweep.json
```

Each request verifies six tokens with 64 heads, head dimension 128 and hidden size 4096.
Weights and inputs are synthetic. Compilation and immutable weight packing happen before capture.
Timing covers the input projection, padding when needed, the mixed kernel and the output projection.
All paths share inputs and weights. Each median comes from 20 timed NPU graph replays after warmup,
with ten forwards per replay by default. Replay order rotates and reverses between rounds.

| JSON field | Measurement |
| --- | --- |
| `native_us` | Unfused median latency in microseconds |
| `previous_us` | Original fused median latency in microseconds |
| `fused_us` | Batch-tuned fused median latency in microseconds |
| `speedup` | Unfused latency divided by batch-tuned latency |
| `speedup_vs_previous` | Original fused latency divided by batch-tuned latency |

Each row includes the underlying timing samples. For two seeds, take the geometric mean of the two median ratios at each batch size.

## Batch-specific schedules

| Batch | Tuned implementation |
| --- | --- |
| 1 | Process one head pair per Cube core in a single wave using a contiguous convolution window. |
| 2 and 10 | Lay out history and current tokens consecutively for four-tap convolution. |
| 3-9 | Use the same window, keep the decay prefix in Vector registers and round the recurrence output to BF16 before normalization. |
| 11-40 | Use the original fused schedule. |
| 41-80 | Unroll the four convolution taps in the original fused schedule. |

## Accuracy checks

The harness compares output, convolution cache, normalized values, correction records, KG records and the FP32 checkpoint with the unfused path. It exercises accepted prefixes of one, two and six tokens, plus mixed prefixes across requests. To repeat the additional correctness runs:

```sh
for seed in 22100 32300 48100; do
    python benchmarks/kernels/kda_verify/benchmark.py --check-only \
        --batches 1,2,3,4,5,6,7,8,9,10 --seed "$seed" --out "checks-$seed.json"
done
```

Cache slot zero is reserved. The harness assigns synthetic slots 1 through batch size.
Serving integration of the mixed kernel requires scheduler-supplied cache indices and validation with real model weights.

## Compare unfused, PyPTO and TileLang

With CANN PyPTO 0.3.0 installed, run the three backends in one process on the same NPU.

```sh
python benchmarks/kernels/kda_verify/compare_backends.py --batches 1:80 --seed 83700 --out three-backends.json
```

The PyPTO kernel statically unrolls four independent heads per work item. It implements the gate projections, four-tap convolution, checkpoint projection, causal corrections and gated RMS normalization. Its wrapper gathers the rollback history by accepted prefix. Timing includes that gather, the two outer projections, all cache and record writes and the small-batch padding where used. The TileLang backend reuses invariant weights across requests at batches 1, 3 and 5�C10. The default `benchmark.py` comparison still exposes the earlier TileLang schedules for reproducing their incremental changes.

PyPTO compiles its dynamic-batch program on first use. Compilation can take several minutes on this configuration and is excluded from replay latency.

Each output row contains raw samples and median microseconds keyed by `unfused`, `pypto` and `tilelang`. By default, the runner captures five forwards per graph and measures fifteen interleaved replays. Before timing, it tests accepted lengths 1, 2, 6 and mixed lengths. It also changes the accepted prefix and fills output buffers with NaNs after capture, then replays each graph and compares the output, convolution history, correction and KG records, normalized values and checkpoint with the unfused reference.
