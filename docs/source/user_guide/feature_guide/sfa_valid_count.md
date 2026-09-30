# Automatic SFA valid-prefix trimming

The A3 custom sparse flash attention kernels automatically reduce work when
only a prefix of the selected indices is valid. Both C8 and non-C8 paths use
this optimization for the V-template with `sparse_mode=0` and
`sparse_block_size=1`. No additional configuration or operator parameter is
needed. Build and install the custom operators from this source to use it;
the Python/Torch schemas, ACLNN signatures and tiling data layout are unchanged.

## Behavior

The existing operator contract places nonnegative valid indices before `-1`
padding. Index values inside that prefix need not be sorted. The kernel finds
the boundary on device with binary search and shortens its S2 loop. It retains
TopK capacity, actual KV lengths, KV ownership, output layouts and collectives.
An empty prefix keeps the legacy path to preserve raw softmax max/sum and LSE
merge semantics. No host readback or dynamic tensor shape is introduced.

The existing DCP decode remap already produces this representation. The kernel
optimization applies to any eligible A3 call, regardless of DCP or PCP size;
those framework topology settings are not operator inputs. Causal mode
(`sparse_mode=3`), other block sizes and A2/A5 builds keep their existing path.
The build selects A3 code through a compute-unit-specific compiler definition;
there is no runtime opt-in or new binary interface.

## Correctness validation

After building in a supported Ascend development environment:

```bash
pytest -q tests/ut/attention/test_sfa_valid_count.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_flash_attention_valid_count.py
```

The operator tests use an independent CPU attention reference. They cover
compact-prefix lengths around tile boundaries, empty rows, zero KV lengths,
TND/BSND layouts, BF16/FP16 non-C8 and BF16-query INT8 C8, unchanged causal and
block-size behavior, and graphs replayed with changing index contents. Graph
outputs are compared with eager outputs, which are checked against the CPU
reference. These tests do not replace multi-rank model validation.

## Compare baseline and optimized builds

The benchmark uses the original operator interface and can be copied to the
baseline checkout. Run the same script and arguments in two separate processes
with separately built/installed operator packages. The baseline must precede
this entire feature (for this branch, `79f4a63f8`), not the earlier opt-in commits.
Keep the hardware, dependencies and input settings identical, record the exact
source and installed binary provenance, and alternate build order for repeated
performance measurements.

```bash
# In the baseline environment:
python benchmarks/ops/benchmark_sparse_flash_attention_valid_count.py \
  --label baseline-79f4a63f8 --output /tmp/sfa-baseline.pt

# In the optimized environment, using the same script:
python benchmarks/ops/benchmark_sparse_flash_attention_valid_count.py \
  --label candidate-SOURCE_SHA --reference /tmp/sfa-baseline.pt \
  --output /tmp/sfa-candidate.pt
```

Repeat both commands with `--c8` and separate output files for C8. Results retain
input hashes, CPU output/max/sum tensors and median NPU-event latency. The
comparison rejects changed inputs and checks output, raw max/sum and derived
LSE before timing. Include counts 0 and 2048 when assessing overhead, not only
sparse rows. This is an eager operator benchmark, not a model TPOT measurement.

Before claiming a model-level gain, compare full model correctness and TPOT,
TTFT and memory using the same model, requests, parallelism and cache starting
state. Include MTP and mixed-batch regressions. Hardware build, precision, graph
and model/performance acceptance must be reported for the exact tested revision.
