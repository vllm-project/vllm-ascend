# SplitQuant operator regression

Build and install vLLM-Ascend with custom kernels enabled on an Atlas A2 host,
then run the equivalence test from the repository root. The test loads the
installed `vllm_ascend/lib/libvllm_ascend_splitquant_dense.so` library.

```bash
python csrc/splitquant_dense_fused/test/run_equivalence.py \
  --profile generic --out /tmp/splitquant-equivalence
```

Use `--profile 30b` for the MoE-sized cases. Add `--reference <outputs.pt>` to
perform an elementwise comparison against a saved reference. Profiling uses
`--profiler-per-case --perf-cases <cases.jsonl>` and the fixed five warmup and
five active iterations implemented by the test.

The test requires an NPU, CANN, torch-npu, and a completed vLLM-Ascend build;
it is not a CPU-only unit test.
