# A5 serial compressor measurement record

**Result: the requested fused/serial >= 0.8 target was not achieved for every case.**

Ten optimization rounds are recorded below. Two final stable Graph task-sum comparisons meet the target. One case remains unstable after a verification repeat and cannot be accepted. No samples were trimmed and no optimization round follows round 10.

## Environment and implementation

- Chip: Ascend950DT_9581V120; architecture 3510; 32 AI Cores and 64 AI Vector cores.
- CANN: 9.2; Python: 3.11.10; torch: `2.10.0+cpu`; matching image-provided torch_npu.
- Immutable container image: `sha256:51095f65e057ce85093e76144b4a4312a029bc20c7522acf72c890eede61e98b`.
- Final kernel commit: `7daf3382a4f7cb54b3e3ab4ae93f788e8ca3220e`. Verification/integration commit: `e131775d67b9dc9106442139abd5e8dd500cd2fd`; it does not change the measured kernels.
- Serial library SHA256: `45abf4062e7ee5d2764675785f33e2aa008756e17fb8e886f25cfffe0aed7eb8`.
- Installed fused library SHA256: `984996ebe42c3038784ba7a7ca9042297e7b1eb5db7296fb6df3d1b19bd9a38b`.
- Final settings: `allow_hf32=True`, `cube_math_type=USE_HF32` (runtime value 3), adaptive launch configuration, NK weight storage.
- Prefill launch: cast/pool/final blocks 64/64/64, 512 pooling threads; decode: 4/64/16, 128 threads.

Device execution is cast -> torch.matmul -> weighted pooling -> RMSNorm/RoPE/cache update. The two projection weights are concatenated before capture. The FP32 projection and full cache contents are checked. This is a direct ASC Torch extension with a prepared Python plan. Existing model dispatch is unchanged. The fused implementation is used as an installed binary black box.

## Final Graph task measurements

Each application emits exactly ten adjacent ArgMax(256 MiB FP16) -> synchronize -> complete Graph replay -> synchronize groups. Every task after the marker belongs to that replay. The complete sequence is summed before taking the median of ten samples. A stable sample set requires `(max - min) / median <= 0.10` for both paths.

C is the complete matmul task duration, including any task-internal work. V comprises the cast, weighted pooling and final normalization/RoPE/cache update tasks. C/(C+V) is the median of ten fractions from the same complete serial replays. It does not measure hardware pipe utilization. A mixed fused kernel is never treated as pure Cube.

| Case | Fused us | Serial us | Fused/serial | C/(C+V) | Acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| prefill-r4-d128 | 50.771 | 72.804 | 0.6974 | 46.08% | below 0.8 |
| prefill-r4-d512 | 57.261 | 140.828 | 0.4066 | 62.33% | below 0.8 |
| prefill-r128-d512 | 82.119 | 101.196 | 0.8115 | 52.19% | passes 0.8 |
| decode-r4-d128 | 21.907 | 27.218 | 0.8049 | 35.47% | passes 0.8 |
| decode-r4-d512 | 22.023 | 35.859 | 0.6142 | 48.88% | invalid: unstable |
| decode-r128-d512 | 21.907 | 49.308 | 0.4443 | 24.36% | below 0.8 |

For decode r4/D512, the initial fused spread was 10.48% and the repeat fused spread was 11.25%. The table shows the repeat observation; it is not a valid accepted ratio. Both attempts and all raw samples are retained. No implementation or launch configuration changed between attempts.

See [final-comparison-verified.json](final-comparison-verified.json) for every replay task, per-sample C/V, spread and acceptance flags. The initial verification is preserved in [final-comparison.json](final-comparison.json). Raw CSV files and corresponding run identities are in `raw/` and `runs/`. Raw-file SHA256 values are included in each summary.

## Ten optimization rounds

The following are diagnostic Event latencies from complete captured Graphs: 31 samples, each averaging 20 replays. They include graph scheduling effects and are separate from the final device task sums above. Do not mix their times with profiler task sums. Every round runs the same six workload classes and passes the output/cache and Graph correctness gates. Units are microseconds; all columns show serial medians.

| Round | Change | Prefill r4/D128 | Prefill r4/D512 | Prefill r128/D512 | Decode r4/D128 | Decode r4/D512 | Decode r128/D512 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [1](iterations/iteration-01.json) | Initial full serial baseline | 220.41 | 334.23 | 318.05 | 30.19 | 35.44 | 141.78 |
| [2](iterations/iteration-02.json) | Two-pass stable pooling; regression retained | 229.04 | 368.93 | 387.70 | 33.28 | 38.67 | 249.52 |
| [3](iterations/iteration-03.json) | Warp-cooperative long windows | 222.32 | 337.32 | 366.36 | 30.65 | 35.84 | 69.69 |
| [4](iterations/iteration-04.json) | Parallel token/channel cache update | 137.63 | 388.17 | 299.00 | 30.96 | 35.56 | 70.05 |
| [5](iterations/iteration-05.json) | Bounded INT32 device indexing | 114.48 | 304.05 | 213.31 | 26.83 | 31.44 | 49.47 |
| [6](iterations/iteration-06-valid.json) | SIMD/DMA prefill cache update | 87.64 | 178.85 | 152.37 | 26.52 | 31.53 | 49.06 |
| [7](iterations/iteration-07.json) | Vector Gather for RoPE | 79.80 | 172.77 | 150.54 | 25.60 | 30.54 | 48.27 |
| [8](iterations/iteration-08.json) | Compile-time pooling specialization | 78.75 | 168.71 | 139.54 | 25.24 | 30.13 | 46.20 |
| [9](iterations/iteration-09.json) | NK weight storage with transpose view | 78.92 | 166.25 | 139.07 | 24.39 | 30.19 | 45.84 |
| [10](iterations/iteration-10.json) | Coalesced long windows, larger cast tiles, adaptive launches, explicit Cube math mode | 65.60 | 133.80 | 93.65 | 23.09 | 28.22 | 38.96 |

Round 10 diagnostic Event fused/serial ratios are prefill-r4-d128: 0.7171, prefill-r4-d512: 0.4269, prefill-r128-d512: 0.8463, decode-r4-d128: 0.7822, decode-r4-d512: 0.7194, decode-r128-d512: 0.5206. Only prefill r128/D512 exceeds 0.8 in that timing method. The acceptance table uses complete Graph device task sums.

The initial round-6 build failed, and a shell error allowed a stale round-5 library to be measured. [The excluded result](excluded/iteration-06-stale-library.json) is retained for audit and is not a valid round. The corrected build and measurement are `iteration-06-valid.json`, commit `2f1b6f2d1628985eb916eab877da6913e6f17486`. Compiler/environment failures are not counted as optimization rounds.

Older immutable round files describe metadata as exported-shape preserving. In fact, their `runtime_shapes` already record the TND capacity adaptation described below. The final verification labels this adaptation explicitly.

## Workload and precision limits

The six shape classes come from the user-provided Zhanlu compressor export for DeepSeek-V4-Flash. Node multiplicities are 21/21/20 per phase. X shapes are prefill [1024,4096] and decode [4,4096]. Metadata values are synthetic and deterministic, rather than runtime trace data. For r4 prefill, 700 valid tokens produce 175 valid output rows. The installed fused binary requires runtime TND RoPE capacity 264, so the exported 175-row RoPE tensors are expanded to 264; the original fixture remains unchanged. Runtime and reference shapes are separately recorded in each Event result. r128 prefill has 1024 valid tokens, eight valid outputs and capacity 16. Decode starts at ratio-1 with one valid token per batch.

Outputs are compared against an independent FP32 CPU formula with rtol=0.02, atol=0.02. The entire cache uses rtol=1e-4, atol=3e-4. Valid output rows and full cache are also compared against the fused black box, and then compared again using actual captured Graph replay. Padded fused output rows are outside the comparison contract. Static weight conversion and buffer allocation occur outside steady-state capture; per-call input conversion is included. No full-model throughput or accuracy result is claimed.

Four [edge validation cases](edge-validation.json) additionally pass Eager and Graph checks for FP16/BF16, BSH/TND, paged/ring caches, both RoPE modes, optional metadata and noncontiguous cache stride(0) padding. Padding remains unchanged. The equivalent reusable test is `tests/ut/ops/test_compressor_serial_npu.py`.

## Reproduction

Build/install instructions are in the [operator README](../../../../csrc/attention/compressor_serial/README.md). Run one fresh msprof directory per case and mode, after configuring the installed fused vendor package and loading its library. The example below uses placeholders for the binary and vendor paths; they must refer to the same installed baseline package.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/path/to/vendors/custom_transformer
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
msprof --aic-mode=task-based --aic-metrics=PipeUtilization \
  --output=results/fresh-profile \
  --application="python -m benchmarks.benchmark_compressor_serial --serial-library build/compressor_serial/libcompressor_serial_ops.so --fused-library /path/to/vllm_ascend_C.so --iteration 10 --hypothesis final-verification --hf32 --cube-math-type USE_HF32 --auto-launch --case prefill-r4-d128 --profile serial --output results/fresh-run.json"
python -m benchmarks.summarize_compressor_serial \
  results/fresh-profile/PROF_.../mindstudio_profiler_output/op_summary_....csv \
  --mode serial --output results/fresh-summary.json
```

The strict summarizer rejects missing markers, changing task signatures and unknown serial tasks. Its output retains every duration. Check sample spread before accepting a comparison. Repeat with `--profile fused --mode fused` in a separate fresh directory.

## Validation scope

- Nine CPU contract/profiler tests pass. The NPU module is skipped on the Windows CPU environment.
- Standalone ASC and parent-project CMake builds/install pass. Automatic loading is exercised in a fresh Python process.
- Six workload classes pass CPU, fused black-box and captured Graph comparisons through all ten valid rounds.
- All 13 tests pass on A5 (`9` contract/profiler tests plus `4` NPU edge cases, 7.57 seconds). The edge cases check Eager and Graph oracle results.
- A complete repository wheel build and model-level test suite were not run.
- `bash format.sh ci` was attempted on Windows but stopped during actionlint dependency bootstrap after ten minutes without progress; this is not a full CI pass. Selected Ruff, codespell, typos, markdownlint, package-init, forbidden-import, context-manager, function-comment and symbolic-meta checks pass.
