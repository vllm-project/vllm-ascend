# HY4 Decode integration (2026-11-25)

This source integration starts at `Hy4-flash-opti` commit `25757d5f1868`.
It retains that branch's sliding-window KV reclamation, Mooncake engine patch,
Prefill optimizations, Engram mapped prefetch, and precise iHC tail fallback.
The accepted Decode/Draft implementations are ported from the tested runtime
based on `2c123f517`; this is executable package source, not a patch archive.

## Runtime and build

Use the matching HY4 vLLM tree and apply `patches/hy4flash-vllm-ba15685.patch`
as required by the existing deployment guide. The experiments used CANN 9.3,
torch-npu 2.10.0.post7 and the deployed CANN transformer custom operators.
This integration does not supply those external binary dependencies.

For optional native CPU Engram lookup, build before starting the service:

```sh
g++ -O3 -std=c++17 -shared -fPIC -fvisibility=hidden \
  vllm_ascend/models/hy_v4_flash/engram_row_lookup_cpu.cpp \
  -o /tmp/hy4-engram-row-lookup.so
```

After preserving any existing library, install the output beside the source
as `engram_row_lookup_cpu.so` before building the wheel or using an editable
checkout. Both the C++ source and any prebuilt library are package data.
Missing native lookup retains the original row preparation path.
Shared pageable CPU buffers are included in the package; an experiment directory
on `PYTHONPATH` is no longer needed.

## Controls

All environment controls are declared in `vllm_ascend/envs.py`. Booleans accept
`0` or `1`. The old `HY4_*` names remain compatibility aliases; an explicit
`VLLM_ASCEND_*` value takes precedence. Restart and recapture after changing
controls. These controls contain no credentials.

| Control | Default | Behavior |
| --- | --- | --- |
| `VLLM_ASCEND_HY4_ENGRAM_BATCH_HASH` | `1` | Batch short Decode hash preparation. |
| `VLLM_ASCEND_HY4_ENGRAM_HASH_GRAPH` | `1` | Replay eligible batch hash arithmetic. |
| `VLLM_ASCEND_HY4_DRAFT_PREP_GRAPH` | `1` | Include supported raw input preparation in Draft graphs. |
| `VLLM_ASCEND_HY4_DFLY_PAGEABLE_ENGRAM` | `0` | Use shared pageable CPU Engram storage. |
| `VLLM_ASCEND_HY4_ENGRAM_ROW_PREP` | `0` | Use native BF16 CPU row lookup and reusable pinned staging. |
| `VLLM_ASCEND_HY4_ENGRAM_TRITON_HASH` | `1` | Use eligible graph-external Triton hashes. |
| `VLLM_ASCEND_HY4_TARGET_PREP_EXTRA_FUSIONS` | `0` | Enable the grouped extra target preparation paths. |
| `VLLM_ASCEND_HY4_TARGET_EMBED_FUSION` | unset | Override the grouped embedding choice; explicit `0` disables it. |
| `VLLM_ASCEND_HY4_TARGET_ROPE_HASH_FUSION` | unset | Override the grouped RoPE/hash choice; explicit `0` disables it. |
| `VLLM_ASCEND_HY4_MOONCAKE_LOCAL_COMM_RES_DIR` | unset | Directory containing per-physical-device resource JSON files. |

Keep `hy4flash_dfly_dp_graph=true` for the validated local TP1/DP4 Draft setup.
Graph capture remains disabled for unsupported distributed Draft layouts.
Native Decode data preparation is enabled on Ascend950 dedicated Decode;
`hy4flash_decode_data_fusions=false` retains the original data preparation.
The target branch's `hy4flash_bulk_c8_decode_reads`,
`hy4flash_fused_ihc_pre_tail`, and `hy4flash_fused_ihc_post` choices remain valid.

## Evidence boundary

The original runtime has four-rank steady Decode profile evidence at 128K/1K,
C32, prefix90 and 16K/1K, C32, zero prefix repetition. Draft raw preparation
passed 14 differential cases and 12 real-weight proposal comparisons.
CPU row preparation passed 80 differential cases with zero error and three
real-weight comparisons per D rank. Extra verify preparation passed 38 prior
and 20 combined cases. These are component checks, not full model quality
benchmarks. The profile runs stopped early and are not complete N128 results.
No matched profiler-off throughput improvement is claimed for this port.

Integration changes preserve existing Prefill paths and package dependencies;
they require their own hardware acceptance before deployment. The historical
service measurements do not constitute a new end-to-end run of this commit.

Integration validation: 1,463 CPU regression tests passed (HY4, Mooncake and
environment controls); the CPU lookup library compiled and loaded successfully.
Changed Python files passed Ruff lint/format and repository AST checks.
The full `bash format.sh ci` run could not initialize its GitHub hook dependency
in the validation container; it is not reported as passing.
