# HY4 Flash on Ascend

HY4 Flash supports text and vision-language serving with Model Runner V2,
BF16 execution, FP8 weights and FP8 KV on Ascend 950DT. DFly, prefix caching
and Mooncake PD separation are supported. Dedicated Prefill also supports
TP2/PP2 in eager mode; match its topology at both connector endpoints.

## Dependencies and launch

Use vLLM `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe` with the bundled
`patches/hy4flash-vllm-ba15685.patch`. The patch supplies matching model,
runner, scheduler and PD interfaces. Source archives need
`SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0.dev0+gba15685` during installation
so development-version feature selection stays correct.

The recorded environment uses Python 3.12, CANN/HIXL 9.3, torch-npu
2.10.0.post7, triton-ascend 3.6.0.post2 and CANN transformer custom operators.
These native dependencies are external. Register torch-npu before probing
Triton or loading RoPE kernels.

Use `examples/hy4flash/start_hy4flash.sh` for one service, or the
[PD launch examples](../../../examples/hy4flash/pd_118/README.md) for separate
Prefill and Decode services. Update model, source and CANN paths for your host.
Mooncake's `kv_connector_extra_config.ascend_local_comm_res_path` must point
to that host's per-device transport descriptors. Start the PD proxy separately.

For native CPU Engram lookup, compile `engram_row_lookup_cpu.cpp` with
`g++ -O3 -std=c++17 -shared -fPIC -fvisibility=hidden` and install the resulting
`engram_row_lookup_cpu.so` beside its source before starting workers.
An absent library retains host gather. CPU offload and DP shared tables
remain controlled by `--engram-config`.

## Optimization controls

| Additional-config key | Default | Supported path |
| --- | --- | --- |
| `hy4flash_dfly_dp_graph` | `false` | Local TP1/DP Draft graph with DFly and `FULL_DECODE_ONLY`; input preparation stays outside the graph. |
| `hy4flash_cann_indexer_decode` | `false` | Paged IndexPool selection for supported Decode batches. |
| `hy4flash_cann_c8_decode` | `false` | Compact C8 sparse Decode attention. |
| `hy4flash_cann_c8_swa_decode` | `false` | Compact C8 sliding-window Decode attention. |

See the [C8 Decode guide](hy4_decode_c8.md) for shape guards and cache lifetime.
Unsupported shapes and layouts retain their fallbacks. Explicit eager mode
wins over Draft graph capture. Other former HY4 per-operator booleans and
experiment environment aliases are removed; use the supported keys above
with generic vLLM/Ascend settings.

## Code layout

| Module | Responsibility |
| --- | --- |
| `attention.py` | Backend metadata, projections, cache writes and shared output arithmetic. |
| `attention_eager.py` | Eager Prefill and Decode, including mixed batches. |
| `attention_decode.py` | Paged Decode selection, graph execution and incremental IndexPool. |
| `attention_ops.py` | Shared geometry, projections, output gates and FP32 Decode attention. |
| `cache.py`, `c8_attention.py` | Packed C8 cache reads/writes, stable compaction and CANN attention adapters. |
| `cache_config.py` | Lightweight cache layout and transfer descriptors used by the common planner and Mooncake. |
| `indexer.py` | IndexPool cache, FP8 encoding, selection and paged Decode updates. |
| `engram.py`, `engram_hash.py`, `engram_ops.py` | Engram layers, hashes, CPU staging and row kernels. |
| `dfly.py`, `dfly_speculator.py`, `dfly_ops.py` | Draft model, speculative execution and input preparation kernels. |
| `ihc.py` | HyperConnection layers and fused first-layer preparation. |
| `model_state.py`, `config.py` | Runner state, graph dispatch, embedding preparation and DFly configuration. |
| `ops.py`, `prefill_ops/` | Decode and Prefill operator interfaces; Prefill kernels are grouped into cache, Indexer and fusion modules. |

The FP32 fallback retains KV and learned-sink normalization through softmax/PV
before the final cast. Stable compaction keeps valid rows and scores aligned.
CANN C8 scaling and rounded MoE/iHC boundaries are preserved. Experimental
prolog, FIA post-correction and alternate GPU scaling implementations have
been deleted.

## Validation

Run `tests/ut/models/test_hy_v4_flash*.py` with the paired runtime. Kernel
preparation tests use `TRITON_INTERPRET=1`. Run the hardware regression
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_hy4_decode_c8.py` on an
available NPU. Component checks do not establish full-model accuracy or speed.

Use AISBench for real-model validation and retain its original result files.
Use the official chat prompt/template for AIME and GPQA. For pure Prefill,
set output length to one and retain raw TTFT. For pure Decode, warm the prefix
through P, retain the P-to-D route, then measure steady-state ITL at the
requested concurrency. Pin P and D to the same revision and record graph
replay and enabled optimization keys.
