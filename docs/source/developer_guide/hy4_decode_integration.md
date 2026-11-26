# HY4 Decode and Prefill integration after the November 26 revert

This integration extends Prefill commit bbec613749, including its parent
821c4a1a4, with the cleaned Decode/Draft implementations. Shared attention,
model-state and runner changes preserve the Prefill paths and select Decode
fusion paths by capability and layout. Layer-specific Prefill metadata selection
is retained. The combined implementation completed startup and real inference
in an independent clean container; model quality and throughput were not evaluated.

## Configuration

The only new Decode optimization boolean is `hy4flash_dfly_dp_graph` in
`--additional-config`. It defaults to false. Set it to true for local TP1/DP
Draft graphs with FULL_DECODE_ONLY and DFly speculation. Unsupported distributed
layouts retain eager Draft. Supported input preparation uses fused kernels outside the Draft graph;
only the Draft model body is captured. Explicit eager mode takes precedence.

The former Decode optimization booleans, environment variables and HY4_* aliases
are removed. Remove them from launch scripts. Removed additional-config names
are rejected instead of silently accepted. Old environment exports have no
consumer. Generic vLLM/Ascend options and Engram storage configuration remain.
The Prefill controls already supplied by commits 821c4a1a4 and bbec613749 remain.

Decode fusion, hash preparation, embedding preparation and row staging select
supported paths automatically. Helpers retain shape/device/layout fallbacks.
Native CPU row lookup falls back when its library is absent. CPU tables read
through host gather use pageable storage; mapped Prefill storage retains UVA.
Precise iHC Pre arithmetic is preserved; this candidate does not restore the
historical full native Decode MHC Pre replacement. C8 scaling retains the
established checkpoint contract. Experimental alternative MLA prolog/logit
backends are not selected.

## Dependencies

Use the matching HY4 vLLM source and existing
`patches/hy4flash-vllm-ba15685.patch`. The recorded runtime used CANN 9.3,
torch-npu 2.10.0.post7 and CANN transformer custom operators. These external
binary dependencies are not supplied by this candidate.

Pin vLLM to `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe` before applying
the companion patch. This is a development revision, not the released 0.33.0.
When installing from a source archive without Git version metadata, provide
`SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0.dev0+gba15685` to the editable vLLM
installation. A release version label selects incompatible Kimi interfaces and
graph-size configuration paths. Register torch-npu before probing Triton;
`examples/hy4flash/pd_118/bin/vllm` checks source paths and RoPE launchability.

For native CPU lookup, compile the packaged source before service startup:

```sh
g++ -O3 -std=c++17 -shared -fPIC -fvisibility=hidden \
  vllm_ascend/models/hy_v4_flash/engram_row_lookup_cpu.cpp \
  -o /tmp/hy4-engram-row-lookup.so
```

After preserving an existing library, install it beside the source as
`engram_row_lookup_cpu.so`. Missing libraries retain normal row preparation.
Mooncake reads per-device transport descriptors from the existing connector
`kv_connector_extra_config.ascend_local_comm_res_path` directory. No HY4
environment variable is required.

## Completed startup validation

The combined source was tested with eight Ascend 950PR devices, Python 3.12.13,
CANN 9.3, torch 2.10.0+cpu, torch-npu 2.10.0.post7, triton-ascend 3.6.0.post2,
transformers 5.19.0, mooncake-transfer-engine-npu 0.3.14 and
cann-ops-transformer 1.0.0. The new container reused native libraries from the
existing CANN 9.3 image, with independently prepared vLLM and Ascend Python sources.

P used TP4 and eager execution on devices 0-3. D used TP1/DP4/EP4 on devices 4-7,
DFLY5 and `hy4flash_dfly_dp_graph=true`. Both used FP8 KV, an 8192-token context,
eight sequences and 4096 batched tokens. D graph request sizes were [1, 2, 4, 8]
with a 48-token capture limit. Engram CPU offload was enabled; D shared its CPU
tables across DP ranks. The existing Mooncake descriptor path supplied this
host's per-device resource JSON files.

All five health endpoints returned HTTP 200. Four proxy completion requests,
each with a 450-token prompt, returned HTTP 200 and 32 generated tokens. All four
D workers completed target/Draft capture and logged target replay; Draft replay
was also observed on the ranks serving the smoke requests. The focused
Prefill/Draft regression suite passed 59 tests. The isolated runtime's 4931
tracked Ascend file contents matched the resolved source before this documentation
update. These checks establish startup and execution, not accuracy or performance.

The private [reproduction archive](https://github.com/peterDengcx/vllm-ascend-118/tree/codex/hy4-verify-fusions-20261125/examples/hy4flash/experiments/20261125-verify-fusions/latest-20261126/reproduce-stack-20261126)
contains the resolved overlay, installation and startup scripts, dependency
hashes, health results, four request/response pairs and graph/test excerpts.
Weights, the existing native image and machine-specific transport descriptors
are external prerequisites. For another machine, supply matching native operators,
update the launcher paths and export that machine's transport descriptors.

## Further validation

`examples/hy4flash/serve_decode_external_dp.sh` follows the supplied eight
positional arguments and adds only the Draft graph boolean. Use TP1/PCP1 for
the intended DP4 local Draft layout. Check actual source imports, all four
rank captures and model replay before evaluating the workload.
Compare this candidate against the reverted baseline for outputs, acceptance,
memory and profiler-off latency/throughput. CPU checks do not establish NPU
correctness or performance.

For profiling, first observe real target traffic at steady state on every D
rank. Keep loading, compilation, capture and ramp-up outside the window. Record
per-rank batches and throughput/latency observations, then capture only pure
Decode for three seconds. Stop AISBench immediately after capture and parse
later. An early-stopped N128 load is not a complete performance result.
