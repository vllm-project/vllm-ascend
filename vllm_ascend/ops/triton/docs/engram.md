# Engram kernels

## Responsibilities

Model code owns embedding tables, request state and auxiliary streams. The
Triton implementations live in `ops/triton`; moving them does not change the
table layout or arithmetic.

| Source | Operation and layout |
| --- | --- |
| `engram_lookup.py` | Gather global hash IDs from a local vocabulary shard. Device tables and registered host tables support BF16, INT8/FP32 scales and MXFP8/E8M0 scales. Each program writes one token/head row; out-of-shard IDs produce zeros and head padding remains untouched. |
| `engram_lookup.py`: `_decode_e8m0` | Decode unsigned exponent bytes. Code 0 represents `2^-127`, and 255 represents NaN. Mask sign extension before constructing FP32 bits. |
| `engram_hash.py` | Map tokens to requests, read preceding tokens from the current chunk, lookback window or slot cache, then calculate per-layer n-gram hashes. Image/dead tokens stop valid history. Empty requests must not steal another request's tokens. |
| `engram_lookback.py` | Gather the tokens preceding each request's chunk from device token history. Respect request-state reordering and fill missing history and padded requests with `-1`. |
| `engram_gate.py` | Fuse the FP32 normalization, key/query weighting, gate and residual addition used by native A5 Engram checkpoints. Retain the unfused operation order rather than contracting the residual into an FMA. |

## Current hardware usage

Hash generation and lookback gathering are shared model paths, including A3.
Lookup also serves A3 BF16/INT8 tables; its MXFP8 path is selected by
`ENGRAM_MXFP8`, currently an A5 capability. The fused `engram_gate.py` path is
selected by `ENGRAM_UNROTATED_GATE`, also currently an A5 capability. Moving
these kernels into `ops/triton` does not change their hardware routing. These
call-path descriptions do not establish A3 runtime validation or CI coverage.

## Memory and synchronization

Host lookup receives an NPU-visible pointer table created after ACL host-memory
registration. Pointers identify chunks so large tables do not require 32-bit
offset arithmetic across the whole allocation. The model must keep registrations
alive until the consuming stream finishes. Kernel execution does not register or
release host memory.

Output has shape `[tokens * pad_heads, width]`. Only `local_heads` rows per token
are written, starting at output head zero; `head_start` selects the corresponding
input ID columns. These two offsets must not be confused.

## SSD FP8 tables on A5

SSD tables and cache planes preserve the existing DDR lookup layouts:

| Checkpoint codes | Cached codes | Cached scales | Packed row bytes |
| --- | --- | --- | ---: |
| INT8 | INT8[256] | FP32[8] | 288 |
| FP8 E4M3 | FP8 E4M3[256] | E8M0[8] | 264 |

A5 code and scale bytes are copied unchanged from the checkpoint. A3 retains
its existing E8M0-to-FP32 scale expansion. Typed cache planes use the unmodified
DDR UVA wrapper and kernel; no SSD-specific dequantization branch is needed.
A single cache requires the same encoding across its Engram layers.

FP8 artifacts use `engram-fp8-e8m0-aos264-v1` and a checkpoint-specific
directory. Existing INT8 artifacts retain their format and reuse identity.
Completed tables are reused after source identity and integrity checks.
The earlier experimental FP8/FP32 layout is rejected and must be regenerated.
FP8 model initialization requires `ENGRAM_MXFP8` hardware support.

Rebuild the host-side `engram_ssd_c` library when installing this change.
The native reader now accepts the row width when opening a table and copies
eight scale bytes for FP8 or 32 for INT8. The existing INT8 ABI is retained;
an old library reports a rebuild requirement when asked to read FP8.
This change does not modify the NPU UVA kernel.

Packing and reuse tests: `tests/ut/model_loader/test_engram_ssd_fp8.py`.
Real SSD fill/UVA and DDR regression tests:
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_engram_ssd_lookup.py`.

### Starting an A5 SSD service

Use an A5-compatible checkpoint with native FP8 Engram weights. For the
validated DeepSeek-V4.1-Flash-1017 checkpoint, select
`--quantization deepseek_v4_fp8` and set `enable_fused_mc2` to `0`.
The following is the tested eight-NPU pure-Prefill configuration; substitute
your checkpoint and local SSD paths. Merge additional settings when adapting
an existing launch command.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=1024
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export HCCL_ALGO=level0:fullmesh
export HCCL_DETERMINISTIC=true

vllm serve /path/to/DeepSeek-V4.1-Flash-1017 \
    --quantization deepseek_v4_fp8 \
    --tokenizer-mode deepseek_v41 --api-server-count 1 \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 \
    --prefill-context-parallel-size 8 \
    --data-parallel-size 1 --data-parallel-size-local 1 \
    --enable-expert-parallel --max-model-len 139264 \
    --max-num-batched-tokens 8192 --max-num-seqs 32 \
    --block-size 128 --gpu-memory-utilization 0.90 \
    --safetensors-load-strategy lazy --enforce-eager \
    --compilation-config '{"cudagraph_mode":"NONE"}' \
    --no-enable-prefix-caching \
    --additional-config '{
        "enable_cpu_binding": true,
        "enable_fused_mc2": 0,
        "enable_flashcomm1": false,
        "enable_shared_expert_dp": true,
        "multistream_engram_overlap": false,
        "enable_force_eplb": false,
        "engram_ssd": {
            "path": "/local/ssd/engram",
            "host_memory_bytes": 32000000000,
            "threads_per_disk": 16,
            "queue_depth": 128,
            "io_setup_events": 1280,
            "layerwise_overlap": true,
            "prefill_lookahead": true
        }
    }'
```

No manually generated table is needed. The SSD path must have space for the
packed checkpoint (approximately 189 GiB for this checkpoint) and temporary
preparation overhead. `host_memory_bytes` budgets host DDR memory, not HBM.
Normal caching is used unless explicit experiment overrides are supplied.
The communication environment above describes the tested A5 deployment, not
new library defaults.

The added tests check byte preservation, scale decoding, artifact reuse,
rejection of obsolete FP8 manifests, cross-4KiB reads, single/dual-disk
fill and merged reads, and gather output against a PyTorch reference with
exact comparison (including NaNs). The native 264-byte layout passed these
component tests on A5. Full-model startup and 128K-input/1-token-output
Prefill also passed on eight Ascend950DT NPUs with the configuration above.
Three warmed requests had TTFTs of 5.672, 5.416 and 5.456 seconds (median
5.456 seconds); the first warmup took 80.930 seconds and is reported
separately. These measurements used normal caching and independent random
word-sequence inputs, with the profiler configured but inactive.
PD Decode, evaluation-dataset accuracy and full-model A3 regression testing
remain unvalidated for this revision. The INT8 packing bytes and artifact
identity match the prior revision; INT8 reader/UVA component tests also
passed on A5. Earlier FP8/FP32 results do not validate this native format.
