# Experimental A3 host-backed Engram

This draft contributes standalone host-VMM and fused INT8 lookup primitives.
It does **not** enable a serving backend or patch a model on import. Model
integration is a follow-up to [#16544](https://github.com/vllm-project/vllm-ascend/pull/16544),
which was still open when this example was prepared. Please review the proposed
memory ownership and lookup interface before integrating it into the model.

## Data path

Allocate shared host physical memory once, export a fabric handle, and map it
into each local worker's NPU address space. The table is DRAM-backed, although
the tensor wrapper has an NPU device. During lookup the NPU reads selected INT8
rows and FP32 scales directly, dequantizes them, and writes BF16 into the final
HBM output. There is no intermediate HBM table or gathered INT8 buffer.

The kernel groups 16 rows, writes active results, clears padding and writes the
validity mask in one launch per table. CPU-generated IDs still move to the NPU.
Startup publication uses temporary HBM staging because the tested CANN runtime
rejects direct CPU-to-imported-host copies. No mapping/unmapping occurs per request.

In the separate validated model integration, this replaces Engram owner-ID
AllToAll, owner lookup, row AllToAll, reorder and TP broadcast. Other model
collectives remain. This example alone does not remove serving collectives.

## Requirements and scope

- A3, CANN 9.1.0, HDK 25.5.1.1 and torch-npu 2.10.0.post4 were tested.
- Host VMM/fabric-handle support and sufficient host huge-page memory.
- INT8 width 256 with eight FP32 scales per row (group size 32).
- Trusted same-UID workers on one node sharing a private directory; this is
  not a cross-node or multi-tenant memory service.
- Expandable NPU allocator and the private torch-npu tensor/storage constructors
  used in `mapping.py`. These are compatibility gates, not stable public APIs.
- Tests below require dedicated test capacity; do not run full-capacity tests
  alongside production workloads.

The allocator tries HOST_NUMA placement using physical-device/2 (a tested A3
topology assumption), then HOST placement, with a 1 GiB to 2 MiB huge-page
fallback. General topology discovery is not implemented. This standalone example
does not introduce any serving environment-variable configuration.

## Build and run

From this directory, in an existing A3 vLLM environment:

```bash
bash build.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python3 tests/test_lifecycle.py
python3 tests/test_exit.py
run_dir=$(mktemp -d /dev/shm/engram-lookup.XXXXXX)
torchrun --standalone --nproc_per_node=2 tests/test_lookup.py --root "$run_dir"
```

The small lookup test allocates four rounded GiB in shared host memory. Optional
`--full` uses 208 GiB of shared host physical memory for two real-sized tables.
It samples deterministic rows at table ends, writer stripes and pointer-table
boundaries, with an independent CPU reference; it is not exhaustive table validation.
The tests cover empty/growing/shrinking inputs, output padding, and stable mapping
counters. The shell build only compiles this native translation unit, not all
vLLM-Ascend operators. No compiled library is committed.

Release failure injection is test-only:

```bash
g++ -shared -fPIC -std=c++17 -pthread tests/release_faults.cpp \
  -I"${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/include" \
  -ldl -o tests/release_faults.so
LD_PRELOAD="$PWD/tests/release_faults.so" ENGRAM_TEST_RELEASE_FAILURES=1 \
  python3 tests/test_lifecycle.py
```

## Ownership and failure contract

Use a fresh private directory for every job. Descriptors contain capabilities;
PID validation is disabled for trusted same-job consumers. Directories must be
owned by the current UID with mode 0700; descriptor files use mode 0600. Do not
expose them to other workers or reuse a live/stale path. Owner creation is exclusive.

Quiesce kernel launches and graph replay before closing. Close `Inputs`, drop
all table tensor views, then close mappings. Aliases prevent release, failed
cleanup is retryable, and partially released mappings retain stable allocation
tokens even if their old virtual address is reused. Failed construction with
failed rollback retains ownership for explicit retry or process-exit cleanup.

Normal process exit attempts synchronized cleanup. SIGKILL and forced
multi-process termination can skip Python cleanup and leave stale descriptor
files; driver process cleanup and a fresh job directory are required. This is
not a guarantee of graceful shutdown for every termination path. Remove stale
capability files only after **all** consumers have exited, including consumers
in other PID namespaces. Never delete another job's directory.

## Existing evidence, not a main-branch performance claim

The source before extraction/formatting was tested with vLLM-Ascend
`e43cf1e9f5d9bead076853aa6bcacb671465de94` and vLLM
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. Lifecycle tests included malformed
descriptors, duplicate ownership, live aliases, release failures/retries,
construction rollback, peer lifetime and normal-exit restart. Two full-model
boots completed 88 stability requests; this was not a long soak.

The tested source/evidence was frozen locally at commit
`c31a16c7f3951175c5e807a27e3a6f7b1abd03a7`. Extraction removes an unused
model-specific initialization helper and an optional legacy granularity override;
the tested default allocation/fallback and kernel paths are preserved. This
extracted example has not been rerun on NPU hardware at the community revision.

Separate matched inference measurements used 32 A3 NPUs, TP8/DP4/EP32, fixed
8 GiB KV/NPU, DSpark5, FlashComm1 and prefix caching disabled, 64 short varied
requests per concurrency and 64 generated tokens per request:

| Concurrency | HBM table tok/s | CPU-offload tok/s | Host VMM tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 51.13 | 50.81 | 54.56 |
| 4 | 136.35 | 141.29 | 144.13 |
| 8 | 221.53 | 219.47 | 226.65 |

CPU-offload was the unchanged Engram class and fused ARM NEON CPU operator from
PR #16544 at `ecb641ec3a47f74bf5173b88491a2aa65e0aec06`, backported into the same
older image with constructor-only wiring. It was **not** a full deployment of
that PR's framework. HBM and VMM measurements were reused from earlier runs.

Observed total chip HBM/NPU was 51.05 / 38.14 / 37.10 GiB respectively. These
are separate-start snapshots, not a proven intrinsic difference between the
two host paths; the HBM control retained inactive VMM mappings. Both host paths
save approximately 12.88 GiB/NPU of logical table storage and roughly preserve
throughput. Small measured throughput differences are not stable speedup claims;
VMM did not improve every TTFT metric. Serial token-ID checks matched, while
concurrent outputs varied in both baseline repeats and host-backed runs.

The often-quoted earlier 7x result was a small two-NPU preparation-chain
microbenchmark, **not** this full-model throughput comparison.

## Before production integration

Land/reconcile the DSV4.1 model dependency, use reviewed configuration and build
integration, replace the pinned-fork hooks with model-native ownership, validate
the supported torch-npu/CANN matrix, and rerun correctness on that exact combined
revision. This draft intentionally leaves the installed package, model defaults,
and serving behavior unchanged.
