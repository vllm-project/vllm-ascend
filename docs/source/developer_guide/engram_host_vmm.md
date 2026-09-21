# Host-backed VMM Engram on A3

This backend is a follow-up to [#16544](https://github.com/vllm-project/vllm-ascend/pull/16544).
That prerequisite merged as `200309da4198f8c150d4dd365e55e98cb88b8400`.
This follow-up now targets the current main Engram interfaces in `engram/npu.py`.
It integrates VMM storage into the model, checkpoint loader, graph input
preparation, worker shutdown and native build. It is not an example or a
runtime monkey patch. Bounded A3 E2E validation has passed; broader concurrency,
performance acceptance and community CI remain draft gates.

## Enable

Use an A3 build with CANN host VMM/fabric-handle APIs and matching torch-npu.
CMake detects support and builds/installs `libengram_host_vmm.so` with the normal
extension target. Other hardware/older CANN builds retain existing backends;
requesting VMM without the library fails clearly.

Add these fields to the existing serving additional configuration:

```json
{
  "engram_vmm_run": "unique-job-id"
}
```

Use the same **fresh** run ID on every rank. Omit `engram_vmm_run` to retain
the existing HBM or CPU-offload path. VMM and `EngramConfig.cpu_offload` are
mutually exclusive, and sleep mode is rejected. VMM requires a real checkpoint, width 256, group32 FP32
scales, FlashComm1 disabled, and the existing node-local Engram constraints
(EP enabled, PP=PCP=DCP=1). Use the expandable NPU allocator. The run ID contains
1–96 alphanumeric, underscore or hyphen characters; never reuse stale job handles.

Tested runtime for the original implementation: A3, CANN9.1.0, HDK25.5.1.1,
torch-npu2.10.0.post4. Tensor aliases use private torch-npu storage constructors.
The native allocator tries HOST_NUMA (physical-device/2 on the tested A3 topology),
then HOST placement, with a 1 GiB to 2 MiB huge-page fallback. General topology
discovery and compatibility across all runtime releases are not claimed.

## Data path

Startup:

1. The existing checkpoint reader assigns each rank its row shard.
2. Each rank publishes its shard into node-shared host tables using bounded HBM
   staging; tested CANN rejects direct CPU-to-imported-host copies.
3. A node-local barrier completes publication before any lookup.

Inference:

1. Existing CPU history computes IDs and validity masks.
2. IDs are copied to NPU input buffers.
3. A grouped gather/dequant kernel directly reads mapped-host INT8 rows and
   FP32 scales, producing BF16 in the existing graph-stable HBM output buffers.
4. The same kernel clears padding and writes the mask.

There is no intermediate HBM table or gathered INT8 buffer. VMM preparation
bypasses Engram owner-ID AllToAll, row-return AllToAll, reorder, TP broadcast
and the separate final zero/copy pass. Other model communication is unchanged.
Graph capture only obtains stable buffers; runtime preparation refreshes them
before replay. Each rank queries the full local shared table independently.

## Ownership and restart

Host physical allocations and mappings live for the model lifetime, not one
request. One rank owns each descriptor; all local consumers import their own
references. Descriptors under `/dev/shm/engram-vmm-<run-id>` contain capabilities.
Directories must be owned by the current UID and mode0700; files use0600.
PID validation is disabled for trusted same-job peers: this is **not** a
cross-node or multi-tenant memory service.

Worker shutdown quiesces the runner before closing Engram inputs and dropping
table views. Live aliases block unmapping. Release failures retain stable
allocation tokens for retry, including partial virtual-address release; failed
construction retains ownership if rollback fails. Normal process exit also
attempts synchronized cleanup.

Forced termination/SIGKILL can skip Python cleanup and leave stale descriptors.
Use a fresh run ID. Remove stale files only after all old consumers have exited,
including other PID namespaces. There is no automatic stale-job deletion.
Driver process cleanup remains necessary after abrupt exits.

## Tests

CPU loader checks run without NPU hardware:

```bash
PYTHONPATH="$PWD" pytest --confcutdir=tests/ut/models \
  tests/ut/models/test_engram_vmm.py
```

These execute the real inherited checkpoint reader and VMM publication adapter,
replacing only physical mappings with CPU tensors. They cover BF16/INT8 sources,
both index filenames, shard boundaries, exact decoded values, unsupported width
and route rejection. Model input/config tests extend `test_engram.py`
and `test_ascend_config.py` in the standard project test environment.

On dedicated A3 test capacity, after building/installing the package:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python3 tests/e2e/engram_vmm/check_lifecycle.py
python3 tests/e2e/engram_vmm/check_exit.py
run_dir=$(mktemp -d /dev/shm/engram-lookup.XXXXXX)
torchrun --standalone --nproc_per_node=2 tests/e2e/engram_vmm/check_lookup.py \
  --root "$run_dir"
```

The small lookup check uses four rounded GiB of shared host physical memory.
Optional `--full` uses208GiB/node and samples table ends, writer stripes and
pointer-table boundaries. It tests empty/growing/shrinking inputs, padding,
independent CPU reference values and unchanged mapping counters. It is not
exhaustive validation of every row.

Release failure injection is test-only:

```bash
g++ -shared -fPIC -std=c++17 -pthread tests/e2e/engram_vmm/release_faults.cpp \
  -I"${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/include" \
  -ldl -o /tmp/engram-release-faults.so
LD_PRELOAD=/tmp/engram-release-faults.so ENGRAM_TEST_RELEASE_FAILURES=1 \
  python3 tests/e2e/engram_vmm/check_lifecycle.py
```

## Prior measurements and limits

The validated pinned-fork implementation used vLLM-Ascend
`e43cf1e9f5d9bead076853aa6bcacb671465de94` and vLLM
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`. Its source/evidence was frozen
locally at `c31a16c7f3951175c5e807a27e3a6f7b1abd03a7`. Prior hardware checks
covered lifecycle failures/retries, peer lifetime, full-sized sampled lookup
and88stability requests across two model boots; not a long soak.

Existing matched short-request output throughput:32A3, TP8/DP4/EP32,8GiBKV/NPU,
DSpark5, FlashComm1/prefix cache disabled,64requests per case,64output tokens:

| Concurrency | HBM tok/s | CPU-offload tok/s | VMM tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 51.13 | 50.81 | 54.56 |
| 4 | 136.35 | 141.29 | 144.13 |
| 8 | 221.53 | 219.47 | 226.65 |

CPU was the unchanged Engram class/fused ARM NEON operator from PR16544 at
`ecb641ec3a47f74bf5173b88491a2aa65e0aec06`, constructor-only backported into
the same older image; not the entire PR-head framework. HBM/VMM data is reused.
Current main uses a different CPU-offload backend: registered host memory read
directly by the NPU, with the existing owner routing retained. The historical
CPU numbers above do not measure that newer UVA implementation.
Total-chip HBM snapshots were51.05/38.14/37.10GiB/NPU, with inactive VMM mappings
retained in the HBM control. Both host paths remove approximately12.88GiB/NPU
of logical tables. Snapshot differences are not an intrinsic1GiB VMM advantage.

Throughput is roughly preserved, not a proven stable speedup. VMM does not
improve every TTFT metric. Serial token-ID checks matched; concurrent outputs
also varied in baseline repeats. The earlier preparation-chain7x microbenchmark
is not an E2E claim.

## Rebased integration validation (2026-09-21)

The rebased runtime at `302d2a64c` was tested with pinned vLLM
`84030bbe3d74d99bad477a3d2e37a973ccd8865c` on the same A3 image, with newly
built native extensions and ACLNN operators. Real-CANN lifecycle/fault tests,
two-consumer full208GiB sampled lookup, two VMM model boots and one HBM boot
passed. All66HTTP requests completed; both VMM boots matched six serial HBM
responses including complete token IDs. The final production typing fix also
passed two-NPU lookup; subsequent `b946b2ba1` changes only test doubles.

Short matched tests (eight requests per concurrency) measured HBM/VMM
48.18/49.35tok/s at C1 and80.98/72.24tok/s at C4. An earlier VMM boot measured
19.44tok/s at C4; the cause of that variability is unproven. These are bounded
sanity measurements, not established throughput parity. Concurrent token IDs
matched8/8 at C1 and5/8 at C4; this revision lacks a repeated HBM control to
explain the latter. The draft remains open for those acceptance questions.

The real-framework unit suite passed25tests and failed the pre-existing UVA
scale-registration test with `aclrtHostRegisterV2 rc=507899 size=160`, also
reproduced on unmodified main. VMM tests passed; that UVA failure is not a pass.
See the PR description for current CI and evidence boundaries. The older
performance table above remains historical, not a measurement of this revision.
