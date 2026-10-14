# Sparse HCCL upstream API reuse

The adapter targets the vLLM commit recorded in
`.github/vllm-main-verified.commit` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`)
and the release reference `v0.30.0`. It does not require unpublished packed
transport, dense seed, checksum, or transport-hook extensions.
The sparse engine, checkpoint loader and weight-transfer base files are identical
between these two dependency references.

`SparseWeightPatch` is re-exported directly from vLLM. HCCL initialization and
update dataclasses inherit their sparse NCCL counterparts. The trainer inherits
construction, one-shot and caller-owned chunk lifecycles, and preparation ordering.
Ascend retains its stricter validation, HCCL rendezvous and broadcast implementation,
payload normalization, and NPU completion synchronization.

The worker receives checkpoint patches on the communicator device, enters Ascend
checkpoint loader views, and invokes the upstream checkpoint patch loader. Native
TP/DP/EP mapping and restoration of inference storage remain unchanged. PP=1,
unquantized ND storage and static expert placement are still required.

Tests assert shared type identity and inherited lifecycle methods in addition to
the existing malformed-payload, int64 bounds, rendezvous and failure coverage.
CPU testing with real pinned vLLM modules and mocked accelerator communication
passed 40 cases. This does not establish NPU execution or performance; hardware
regression is required before publishing the adapter update.
The complete weight-transfer and routed-experts UT suite also passed 101 cases
in the isolated Ascend container with the pinned vLLM source. Real HCCL E2E
verification is still pending; these UT results are not a throughput claim.
The real checkpoint E2E test respects the caller's selected physical NPU list,
reserving its first device for the trainer and the next devices for workers.
