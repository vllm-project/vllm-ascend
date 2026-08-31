# FillLocalCopy: fused HBM->HBM resident-slot fill kernel

One AscendC kernel launch per layer fills the current forward's fresh-token K/V
rows from the local activations directly into the resident top-k buffers,
replacing a 13-op torch gather/scatter/RMW chain (~6ms/step at TP16, 79 layers)
with a single launch (~0.4ms/step).

## Design

- Kernel arguments are four small int32 descriptor arrays (`rows`, `slots`,
  `valid`, `params`) plus per-layer K/V source/destination base pointers.
- The kernel reads the descriptors from device memory at execution time, so
  under ACL graph replay the same captured launch follows the per-forward
  indices refreshed through pinned->NPU copies (address indirection).
- `valid == 0` entries are strict no-ops (skipped inside the loop), which
  removes the read-modify-write the torch version needed for empty slots.
- Copies go GM -> UB (176KB) -> GM via `AscendC::DataCopyPad`, byte-granular,
  dtype-agnostic (no cast ops).
- 32 AIVs stride-partition up to `maxN` (8) entries; K and V are both filled
  inside the same launch.

## Build

```bash
bash build.sh            # dav-c220 (default); FILL_KERNEL_CCE_ARCH=dav-c310 on A5
```

Requires `bisheng` from CANN >= 8.3.RC1. The resulting `libfill_local_copy.so`
is loaded lazily by the manager from this directory, or from the path in
`VLLM_ASCEND_FILL_KERNEL_SO`.

## Test

`test_fill_kernel.py` is a standalone torch_npu + ctypes test (no vLLM):
eager correctness, in-place descriptor refresh, NPUGraph capture/replay, and
replay-with-new-descriptors (the graph-indirection guarantee).
