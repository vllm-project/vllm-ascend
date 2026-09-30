# GmmDequantSituQuant (A3)

Fused exact A8W4 MSD grouped matmul, dequantization, SiTU, and per-token
INT8 quantization. The compute classes and arithmetic are preserved from the
direct-launch implementation.

## Integration

```text
w4a8.py
  -> torch.ops._C_ascend.gmm_dequant_situ_quant
  -> gmm_dequant_situ_quant_torch_adpt.h
  -> EXEC_NPU_CMD(aclnnGmmDequantSituQuant, ...)
  -> op_host: OpDef / shape inference / tiling
  -> op_kernel/gmm_dequant_situ_quant.cpp
```

The public Torch schema and Meta implementation are unchanged. The adapter
normalizes group lists to contiguous INT64 on the device on every call. Routing
values remain device-resident and are read again on each graph replay.

Weights retain their original storage. The adapter presents the packed INT32
carrier as an ND ACL descriptor at each expert's `data_ptr()` (including its
storage offset). The `weight_nz` attribute selects the existing native NZ Cube
path; it does not request a format conversion. This avoids interpreting the
INT8 NZ storage descriptor as an INT32 NZ allocation. Scale tensors retain the
existing conversion to ND when required.

ACLNN dynamic TensorLists replace the manually cached pointer tables. Tiling
provides the original scalar launch parameters and scratch sizes: eight slots,
272 rows per slot, packed activations followed by INT32 accumulators. The common
ACLNN adapter owns workspace allocation; this operator no longer retains a
global pointer-table or per-stream scratch cache.

## Build and validation

`csrc/build_aclnn.sh` includes this operator in the default `ascend910_93` build.
The operator uses the repository ACLNN package build and has no dedicated build
switch or entry in the top-level direct-kernel target. Other devices do not
compile this A3 kernel. The Torch registration remains unconditional and resolves
the ACLNN symbol at call time, without a direct A3 launcher link dependency.

On an A3 host with the repository's supported CANN and PyTorch environment, run
from the repository root:

```bash
python -m pip install -v --no-build-isolation -e .
pytest -sv tests/e2e/pull_request/one_card/aclgraph/test_gmm_dequant_situ_quant_msd.py
```

The regression suite covers ND/native NZ expert slices, one and multiple experts,
optional linear SiTU, both group-list modes, empty ranks, and dynamic routing in
captured graphs. Compare the same inputs and profiling workload with the direct
implementation when validating numerical and performance equivalence. An A5
build/import check should also verify that no A3 launcher symbol is required.

Source-level equivalence and static checks do not replace a CANN compilation,
NPU execution, or an output/performance comparison between the two builds.
