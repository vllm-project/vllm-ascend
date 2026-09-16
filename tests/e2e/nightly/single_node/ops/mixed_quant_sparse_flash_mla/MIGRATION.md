# MQSMLA original test migration

## Source and scope

Copied from `cann/ops-transformer` commit
`c6240b268a6818ff343c721b8508217d4cfb9812`.
`upstream_manifest.json` records each source path, original SHA-256,
LF-normalized source SHA-256, and migrated file SHA-256.
The copied files retain their original CANN Open Software License Agreement 2.0;
see `LICENSE` in this directory. The newly authored singlecard wrapper has its
own Apache-2.0 header.

The original `mixed_quant_sparse_flash_mla_paramset.py`,
`result_compare_method.py`, `check_valid_param.py`, and `utils.py` retain the
upstream contents; copied text uses LF line endings. The golden input builders
are extended for D448/RoPE0 as described below. The original D512/RoPE64
parameters, independent CPU QK/softmax/PV, and tolerance policy are preserved.
The default parameter set contains **one** `decode_first` case: C8 FP8 E4M3FN,
BF16 query, TND / PA_BBND, B48, S1=2, S2=8192, N1=128, D512, K1024,
tile64, rope64, CSA, with both attention output and LSE checked against the
original CPU golden and original tolerance policy. This is an NPU precision
test, distinct from the copied C++ UT.

## vLLM binding adaptation

The executor follows the A5 single-operator test setup: it calls
`bootstrap_custom_op_env(include_vendor_lib=True)`, imports the actual
`vllm_ascend.vllm_ascend_C` extension, and then invokes
`torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla_metadata` and
`torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla`.
This directly loads the test binding without changing the A5 runtime hardware
profile or its feature switches. The original baseline input semantics, CPU
attention calculation, comparisons and synchronization are preserved; the
D448/RoPE0 input-building extensions are described below. TorchAir imports are delayed until graph
execution. The original graph and Excel batch tools are retained for follow-up
work; their availability is not a claim that graph compilation or the entire
parameter catalog has been validated.

The shared `batch_consistency` package is copied from
`attention/sparse_flash_mla/tests/pytest/batch_consistency`. It is needed at
import time by the original single-case test. Relocated imports point here.
The original README remains in `README.md` for upstream usage details.

## Run the baseline

Build and install this branch's A5 extension and OPP first. In the configured
A5 environment, from the repository root:

```bash
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_mixed_quant_sparse_flash_mla.py::test_mixed_quant_sparse_flash_mla_original_c8
```

The wrapper uses a fresh pytest subprocess with an explicit configuration and
conftest boundary, isolating the upstream terminal patch and generic module
names. It requires exactly one passed case, rejects skips, and stores the
child log, JUnit report and XLSX under pytest's test temporary directory.
Non-A5 machines skip the wrapper; missing A5 bindings fail the actual case.
The visibility of physical cards is controlled by the existing launch
environment; the original test addresses logical device 0.

For direct reproduction of the upstream entry point:

```bash
cd tests/e2e/nightly/single_node/ops/mixed_quant_sparse_flash_mla
bash test_run.sh single
```

The copied scripts keep their upstream `MQSMLA_*` / `SAVE_PT` test controls.
No new product runtime environment variables were introduced. A5 nightly
scheduling must explicitly select the singlecard wrapper; adding this file
does not itself add it to the existing A5 model-only matrix.

## C++ UT and harness

The original 32 C++ cases are under
`csrc/attention/mixed_quant_sparse_flash_mla/tests/ut`: 22 tiling, four
shape/dtype, and six ACLNN cases. The common framework is copied under
`csrc/tests/ut/framework_normal` (common, op_host, op_api, top-level CMake and
empty.cpp). The original Metadata operator has no independent UT directory.

The Host UT fixture is copied from the original sibling SMLA test into
`tests/ut/op_host/support/test_sparse_flash_mla_tiling.h`. Only its include
paths are adapted; it uses the existing SMLA production tiling and arch22
metadata headers without depending on another operator's test directory. API UT requires the copied main
operator's `op_api/aclnn_mixed_quant_sparse_flash_mla.h`.
CMake must enable `OP_HOST_UT` / `OP_API_UT`, the common framework and GTest.
Copying these files is not evidence that the C++ executables have been built
or run; record their actual results separately from NPU precision results.

The baseline migration preceded the separate D448/RoPE0 test extension described below.

## Reproduce the original C++ UT

After sourcing CANN and preparing the same public build dependencies, use an
independent build directory:

```bash
bash tests/e2e/nightly/single_node/ops/mixed_quant_sparse_flash_mla/run_cpp_ut.sh     /home/z00980808/zrr_dev/mqsmla-vllm-validation/ut-build     /home/z00980808/zrr_dev/mqsmla-vllm-validation/deps
```

The script sets the upstream `BUILD_PATH` and the UT library search paths,
requires exactly 36 Host and six API cases to run without skips, and writes
logs plus XML/JSON results beside the UT build directory. GoogleTest must use
`_GLIBCXX_USE_CXX11_ABI=0`, matching the upstream harness. The 104 validation
built GoogleTest 1.14.0 in its isolated dependency prefix.

Three build fixes support this path: load test CMake helpers for normal
`ENABLE_TEST` builds, add GTest download dependencies only when that target
exists, and place API stub libraries in `CMAKE_BINARY_DIR`. The last fix keeps
independent UT and standard kernel builds isolated.

On 104 / zrr_dev, the original C++ cases passed: 26 Host and six API. This
Host/API result is separate from the actual NPU precision baseline.


## D448/RoPE0 test extension

The original `decode_first` D512/RoPE64 precision case, parameters and comparator
are retained. A separate singlecard entry runs ten D448/RoPE0 cases in its own
pytest process. Cases cover quantization modes 1/2, CSA decode, SWA prefill,
ORI_SPARSE K1=61, query/KV tails, BSND/TND query layouts, mode-1 nonpaged KV,
and disabled LSE. Mode 2 is restricted to PA_BBND. The new cases explicitly
check Q/output448, KV480/456 bytes and metadata448/rope0 before comparison.
They use the original independent CPU QK/softmax/PV and its unchanged error
thresholds. Input builders compute physical FP8 feature offsets from the actual
NOPE/RoPE dimensions and safely represent a zero-width BF16 RoPE byte view.

Ten additional Host tests exercise the actual registered tiling implementation,
including rejection of incompatible query/rope/KV widths and mode-2 nonpaged
layouts. These Host tests launch no NPU kernel. Together with the original
32 cases, `run_cpp_ut.sh` now requires 42 passes (36 Host, six API). Keep actual
execution results separate from this description of test coverage.

## Flash Decode runtime coverage limit

The ten RoPE0 precision cases use ordinary execution
(`batch_consistency=False`). They compare the real attention output and, when
enabled, LSE before checking the actual Metadata tensor. Metadata FD-enable
counts are logged for every case and must be zero in this execution mode.
The strict `fd_enabled_cores > 0` assertion remains for a future case explicitly
configured with `batch_consistency=True`; such a case must execute Metadata and
attention under the same supported runtime deterministic level 3 and use the
independent batch-consistency CPU reference.

The upstream Metadata scheduler sets `supportFd_` only for batch consistency.
On 104 / zrr_dev with torch_npu 2.10.0.post4 and CANN 9.1.0, the runtime probe on
2026-09-16 found that `torch_npu.npu.set_deterministic_level(3)` rejects level 3
(the framework accepts only levels 0/1/2). A direct runtime call to
`aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3)` also returned 107000; restoring
level 0 returned success. Consequently this environment cannot enter the FD
path through the real supported interfaces. FD precision is **not validated**.
Changing shape alone cannot enable it. The failed runtime probe and initial
coverage failure are retained as separate evidence; they are not counted as
passed or skipped RoPE0 precision cases. No Metadata values or scheduling
decisions are replaced by the tests.
