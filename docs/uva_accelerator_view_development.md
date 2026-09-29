# UVA accelerator view for MRV2

## Implementation

`csrc/torch_binding.cpp` registers one Host runtime operation in `vllm_ascend_C`. Its name follows vLLM's `get_cuda_view_from_cpu_tensor` convention, with `npu` in place of `cuda`:

- `torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu_tensor) -> Tensor | None`

For a nonempty tensor, the helper calls `aclrtHostGetDevicePointer` once on the pinned CPU storage base, adds `storage_offset * element_size` to the mapped address, and constructs an NPU tensor with the input shape, strides, and dtype. If the allocation cannot be mapped, it returns `None` so the caller can use its copy fallback. The view does not copy data or own the mapped allocation. Its deleter retains the CPU tensor so that its storage remains alive while the view exists. `torch_npu` remains responsible for Host memory registration and unregistration. An empty input produces an empty NPU tensor with the same metadata.

When torch_npu can register mapped pinned memory, the ownership chain is `torch.zeros(..., pin_memory=True)` -> `torch_npu` CachingHostAllocator -> `aclrtHostRegisterV2(..., ACL_HOST_REG_MAPPED)` -> this helper -> `aclrtHostGetDevicePointer` -> NPU-typed mapped view. The helper queries under an `NPUGuard` for the selected device and never registers or unregisters Host memory. The mapped address is for device access; it is not an ordinary NPU allocation for unsupported operations such as generic memory-copy APIs.

This helper is part of the `vllm_ascend_C` Host runtime extension. It is not an AscendC/ACLNN compute operator and does not belong in `CUSTOM_OPS`, `op_host`, or `op_kernel`. The helper checks each allocation at runtime; API availability at compile time does not establish that a particular allocation is mapped.

`vllm_ascend/patch/worker/patch_v2/patch_uva.py` loads `vllm_ascend_C` when real UVA is requested and calls the registered operation directly. `UvaBufferWrapper` uses the mapped view when the operation returns a tensor. When it returns `None`, the wrapper retains the asynchronous Host-to-Device copy path and tracks modified CPU rows. It does not add a forwarding Python module.

## Functionality and input/output requirements

The feature supplies an accelerator-typed view for MRV2 runtime metadata and state buffers backed by pinned CPU storage. It does not provide general weight offload or mapping for arbitrary CPU allocations.

| Interface | Input requirements | Output and failure behavior |
| --- | --- | --- |
| `get_npu_view_from_cpu_tensor` | A defined CPU tensor with strided layout. Views with storage offsets and noncontiguous strides are supported. | Returns an NPU tensor preserving dtype, shape, and strides when the allocation is mapped. A nonempty result aliases the mapped CPU storage and keeps its CPU owner alive. Unpinned or unmapped nonempty input returns `None` for copy fallback; invalid device or layout raises an error. An empty input returns an empty NPU tensor. |
| `UvaBufferWrapper(size, dtype)` | A valid buffer size and PyTorch dtype; the wrapper creates pinned CPU storage. | `cpu` and `np` expose the CPU buffer. `uva(n)` returns the full NPU buffer or its first `n` rows. If the allocation is mapped and UVA is enabled, it returns a view; otherwise it copies modified rows to an NPU buffer before returning it. |

The mapped view does not perform a payload copy or synchronization. Callers must respect the ordering requirements of CPU writes and NPU reads. A successful mapping applies to the current allocation; it is not a device-wide promise that future allocations will be mapped. Unlike vLLM's CUDA implementation, this helper does not allocate replacement pinned storage for an unpinned input; the MRV2 wrapper uses its existing copy fallback.

## Test method

### Build and install

Run in a Docker container with compatible CANN, PyTorch, `torch_npu`, vLLM, vLLM-Ascend, and Triton-Ascend. Install matching vLLM with its target device set to `empty`, then install this repository from its source directory:

```bash
# In the matching vLLM source directory:
VLLM_TARGET_DEVICE=empty \
  python -m pip install -e . --no-build-isolation --no-deps

# In the vLLM-Ascend source directory:
COMPILE_CUSTOM_KERNELS=1 \
  python -m pip install -e . --no-build-isolation --no-deps
```

`--no-deps` preserves the selected Triton-Ascend build. The active environment must already contain build requirements because build isolation is disabled. With `COMPILE_CUSTOM_KERNELS=1`, `setup.py` first executes `build_aclnn -> csrc/build_aclnn.sh -> csrc/build.sh --pkg` and then builds `vllm_ascend_C` with CMake.

For a clean full-build A/B comparison, use separate clean source trees for the base and candidate commits in the same container. Both trees must start without `csrc/build` or top-level `build` artifacts. Use identical environment settings and capture each command's full output and exit status:

```bash
COMPILE_CUSTOM_KERNELS=1 CCACHE_DISABLE=1 \
  python setup.py build_ext --inplace >full-build.log 2>&1
```

Verify extension import and operation registration from the candidate source directory:

```bash
python - <<'PY'
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

assert hasattr(torch.ops._C_ascend, "get_npu_view_from_cpu_tensor")
PY
```

### Component tests

Run the component tests on an Ascend NPU with registered pinned Host memory:

```bash
PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True \
  python -m pytest -vv -rs \
  tests/ut/device/test_uva.py
```

Keep the per-node results and skip reasons from `-vv -rs`. The tests have different execution boundaries:

| Tests | What they verify | Device execution |
| --- | --- | --- |
| `test_npu_view_preserves_metadata`, `test_empty_npu_view`, `test_npu_view_returns_none_for_unpinned_input` | Dtype, shape, strides, device type, fallback result, and invalid device input. | Create NPU tensors; do not read mapped payload on the device. |
| `test_npu_views_keep_cpu_storage_alive` | Multiple views retain their CPU owner until the last view is released. | Host weak references and garbage collection; no device payload read. |
| `test_npu_view_reads_cpu_updates_without_copy`, `test_npu_view_keeps_cpu_storage_alive` | CPU updates and storage lifetime through a mapped view. | Real NPU/Triton `tl.load` only when these nodes are not skipped. The installed Triton-Ascend launcher must accept mapped Host pointers. |
| `test_fallback_copies_modified_prefix_and_sparse_rows`, `test_unmapped_storage_uses_fallback`, `test_pool_fallback_growth_shrink_and_round_robin` | Modified-row copies, unmapped fallback, input types, pool growth, and slot rotation. | Real NPU H2D copies and readback; mapping failure is mocked where needed. |
| `test_real_path_uses_npu_typed_view`, `test_pool_real_path_returns_mapped_view` | Wrapper and pool return an NPU mapped view. | Real view creation with mocked availability; no device payload read. |

The pool pointer test compares two NPU mappings of the same CPU allocation. It does not assume the Host and NPU virtual addresses are equal.

### Integration checks

Run the repository's `tests/e2e/pull_request/one_card/model_runner_v2/test_uva.py` on a compatible NPU environment when its CI skip is lifted. It uses a real `VllmRunner`, Qwen3-0.6B, MRV2, and asynchronous scheduling, so it is an end-to-end test rather than a component UT. Confirm both the mapped path and forced fallback path, repeated requests, and output equivalence. Separately test CPU updates across ACL Graph capture and replay, concurrent buffer use, and performance with a profiler. Record which path each integration run actually exercised.
