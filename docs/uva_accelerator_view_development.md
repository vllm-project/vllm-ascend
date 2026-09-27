# UVA accelerator view: 方案、编译与测试

> 状态：implementation candidate / host-side validation complete。本文描述 MRV2 pinned CPU buffer 的 NPU view 实现和复现方法；Triton direct-read、MRV2 E2E、ACL Graph、并发与性能 Gate 尚未通过。

## 范围与语义

上游 `UvaBuffer` 需要一个与 pinned CPU Tensor 共享存储的 accelerator-typed view。本实现仅处理 MRV2 runtime metadata/state buffer，不覆盖权重 UVA、任意 NPU 算子或大型表的 offload。

对非空输入，helper 要求 CPU、strided、pinned Tensor。`aclrtHostGetDevicePointer` 查询已注册 storage base 的设备映射地址；将 `storage_offset * element_size` 加到该地址后，以原 shape、stride、dtype 创建 NPU Tensor view。view 的 deleter 捕获原 CPU Tensor，使 storage 在 view 存活期间不被释放；注册和注销仍由 `torch_npu` 的 Host allocator 负责。helper 不复制 payload，也不主动同步。空 Tensor 返回保留元数据的 NPU 空 Tensor。

| 层 | 实现 | 职责 |
| --- | --- | --- |
| Host runtime | `csrc/torch_binding.cpp` | 校验输入、查询 mapping、创建非 owning view、注册 `_C_ascend` op |
| Python API | `vllm_ascend/device/uva.py` | 加载扩展并调用两个 `_C_ascend` op |
| MRV2 wrapper | `vllm_ascend/patch/worker/patch_v2/patch_uva.py` | 仅在 real UVA 已请求且当前 allocation 可映射时选用 view；否则保持异步 H2D fallback |

这是 `vllm_ascend_C` 的 Host runtime utility，不是 AscendC/ACLNN compute op；不要放入 `CUSTOM_OPS`、`op_host` 或 `op_kernel`。当前源码的 CANN 基线为 9.1.0，不需要 `aclrtHostGetDevicePointer` 的 CMake API existence probe。API 可编译不等于任意 pinned allocation 均可映射；以每次 runtime 查询的返回值为准。

## 构建分层

从干净、隔离的源码树运行完整源码构建：

```bash
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=32  # 按机器资源调整；A/B 比较时两侧保持一致
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export CCACHE_DISABLE=1
python3 setup.py build_ext --inplace
```

`setup.py` 先运行 `build_aclnn` → `csrc/build_aclnn.sh` → `csrc/build.sh --pkg`，构建并安装 ACLNN/AscendC custom ops；之后才由 CMake 构建 `vllm_ascend_C`。`build_aclnn.sh` 保留 `csrc/build`，所以排查构建回归时应为 baseline 和候选各准备一份**没有 `csrc/build` 和顶层 `build` 目录**的源码树，在同一容器和同一环境下顺序执行完整命令，分别保留 stdout/stderr 与退出码。增量构建的 linker 错误不能直接归因于环境。

需要单独定位扩展时，可以在另一份干净源码树中运行 targeted CMake build。以下是 CANN 9.1.0 / A3 示例；其他设备应替换对应路径与 `SOC_VERSION`：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export SOC_VERSION=ascend910_9391
src="$PWD"
build_dir="$src/build-uva-targeted"
pybind_path="$(python3 -m pybind11 --cmakedir)"
python_include="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
torch_npu_path="$(python3 -c 'import os, torch_npu; print(os.path.dirname(torch_npu.__file__))')"

cmake -S "$src" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DPYTHON_INCLUDE_PATH="$python_include" \
  -DCMAKE_INSTALL_PREFIX="$src/vllm_ascend" \
  -DCMAKE_PREFIX_PATH="$pybind_path" \
  -DSOC_VERSION="$SOC_VERSION" \
  -DFETCHCONTENT_BASE_DIR="$src/.deps" \
  -DTORCH_NPU_PATH="$torch_npu_path"
cmake --build "$build_dir" --target vllm_ascend_C -j "${MAX_JOBS:-4}"
cmake --install "$build_dir"
```

`vllm_ascend_C` 链接的 `vllm_ascend_kernels` 仍会作为 CMake 依赖构建；这个 targeted 命令不执行 `build_aclnn.sh --pkg`，不能替代上面的完整源码构建。构建后从当前源码树确认 import 与注册：

```bash
PYTHONPATH="$PWD:${PYTHONPATH:-}" python3 - <<'PY'
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

assert hasattr(torch.ops._C_ascend, "can_get_npu_view_from_cpu_tensor")
assert hasattr(torch.ops._C_ascend, "get_npu_view_from_cpu_tensor")
PY
```

## Component UT 方法与执行边界

先让 vLLM 源码/安装版本与 vLLM-Ascend 对齐，再在有 Ascend NPU 的环境中运行：

```bash
PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True \
  python3 -m pytest -vv -rs \
  tests/ut/device/test_uva_view.py \
  tests/ut/device/test_uva_wrapper.py
```

使用 `-vv -rs` 保存每个 node id、结果和 skip reason；单独记录测试是否真的执行 NPU/Triton：

| 用例组 | 检查内容 | 实际边界 |
| --- | --- | --- |
| `test_npu_view_preserves_metadata`、`test_empty_npu_view`、`test_npu_view_rejects_unpinned_or_device_input` | dtype、shape、stride、NPU 类型及无效输入 | 创建 NPU view/输入；没有 mapped view 的设备 payload read |
| `test_npu_views_keep_cpu_storage_alive` | CPU owner 与两个 view 的 host 引用生命周期 | weakref/GC；没有设备读取 |
| `test_npu_view_reads_cpu_updates_without_copy`、`test_npu_view_keeps_cpu_storage_alive` | CPU 更新可见性及 owner 释放后的 Triton direct-read | 仅在未 skip 时执行 Triton 读取；当前 Triton-Ascend 3.2.1/3.2.2 按已知 launcher 问题 skip |
| `test_fallback_copies_modified_prefix_and_sparse_rows`、`test_unmapped_storage_uses_fallback`、`test_pool_fallback_growth_shrink_and_round_robin` | fallback、扩缩、2/3 slot 轮转 | capability/availability 部分使用 mock，但执行真实 NPU H2D 与 readback |
| `test_real_path_uses_npu_typed_view`、`test_pool_real_path_returns_mapped_view` | wrapper/pool real-path 的类型与 mapped pointer | availability 使用 mock；创建真实 view，但没有设备读取 |

即使 component UT 全部可运行，仍需单独通过 Triton direct-read、MRV2 E2E、ACL Graph replay、并发压力和 profiler。仓库已有的 `tests/e2e/pull_request/one_card/model_runner_v2/test_uva.py` 是 `VllmRunner + Qwen/Qwen3-0.6B + MRV2 + async_scheduling` 的真实 E2E，不是 UT。当前 main 的 `.github/workflows/scripts/test_config.yaml` 因 CANN/Triton 升级暂时跳过整个 E2E 文件；恢复后需单独运行并报告结果。

## 当前 A3 证据与剩余 Gate

在 CANN 9.1.0、torch_npu 2.10.0.post4、Triton-Ascend 3.2.2 的同一 A3 容器中，base `75c3ff9d` 和候选 `71ea84e6` 的 clean full source build 都通过；候选独立 clean targeted build、extension import 和 `_C_ascend` symbol 注册也通过。20 个 component node 为 16 passed、4 skipped。单独调用 int32 direct-read 测试体时，Triton launcher 在 kernel 执行前报 `ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)`。

因此 G1/G2/G4 只有部分证据，G3 未通过，G5 仅 A3 构建层通过；MRV2 E2E、ACL Graph、并发与 profiler 尚未验证。不要将当前实现标记为已验证 UVA feature，也不要把 fallback 的 NPU H2D 测试解释为 mapped Host direct-read 通过。
