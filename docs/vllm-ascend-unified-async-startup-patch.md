# vLLM-Ascend：统一异步并行进程启动（TP Worker + DP EngineCore）

本文档描述 **仅在 vllm-ascend 仓库** 通过平台 patch 对齐上游 vLLM 行为的方式，语义参考 commit  
`10189846215751d9c4eb1b8b94e86e9d2940f877`。  
与工作区其它独立存档（例如 `cursor_git/docs/vllm-async-tp-worker-startup-scheme.md`）**不同名、不合并**，本片为 ascend 侧的落地说明。

---

## 1. 目标范围

| 维度 | upstream 映射 | ascend 承载 |
|------|----------------|-------------|
| 本机多 **worker**（TP/PP 等 rank）并行触发 `Process.start` | `vllm/v1/executor/multiproc_executor.py` | [`patch_multiproc_executor.py`](../vllm_ascend/patch/platform/patch_multiproc_executor.py)，由 [`platform/__init__.py`](../vllm_ascend/patch/platform/__init__.py) 常驻加载 |
| 本机多 **EngineCore**（典型为 DP）并行触发 `proc.start` | `vllm/v1/engine/utils.py` `CoreEngineProcManager` | [`patch_engine_core_parallel_startup.py`](../vllm_ascend/patch/platform/patch_engine_core_parallel_startup.py)，无能力时回填类 |

两类优化均作用于 **引擎父进程侧的进程创建调度**，不改变推理阶段的 TP 集合通信语义。

---

## 2. 加载路径（与原 EPLB 门禁的差异）

### 原先

[`patch/platform/__init__.py`](../vllm_ascend/patch/platform/__init__.py) 仅在 `DYNAMIC_EPLB` 或 `EXPERT_MAP_RECORD` 为真时 `import patch_multiproc_executor`，多数场景 **不会** 替换 `MultiprocExecutor`，异步 worker 起动与昇腾侧 `daemon=False` 均未生效。

### 现在

在 [`platform/__init__.py`](../vllm_ascend/patch/platform/__init__.py) 末尾 **无条件**、**固定顺序** 导入：

1. `patch_multiproc_executor` — Monkey-patch `MultiprocExecutor`（TP 等 worker）。  
2. `patch_engine_core_parallel_startup` — 若上游无 `_run_async_startup`，则替换 `CoreEngineProcManager`（DP EngineCore）。

**导入顺序**有意为「multiproc 先于 engine.utils 回填」：`patch_multiproc_executor` 不依赖 `vllm.v1.engine.utils`；DP patch 在导入时才会 `import vllm.v1.engine.utils` 并可能替换类。

---

## 3. TP：`AscendMultiprocExecutor`

- **条件**（与上游一致）：`spawn`（非 fork）、`not current_platform.is_cpu()`、`local_world_size > 1` 时，`asyncio` + `to_thread(AscendWorkerProc.make_worker_process, …)` 并行起动。  
- **Fork**：仍顺序维护 `inherited_fds`。  
- **昇腾差异**：MessageQueue `connect_ip` 使用 `master_addr`；worker 进程 **`daemon=False`**（NPU 上统一非 daemon，与 EPLB 再起子进程场景兼容）。  

在 **NPU 平台** 上，`NPUPlatform.pre_register_and_update()` 会调用 `adapt_patch(is_global_patch=True)`，从而加载整个 `patch/platform` 包，上述逻辑 **与设备类型无关地** 在 ascend 启用时生效（不依赖 CUDA）。

---

## 4. DP：`CoreEngineProcManager` 回填策略

- 若 `hasattr(CoreEngineProcManager, "_run_async_startup")`：**无操作**（`logger.debug`）。当前常见的新版 vLLM 走此分支，**不增加运行时代价**。  
- 否则：将 `engine_utils.CoreEngineProcManager` 替换为 `_AscendCoreEngineProcManagerBackport`，并打出 **warning**，提示应升级 vLLM 以去掉 shim。  
- 阈值与上游一致：`local_engine_count > 1` 时并行 `proc.start`（`_ASYNC_STARTUP_ENGINE_THRESHOLD = 1`）。  
- **NPU / 非 CUDA**：`need_env_control` 为真时使用 `_enginecore_bootstrap` + `get_device_indices`，与上游对非 CUDA DP 的语义一致。

### 已知局限（非 bug）

- **早期 import 绑定**：若某模块在平台 patch 执行 **之前** 已执行 `from vllm.v1.engine.utils import CoreEngineProcManager` 并缓存了 **旧类** 引用，回填不会更新该缓存。vLLM 正常启动路径下 `launch_core_engines` 等在 `adapt_patch` 之后，一般无此问题。  
- **检测启发式**：以类上是否存在 `_run_async_startup` 判断「上游已具备并行 EngineCore 起动」；若未来上游改名或拆分，需同步调整检测逻辑。

---

## 5. Commit `b078d3f81f0f16776caab49fbe03adac15b554be` 审查结论（功能完备性）

| 检查项 | 结论 |
|--------|------|
| TP 并行起动条件与上游 1018984 对齐 | 是（spawn、非 CPU OMP、`local_world_size > 1`；fork 串行） |
| DP 并行起动在新 vLLM 上 | 跳过 shim，使用上游实现 |
| DP 并行起动在旧 vLLM 上 | backport 与 `utils.py` 中进程表构建、NUMA、`asyncio` 路径一致 |
| NPU 上是否加载 patch | 是，经 `adapt_patch(is_global_patch=True)` 与 `platform/__init__.py` |
| 与 EPLB 门禁解耦 | 是；不再依赖 `DYNAMIC_EPLB` 才加载 multiproc patch |

**进一步精简 patch**：已去掉仅做 re-export 的 `patch_parallel_async_startup.py`，由 `platform/__init__.py` 直接串联两个模块，减少一层文件与维护面。无法再合并 `patch_engine_core_parallel_startup` 与 `patch_multiproc_executor` 而不混淆职责；二者分别对应 `engine.utils` 与 `multiproc_executor` 两个上游模块。

---

## 6. 验证建议

- 日志（TP / worker）：`Using async parallel startup for %d worker processes (Ascend).`  
- 日志（DP / EngineCore，回填路径）：`Using async parallel startup for %d EngineCore processes (Ascend patch backport).`  
- 配置：`tensor_parallel_size` 或本机 `data_parallel_size_local` 等使 **本机子进程数 > 1**；multiprocessing 为 **spawn**。  

---

## 7. 维护清单

| 文件 | 说明 |
|------|------|
| [`vllm_ascend/patch/platform/__init__.py`](../vllm_ascend/patch/platform/__init__.py) | 常驻顺序加载 `patch_multiproc_executor`、`patch_engine_core_parallel_startup` |
| [`patch_multiproc_executor.py`](../vllm_ascend/patch/platform/patch_multiproc_executor.py) | TP / MultiprocExecutor |
| [`patch_engine_core_parallel_startup.py`](../vllm_ascend/patch/platform/patch_engine_core_parallel_startup.py) | DP / CoreEngineProcManager 可选回填 |
| [`vllm_ascend/patch/__init__.py`](../vllm_ascend/patch/__init__.py) | 平台 patch 目录说明（条目 3） |

升级 **pin 的 vLLM** 时：若 `CoreEngineProcManager` 已合并异步起动，可逐步删除 backport 类；若 `MultiprocExecutor` 上游行为与 ascend 需求一致，可收缩 `patch_multiproc_executor`。

---

## 8. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-13 | 初版：统一常驻加载、DP 条件回填、文档与 patch 目录同步 |
| 2026-05-14 | 审查：去掉薄封装 `patch_parallel_async_startup.py`，由 `platform/__init__.py` 直接 import；补充审查结论与局限 |
