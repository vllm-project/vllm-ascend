# Mooncake cache_prefix：`codex/ascend-store-cache-prefix-pr` 验证报告

- 报告日期：2026-11-20
- 分支/HEAD：`codex/ascend-store-cache-prefix-pr`，`f940a0874c581b9aaa88f0ec66351fe345a8e873`
- 功能起点（首个功能提交 `04432dd3a` 的父）：`352168ead`
- 配对 vLLM：`c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`（本环境已把 vLLM 源码切到该提交）
- 安装版本：`vllm_ascend 0.1.dev6820+gf940a0874`（editable，构建后已清理旧 dist-info / 旧 egg-info）

## 1. 结论摘要

| 项 | 结果 |
|---|---|
| 代码更新与重装 | 已切到 `f940a0874`；vLLM 切到配对 `c410543d`；重装 `vllm_ascend 0.1.dev6820+gf940a0874` |
| 三处 var-annotated | 已存在（`metadata.py:57`、`test_cache_prefix_config.py:84`、`test_cache_prefix.py:297`）；定向 mypy 3.12/3.11/3.10 均 0 error、0 var-annotated |
| UT | 真实配置 **5 passed（无需 shim）**；回归组 **264 passed**；layerwise/hybrid **34 passed** |
| `format.sh ci` | 除 gitleaks 钩子外全部 Passed；等价 gitleaks 范围扫描 `352168ead..HEAD`（7 commits）**no leaks** |
| E2E（普通/layerwise） | **被 Mooncake 集群故障阻塞**（master HA/oplog 未选出主），与 `-pr` 代码无关，见第 6 节 |
| A/B/C 对比（native/Mooncake/UCM） | 方法已记录（见第 7 节）；UCM 未安装、Mooncake 当前不可用，尚未执行 |

## 2. 环境

| 项 | 值 |
|---|---|
| 源码 | `/root/work/cache-prefix-arm64/vllm-ascend`（HEAD `f940a0874`，工作树干净） |
| vLLM | 源码 `/root/work/cache-prefix-arm64/vllm` HEAD `c410543d`（editable，empty target）；`import vllm` 指向该源码 |
| vllm-ascend | `0.1.dev6820+gf940a0874` |
| Python / torch / torch_npu / transformers | 3.12.16 / 2.10.0+cpu / 2.10.0.post7 / 5.19.0 |
| CANN/driver | 25.5.1 |
| 硬件 | A2 单机 8× Ascend 910B3 |
| Mooncake | HA master（`mooncake-master-0/1`）+ 3× etcd；当前故障，见第 6 节 |

说明：vLLM `pyproject` 中 `torch==2.14.0` 属于 **build-system.requires**，用 `--no-deps --no-build-isolation` 跳过；Ascend 实际运行用 torch 2.10 + torch_npu 2.10。

## 3. 类型检查（var-annotated）

修复仍在 `bfe8c9e81` 之后的版本中；定向 mypy（`--follow-imports skip --check-untyped-defs`，`MYPYPATH` 指向 vLLM 源码）：

| 目标版本 | exit | var-annotated | error |
|---|---|---|---|
| 3.12 | 0 | 0 | 0 |
| 3.11 | 0 | 0 | 0 |
| 3.10 | 0 | 0 | 0 |

日志：`/workspace/mypy_pr_3.1{0,1,2}.log`。

## 4. UT 回归（真实依赖，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）

| 组 | 结果 |
|---|---|
| 真实配置 `test_cache_prefix_config.py` | **5 passed（无需 connector 注册 shim）** —— `bb22a21b1 test(kvpool): make cache prefix config tests portable across CI versions` 已使其可直接跑 |
| 回归组 `test_cache_prefix.py, test_mooncake_pipeline.py, test_backend.py, test_layerwise_keys.py, test_metadata.py` | **264 passed** |
| `test_mooncake_layerwise.py, test_mooncake_hybrid.py` | **34 passed** |

注意：`-pr` 分支移除了 `test_cache_prefix_probe.py`、`test_cache_prefix_pair.py` 两个单测文件（相对 `...-12` 分支），故回归命令按 `-pr` 实际布局调整；探针脚本 `tests/e2e/common/kv_pool/cache_prefix_probe.py` 仍在。

## 5. CI

### 5.1 `bash format.sh ci`
除 gitleaks 外全部 Passed（ruff check/format、codespell、typos、clang-format、markdownlint、actionlint、shellcheck、png、文件名、`__init__`、logger、forbidden-imports、boolean-ctx、long-functions、symbolic-meta、suggestion）。
- gitleaks 钩子：仓库脚本下载的 OBS aarch64 二进制运行即 `fatal error: lfstack.push`（环境/二进制崩溃）。

### 5.2 等价 gitleaks 扫描
用源码编译的 gitleaks v8.28（Go 1.27.4 + aliyun GOPROXY）：
`gitleaks git --config .gitleaks.toml --redact --log-opts "352168ead..HEAD" .`
结果：**7 commits scanned，~46.56 KB，no leaks found，exit 0**。

## 6. E2E 阻塞：Mooncake master HA/oplog 故障（与代码无关）

启动普通路径（Qwen 8195/8196）时引擎初始化失败，根因：

```
etcd_helper.cpp: key=mooncake-store/mooncake-ha-prod/master_view, error=key not found in etcd
mooncake_backend.py: Initialize mooncake failed. ret=-600, metadata_server=P2PHANDSHAKE
RuntimeError: Engine core initialization failed ... Failed core proc(s): {}
```

`mooncake-master-0/1` 均在反复报：

```
OpLogApplier: recovered from sequence_id=0, expected_sequence_id set to=1
standby_state_machine: SYNCING -> FAILED (event: FATAL_ERROR)
Failed to promote standby for serve: INCOMPLETE_OPLOG_CATCH_UP, retrying in 1s
```

即两个 master 都无法完成 oplog 追平、无法选出主，故 etcd 无 `master_view`，客户端无法初始化 Mooncake store。已尝试 `kubectl rollout restart statefulset`（StatefulSet 为 OnDelete，未真正重建）与**删除 master pod 触发重建**，重建后仍 `INCOMPLETE_OPLOG_CATCH_UP`——属 oplog 基线/快照状态问题，非进程问题。

影响：普通路径 T1/T2 与 layerwise LW-T1/LW-T2 无法在本轮完成（需要可用的 Mooncake store 才能产生外部命中与 range 加载证据）。

恢复选项（需确认后执行，含破坏性）：
1. 清空该 cluster 的 etcd oplog/快照状态（如 `mooncake-store/mooncake-ha-prod` 相关 key）并清理 `/data1/mooncake_snapshots` 后重启 master，建立全新空集群（会丢弃已有缓存对象/集群状态）；
2. 由你修复/回滚 master 的 oplog 相关部署（本次更改前状态）后我再复跑；
3. 若允许，我也可以禁用 oplog 相关开关做一次临时验证（非生产语义）。

## 7. A/B/C 对比方法（native / Mooncake / UCM）— 待执行

按你的口径：**2k、8k 两种长度各构造 100 条用例；先跑一轮冷填充（A），再分别测试 offload 到 CPU（B）与 offload 到 SSD（C）**；对 native、Mooncake、UCM 三种 offloading 各跑同一 workload 做横向对比。

- native：vLLM-Ascend 内置 `OffloadingConnector`（`CPUOffloadingSpec`/`TieringOffloadingSpec`，见 `kv_cache_cpu_offload.md`）。其 A/B/C 基线脚本/结果不在本容器（需你提供或指定路径以 1:1 复刻）。
- Mooncake：`AscendStoreConnector` + Mooncake store。**当前 Mooncake store 不可用（第 6 节）**，需先恢复。
- UCM：外部 KV 存储（`ucm_deployment.md`）。仓库有 `ucm_connector` 代码，但**本环境未安装 UCM 库，集群无 UCM 服务/Store**；需提供 UCM 安装产物与后端（本地 SSD/NFS/3FS）后才能跑。

需要的输入：native 的 A/B/C 脚本与结果模板（模型、prompt 集、并发、`max_model_len`、指标口径）；UCM 的安装与存储后端；以及 Mooncake 集群恢复。

## 8. 产物（容器 `/workspace/`）
- `reinstall_pr.log`、`format_ci_pr.log`
- `mypy_pr_3.10.log`、`mypy_pr_3.11.log`、`mypy_pr_3.12.log`
- 说明：vLLM 源码已切到 `c410543d`；vllm-ascend 为 `0.1.dev6820+gf940a0874`；工作树代码干净。
