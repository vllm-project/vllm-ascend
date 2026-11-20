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
| NPU E2E：普通路径 | **通过**（T1 同前缀命中、T2 异前缀隔离 + writer 正向对照） |
| NPU E2E：layerwise | **通过**（LW-T1 同前缀 range 加载、LW-T2 异前缀隔离 + writer 正向 range 对照） |
| NPU E2E：cache_prefix=auto | **通过**（两端同身份解析同一 `auto-*` 前缀；跨实例命中） |

> 本轮 NPU E2E 范围按需求限定为：**普通路径、layerwise、`cache_prefix=auto`** 三个场景。

## 2. 环境

| 项 | 值 |
|---|---|
| 源码 | `/root/work/cache-prefix-arm64/vllm-ascend`（HEAD `f940a0874`，工作树干净） |
| vLLM | 源码 `/root/work/cache-prefix-arm64/vllm` HEAD `c410543d`（editable，empty target）；`import vllm` 指向该源码 |
| vllm-ascend | `0.1.dev6820+gf940a0874` |
| Python / torch / torch_npu / transformers | 3.12.16 / 2.10.0+cpu / 2.10.0.post7 / 5.19.0 |
| CANN/driver | 25.5.1 |
| 硬件 | A2 单机 8× Ascend 910B3（E2E 使用 4 卡：A=NPU0,1 / B=NPU2,3） |
| Mooncake | HA master（`mooncake-master-0/1`）+ 3× etcd；测试时 leader 已 `serving` |

说明：vLLM `pyproject` 中 `torch==2.14.0` 属于 **build-system.requires**，用 `--no-deps --no-build-isolation` 跳过；Ascend 实际运行用 torch 2.10 + torch_npu 2.10。

## 3. 类型检查（var-annotated）

定向 mypy（`--follow-imports skip --check-untyped-defs`，`MYPYPATH` 指向 vLLM 源码）：

| 目标版本 | exit | var-annotated | error |
|---|---|---|---|
| 3.12 | 0 | 0 | 0 |
| 3.11 | 0 | 0 | 0 |
| 3.10 | 0 | 0 | 0 |

日志：`/workspace/mypy_pr_3.1{0,1,2}.log`。

## 4. UT 回归（真实依赖，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）

| 组 | 结果 |
|---|---|
| 真实配置 `test_cache_prefix_config.py` | **5 passed（无需 connector 注册 shim）** —— `bb22a21b1` 已使其可直接跑 |
| 回归组 `test_cache_prefix.py, test_mooncake_pipeline.py, test_backend.py, test_layerwise_keys.py, test_metadata.py` | **264 passed** |
| `test_mooncake_layerwise.py, test_mooncake_hybrid.py` | **34 passed** |

注意：`-pr` 分支移除了 `test_cache_prefix_probe.py`、`test_cache_prefix_pair.py` 两个单测文件（相对 `...-12` 分支）。因此 E2E 使用的客户端探针 `tests/e2e/common/kv_pool/cache_prefix_probe.py` 在 `-pr` 上也不存在，本次从上一提交 `7f75916c5` 提取同一脚本到 `/workspace/cache_prefix_probe.py` 使用（纯客户端脚本，语义一致）。

## 5. CI

### 5.1 `bash format.sh ci`
除 gitleaks 外全部 Passed（ruff check/format、codespell、typos、clang-format、markdownlint、actionlint、shellcheck、png、文件名、`__init__`、logger、forbidden-imports、boolean-ctx、long-functions、symbolic-meta、suggestion）。
- gitleaks 钩子：仓库脚本下载的 OBS aarch64 二进制运行即 `fatal error: lfstack.push`（环境/二进制崩溃）。

### 5.2 等价 gitleaks 扫描
源码编译 gitleaks v8.28（Go 1.27.4 + aliyun GOPROXY）：
`gitleaks git --config .gitleaks.toml --redact --log-opts "352168ead..HEAD" .`
结果：**7 commits scanned，~46.56 KB，no leaks found，exit 0**。

## 6. NPU E2E（普通路径 / layerwise / auto）

探针：`cache_prefix_probe.py`（writer 存活成对；bulk 以 `external_hits + ascend_store_load_get_keys` 为加载证据；layerwise 以实时日志 `[KVPOOL_RANGE_DEBUG]` 的 range 字节为加载证据，须等于请求字节）。两端 `kv_both`、`--no-enable-prefix-caching`。

### 6.1 普通路径（use_layerwise:false）
| 用例 | 配置 | 证据 | 结果 |
|---|---|---|---|
| T1 同前缀共享 | 两端 `cache_prefix=model-v1` | writer miss(`hits=0`)；reader `external_hits=1535, bulk_get_keys=24, local_hits=0`，输出一致 | passed |
| T2 异前缀隔离 | A=`model-v1`, B=`model-v2` | writer A `external_hits=1535, bulk_get_keys=24`（正向对照）；reader B `external_hits=0, bulk_get_keys=0`，输出一致 | passed（重跑后） |

产物：`pr-bulk-hit.json`、`pr-bulk-isolation.json`。

### 6.2 layerwise（use_layerwise:true + kv_pool_range_debug:true）
| 用例 | 配置 | 证据 | 结果 |
|---|---|---|---|
| LW-T1 同前缀 range 加载 | 两端 `cache_prefix=model-v1-layerwise` | writer miss；reader `external_hits=1535, local_hits=0, range_load_bytes=150994944`，输出一致 | passed |
| LW-T2 异前缀隔离 | A=`model-v1-layerwise`, B=`model-v2-layerwise` | writer A `external_hits=1535, range_load_bytes=150994944`（正向 range 对照）；reader B `external_hits=0, range_load_bytes=0` | passed（重跑后） |

产物：`pr-lw-hit.json`、`pr-lw-isolation.json`。

### 6.3 cache_prefix=auto
- 两端 `cache_prefix=auto` 且 **同一构建身份** `VLLM_BUILD_COMMIT=cache-prefix-test-c410`。
- 解析结果：A、B（含 scheduler/worker）均为 **同一前缀** `auto-2fd4deada40df53a3fd653babab25baebc52ad87ae9bf643b09e3d98b345da64`。
| 用例 | 证据 | 结果 |
|---|---|---|
| auto 同身份跨实例共享 | writer miss；reader `external_hits=1535, bulk_get_keys=24, local_hits=0`，输出一致 | passed |

产物：`pr-auto-hit.json`。

### 6.4 过程观察
- T2 / LW-T2 首轮 writer 均瞬时 miss（池中对象在 reader 重启的数分钟内被淘汰，原因未定），该次请求即重算并重新写入；随后重跑即在 `--writer-expect hit` 下取得有效正向对照（bulk 用 GET 指标、layerwise 用 range 字节）。与既有“数分钟后不能命中”的观察一致。

## 7. 产物（容器 `/workspace/`）
- 安装/CI：`reinstall_pr.log`、`format_ci_pr.log`
- mypy：`mypy_pr_3.10.log`、`mypy_pr_3.11.log`、`mypy_pr_3.12.log`
- E2E：`pr-bulk-hit.json`、`pr-bulk-isolation.json`、`pr-lw-hit.json`、`pr-lw-isolation.json`、`pr-auto-hit.json`、`cache_prefix_probe.py`（自 `7f75916c5` 提取）
- 脚本：`qwen3_30B_A3B_mix.sh`、`qwen3_30B_A3B_mix2.sh`、`kill_port.py`

复验时状态：A(8195)/B(8196) 运行中、`kv_both`、`cache_prefix=auto`+`VLLM_BUILD_COMMIT=cache-prefix-test-c410`，`/health` 均 200。
