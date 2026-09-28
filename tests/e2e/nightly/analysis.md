## 一、摘要

针对将 AISBench Agent 数据集测试集成到 vllm-ascend社区nightly自动化框架的可行性进行了系统分析。通过对当前 nightly 框架架构、AISBench Agent 测评链路、CI 基础设施能力的深入调研，识别出五大核心难点。经评估，**五大难点均可解决或在合理工程取舍下规避**，项目整体可行性为中高。

---

## 二、背景与目标

### 2.1 项目背景

vllm-ascend 是 vLLM 社区官方维护的昇腾 NPU 硬件插件，其 nightly 自动化框架位于 `tests/e2e/nightly/`，采用"YAML 驱动 + Dispatcher 分发"架构，当前支持三种测试类型：

| case_type | 用途 | 判定逻辑 |
|-----------|------|----------|
| `accuracy` | 精度测试 | `abs(result - baseline) <= threshold` |
| `performance` | 性能测试 | `throughput >= baseline * threshold` |
| `spec_decode` | 投机解码接受率 | `rate >= golden * (1 - tolerance)` |

AISBench 已原生集成 Harbor 框架作为 Agent 测评引擎（`--mode agent`），提供 6 个 Agent 数据集、40+ Agent 适配器，是一条成熟的 Agent 评测链路。但该链路与 vllm-ascend nightly 框架完全独立，尚未集成。

### 2.2 目标

将 AISBench Agent 数据集测试纳入 vllm-ascend nightly YAML 驱动框架，实现：

1. 在 YAML 配置中声明 `case_type: agent` 的 benchmark
2. 框架自动拉起 vLLM 服务 + Agent runtime 容器 + 执行评测
3. 结果接入现有 postprocess / openlibing 上报链路
4. 支持 nightly / weekly 两种执行频率

### 2.3 价值

- **补齐 Agent 评测能力**：当前 nightly 仅覆盖精度和性能，缺乏 Agent 能力评估
- **及早发现回归**：vLLM 版本迭代可能影响 Agent 调用质量（工具调用、代码生成等）
- **对标行业实践**：SWE-Bench 已成为 LLM Agent 能力事实标准，集成后可对标行业

---

## 三、现状分析

### 3.1 当前 nightly 框架执行链路

```
pytest 启动
  → SingleNodeConfigLoader.from_yaml_cases() 加载 YAML
  → pytest parametrize 逐 case 执行
  → 启动 vLLM 服务（openai 或 epd 模式）
  → 分发 test_content 阶段（completion / chat_completion / image 等）
  → 运行 benchmarks（调用 aisbench 原生推理链路）
  → _save_benchmark_results_json() 落盘
  → postprocess_benchmark_results() 上报 openlibing
  → 关闭服务
```

核心文件：
- `tests/e2e/nightly/single_node/models/scripts/single_node_config.py` — 配置解析
- `tests/e2e/nightly/single_node/models/scripts/test_single_node.py` — 测试分发与执行
- `tests/e2e/nightly/scripts/result_postprocess.py` — 结果后处理

### 3.2 AISBench Agent 测评链路

```
准备 Harbor 格式数据集 + case 镜像 + Agent 依赖
  → 启动 agent-runtime 容器（start_agent_runtime.sh）
  → ais_bench --mode agent -a <agent> --api-base <vllm_url> -p <dataset_path>
  → Harbor 为每个 case 拉起 Docker 容器
  → Agent 在容器内调用 vLLM API 解决任务
  → Verifier 验证结果，产出 reward.json / ctrf.json
  → HarborSummarizer 汇总 avg_score / correct / wrong / exception
```

### 3.3 CI 基础设施现状

| Runner | 架构 | 硬件 | 用途 |
|--------|------|------|------|
| `linux-aarch64-a2b3-*` | aarch64 | A2 NPU | A2 测试 |
| `linux-aarch64-a3-800t-0` | aarch64 | A3 NPU | A3 测试 |
| `linux-amd64-cpu-*` | x86_64 | 纯 CPU | 镜像构建、控制面 |

CI 容器特点：
- nightly 测试镜像基于 `Dockerfile.nightly.a3` 构建，内含 aisbench 工具
- CI Job 容器内**只有一层 Docker**，无嵌套 Docker
- 现有 K8s 集群基础设施（`KUBECONFIG_B64`）可用于多节点部署

---

## 四、五大难点及可行性分析

### 难点 1：调用接口不兼容

**现状**：  
当前框架通过 `from tools.aisbench import run_aisbench_cases` 调用 aisbench 原生推理链路。Agent 测评走 `--mode agent` 独立链路，入口完全不同，无法通过现有 `run_aisbench_cases()` 调用。

**影响**：中  
**可行性**：✅ 可解决

**解决方案**：
1. 在 `test_single_node.py` 的 `_run_benchmarks()` 中新增 `case_type == "agent"` 分支
2. 新增 `_run_agent_benchmarks()` 函数，通过 subprocess 调用 `ais_bench --mode agent`，或直接 import aisbench 的 `HarborRunner` / `HarborAgentTask` API
3. 构建 `harbor_agent_task.py` 配置文件模板，支持从 YAML 字段自动生成

**技术评估**：aisbench 是 Python 包，已预装在 nightly 镜像中。`HarborRunner` 和 `HarborAgentTask` 可直接 import 调用，无需 subprocess 开销。

---

### 难点 2：Docker 容器依赖与生命周期管理

**现状**：  
当前 nightly 框架只管理 vLLM 子进程（`RemoteOpenAIServer` / `RemoteEPDServer`），不涉及 Docker 容器编排。Agent 测评需要：
- agent-runtime 容器（运行 Harbor + Agent 框架）
- 每个 case 的 Docker 镜像（任务执行环境）
- Docker-in-Docker 或 Docker socket 挂载

**影响**：高  
**可行性**：✅ 可解决

**解决方案**：
1. **CI workflow 层**：在 `_e2e_nightly_single_node.yaml` 中为 agent 测试 job 增加 Docker socket 挂载：
   ```yaml
   container:
     image: ${{ inputs.image }}
     volumes:
       - /var/run/docker.sock:/var/run/docker.sock
     options: --privileged
   ```
2. **Dockerfile 层**：在 `Dockerfile.nightly.a3` 中增加 Docker CLI 安装：
   ```dockerfile
   RUN apt-get update && apt-get install -y docker.io
   ```
3. **框架层**：新增 `AgentRuntimeContainer` context manager，封装 `start_agent_runtime.sh` 的调用与生命周期管理：
   ```python
   with RemoteOpenAIServer(...) as server:
       if has_agent_benchmarks:
           with AgentRuntimeContainer(
               image=config.agent_runtime_image,
               vllm_url=server.url,
               ...
           ) as agent_env:
               _run_agent_benchmarks(config, agent_env)
   ```

**技术评估**：AISBench 的 `start_agent_runtime.sh` 已封装 socket 模式和 dind 模式。CI runner 是否支持 Docker socket 挂载需与运维团队确认，但从现有 K8s 基础设施判断，可行性较高。

---

### 难点 3：架构兼容性限制（核心难点）

**现状**：  
Ascend NPU 芯片为 aarch64 架构，所有 NPU runner 均为 aarch64。而 AISBench 提供的 6 个 Agent 数据集中，**仅 2 个支持 aarch64**：

| 数据集 | x86_64 | aarch64 | 外网依赖 |
|--------|--------|---------|---------|
| SWEBench Verified | ✅ | ❌ | 无 |
| SWEBench Multilingual | ✅ | ❌ | 无 |
| SWEBench Pro | ✅ | ❌ | 无 |
| terminal-bench 2.0 | ✅ | ✅ | **用于校验结果; agent 测评环境，如果是在内网，需要代理和证书** |
| terminal-bench 2.1 | ✅ | ✅ | **用于校验结果; agent 测评环境，如果是在内网，需要代理和证书** |
| DeepSWE | ✅ | ❌ | 无 |

**PS: 
docker 版本 >= 20.10.0
有足够的存储空间
**

**影响**：高  
**可行性**：⚠️ 部分可解决（需分阶段策略）

**解决方案矩阵**：

| 方案 | 原理 | 可行性 | 数据集覆盖 |
|------|------|--------|-----------|
| A. terminal-bench only | aarch64 原生运行 | ✅ 高 | 33% |
| B. QEMU 跨架构模拟 | aarch64 上模拟 x86_64 | ❌ 不可行 | 100% |
| C. 远程分离部署 | vLLM on aarch64 + Agent on x86_64 | ✅ 高 | 100% |
| D. 自建 aarch64 镜像 | 为 SWE-Bench 重建 aarch64 case 镜像 | ⚠️ 长期 | 100% |

**推荐策略**：
方案 C 是已有基础设施支撑（x86 runner + K8s 集群）。核心挑战是跨 Job 网络打通，可通过 K8s Service 或端口转发解决。

---

### 难点 4：执行时间与 CI 时长

**现状**：  
1. 当前 nightly 全量测试已运行数小时。
2. Agent 测评单个 case 可能耗时数分钟到数十分钟。
3. 完整 SWE-Bench Verified 有 500+ cases。即使 mini 数据集也有数十个 cases。

**影响**：中  
**可行性**：✅ 可解决（需取舍）

**解决方案**：
1. **降级为 weekly**：Agent 测评放在 weekly 而非 nightly
2. **设置超时**：AISBench 支持 `--timeout-multiplier` 和 `override_timeout_sec`

**时间预估**：
- terminal-bench 2.0 mini（20 cases，并发 5）：约 30-60 分钟
- SWE-Bench Verified mini（30 cases，并发 5）：约 1-2 小时
- SWEBench Multilingual：约 1-2 天

---

### 难点 5：结果格式与后处理适配

**现状**：  
当前 `result_postprocess.py` 和 `test_single_node.py` 的结果处理只认三种格式：
- `accuracy`：`isinstance(result, (int, float))` → `metrics["accuracy"]`
- `performance`：`isinstance(result, list) and len(result) == 2` → throughput metrics
- `spec_decode`：`isinstance(result, list)` → acceptance rates

Agent 结果格式完全不同：`avg_score` / `correct` / `wrong` / `exception` / `reward_distribution`。

**影响**：低  
**可行性**：✅ 可解决

**解决方案**：
1. 扩展 `test_single_node.py` 的 `_build_task_entry()`：
   ```python
   elif case_type == "agent" and isinstance(result, dict):
       metrics["avg_score"] = round(result.get("avg_score", 0), 4)
       metrics["correct"] = result.get("correct", 0)
       metrics["wrong"] = result.get("wrong", 0)
       metrics["exception"] = result.get("exception", 0)
   ```
2. 扩展 `_task_passed()`：
   ```python
   elif case_type == "agent" and isinstance(result, dict):
       return result.get("avg_score", 0) >= float(case_config.get("threshold", 0.5))
   ```
3. 扩展 `result_postprocess.py` 的 `merge_postprocess_payload()`：
   ```python
   elif case_type == "agent" and isinstance(result, dict):
       indicator["avg_score"] = round(float(result.get("avg_score", 0)), 4)
       indicator["correct"] = result.get("correct", 0)
   ```
4. 解析 AISBench 落盘的 `results/{模型}/{数据集}.json` 提取汇总指标

**技术评估**：改动集中在 3 个函数，风险可控。

---

## 五、可行性总结

| 难点 | 可解决性 | 影响度 | 结论 |
|------|---------|--------|------|
| 1. 调用接口不兼容 | ✅ 可解决 | 中 | 新增 agent 分支，import HarborRunner API |
| 2. Docker 容器管理 | ✅ 可解决 | 高 | 扩展 CI workflow + Dockerfile + context manager |
| 3. 架构兼容性限制 | ⚠️ 部分解决 | 高 |  远程分离 |
| 4. 执行时长 | ✅ 可解决 | 中 | mini 数据集 + 并发 + 降级 weekly |
| 5. 结果格式适配 | ✅ 可解决 | 低 | 扩展 3 个函数 |

**整体可行性评估：中高**  
五大难点中，4 个完全可解决，1 个（架构兼容性）可通过分阶段策略在可接受范围内解决。

---

## 六、推荐技术方案

### 6.1 YAML 配置扩展设计

```yaml
_benchmarks: &benchmarks
  agent_tb2:
    case_type: agent                          # 新增 case_type
    agent_name: terminus-2                   # Agent 名称
    dataset_path: /data/terminal-bench-2     # Harbor 格式数据集路径
    case_images: /data/tb2-images.tar        # case 镜像包
    agent_deps: /data/terminus-2-deps/       # Agent 依赖包
    agent_runtime_image: ghcr.io/aisbench/agent-runtime:v3.1-...
    n_concurrent: 5                          # 并发 trial 数
    n_attempts: 1                            # 尝试次数
    n_tasks: 20                              # 限制任务数
    baseline: 0.5                            # avg_score 基线
    threshold: 0.4                           # 最低通过线
    environment_type: docker                 # 环境类型
```

### 6.2 框架代码改动清单

| 文件 | 改动内容 |
|------|---------|
| `single_node_config.py` | `SingleNodeConfig` 无需改动（benchmarks 已是 `dict[str, Any]`） |
| `test_single_node.py` | 新增 `_run_agent_benchmarks()` + 扩展 `_run_benchmarks()` + 扩展 `_build_task_entry()` + 扩展 `_task_passed()` |
| `result_postprocess.py` | 扩展 `merge_postprocess_payload()` 的 `case_type == "agent"` 分支 |
| `Dockerfile.nightly.a3` | 增加 `docker.io` 安装 + agent 依赖预装 |
| `_e2e_nightly_single_node.yaml` | agent 测试 job 增加 docker socket 挂载 |
| 新增 `agent_runtime.py` | `AgentRuntimeContainer` context manager |
| 新增 `harbor_agent_task.py` | AISBench agent 配置模板生成器 |

---

## 七、实施路径

### Phase 1：最小可用验证

**目标**：在 aarch64 Ascend 环境上跑通 terminal-bench 2.0 mini

**关键任务**：
1. 确认 CI runner 支持 Docker socket 挂载（前置阻塞条件）
2. 解决 terminal-bench 外网访问限制（代理/白名单）（前置阻塞条件）
3. 实现 `_run_agent_benchmarks()` + `AgentRuntimeContainer`
4. 扩展结果处理（`_build_task_entry` / `_task_passed` / postprocess）
5. 编写 terminal-bench YAML 配置
6. 端到端联调，作为 weekly job 试运行

**验收标准**：
- Agent 测评在 CI 中成功执行并产出 `benchmark_results/*.json`
- `avg_score` 指标正确上报到 openlibing

### Phase 2：全量数据集覆盖

**目标**：通过远程分离部署支持全部 6 个 Agent 数据集

**关键任务**：
1. 设计 K8s vLLM 部署 + x86 runner Agent 测评的协调 workflow
2. 实现 vLLM K8s Service 暴露 + x86 runner 连接
3. 预加载 SWE-Bench 系列镜像到 x86 runner
4. 新增 SWE-Bench / DeepSWE YAML 配置
5. 多 Agent 对比测试支持（同一模型 + 不同 agent）
6. 端到端联调 + 性能优化

**验收标准**：
- SWE-Bench Verified / Multilingual / Pro 在 x86 runner 上成功执行
- vLLM 在 aarch64 NPU 上稳定服务，Agent 通过内网 API 调用

### Phase 3：生态共建

**目标**：推动 AISBench 社区构建 aarch64 case 镜像，实现原生全量支持

**关键任务**：
1. 向 AISBench 社区提 Issue / RFC
2. 协助构建 SWE-Bench aarch64 镜像原型
3. 推动自动化 aarch64 镜像构建流水线

---

## 八、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| CI runner 不支持 Docker socket 挂载 | 中 | 高 | 提前与运维确认；备选 DinD 模式 |
| terminal-bench 外网访问受限无法解决 | 中 | 中 | Phase 1 降级为手动触发；Phase 2 用 x86 runner 规避 |
| Agent 测评结果不稳定（同一 case 多次结果不同） | 中 | 中 | 设置 `n_attempts > 1`；使用 `avg_score` 而非单次 pass/fail |
| 跨机网络延迟影响 Agent 性能 | 低 | 低 | LLM 推理耗时远大于网络延迟；内网延迟 < 1ms |
| AISBench 版本升级导致接口变更 | 低 | 中 | 锁定 AIS_BENCH_TAG 版本；建立兼容性检查 |
| K8s 集群资源不足（A3 并发上限 5×16 NPU） | 中 | 中 | Agent 测评放 weekly 错峰执行；复用已有资源池 |

---

## 九、前置阻塞条件

以下两项需优先确认，否则阻塞 Phase 1 启动：

1. **CI runner 是否支持 Docker socket 挂载**——Agent runtime 容器依赖此项
2. **terminal-bench 外网访问是否可配置代理/白名单**——terminal-bench 执行中 Agent 需访问外网

---

## 十、结论与建议

### 10.1 结论

将 AISBench Agent 数据集测试集成到 vllm-ascend nightly 框架**技术可行**，五大难点均有对应解决方案。其中架构兼容性是最大挑战，但通过"terminal-bench 先行 → 远程分离部署 → 社区共建"的分阶段策略可有效规避。

### 10.2 建议

1. **优先确认两个前置阻塞条件**（见第九节）
2. **Phase 1 先行启动**：以 terminal-bench 2.0 mini 验证端到端链路
3. **Phase 2 纳入规划**：远程分离部署是最终形态，需提前协调 K8s 资源和 x86 runner
4. **同步推动社区共建**：向 AISBench 社区提出 aarch64 镜像需求，降低长期维护成本

### 10.3 预期收益

- **短期**：vllm-ascend 成为首个集成 Agent 测评的 vLLM 硬件插件 nightly 框架
- **中期**：Agent 测评覆盖 6 个主流数据集，全面评估昇腾 NPU 上 vLLM 的 Agent 能力
- **长期**：推动 AISBench + vllm-ascend 生态共建，建立行业 Agent 评测标准

---

**附录**：
- [AISBench Agent 测评文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/agent_benchmark.html)
- [vllm-ascend nightly 框架源码](https://github.com/vllm-project/vllm-ascend/tree/main/tests/e2e/nightly)
- [Harbor 框架（AISBench fork）](https://github.com/AISBench/harbor)