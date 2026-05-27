# 日志标准规范（外部通用版）

AI 推理服务（MindIE-PyMotor / vLLM-Ascend）的日志质量标准。

---

## 核心原则

AI 推理服务（MindIE-PyMotor / vLLM-Ascend）的日志质量标准，定义如下：

### 标准 1：事件必记

**原则：明确哪些场景必须打日志，确保关键操作和状态变化可追溯。**

#### 1.1 状态改变

系统或组件的状态发生改变时，必须记录日志。

| 状态改变类型 | 记录内容 | 日志级别 |
|------------|---------|---------|
| 服务状态切换 | 状态变更前后值、触发原因、操作者 | INFO 或 ERROR |
| 角色切换（P/D 切换） | 切换原因、切换前后角色、影响范围 | INFO |
| 实例上下线 | 上线/下线原因、时间、实例标识 | INFO |
| 配置变更生效 | 变更前后的配置值、变更来源、审批人 | INFO |
| 连接状态变化 | 连接建立/断开、Peer 信息、原因 | INFO 或 ERROR |

#### 1.2 系统进程关键动作

系统进程的关键动作必须记录，确保可审计和可复现。

| 关键动作 | 记录内容 | 日志级别 |
|---------|---------|---------|
| 进程启动/退出 | 进程名、PID、启动参数、退出码 | INFO |
| 进程挂死/超时检测 | 检测时间、最后一次正常心跳、等待时长 | ERROR |
| 进程被 kill | 触发 kill 的原因、操作者（人或系统） | WARNING |
| 定时任务触发/完成 | 任务名、触发时间、耗时、结果 | INFO |
| 主备切换 | 切换原因、新主节点、切换耗时 | INFO |

#### 1.3 硬件资源

硬件资源异常或触及阈值时必须记录。

| 资源类型 | 记录内容 | 日志级别 |
|---------|---------|---------|
| GPU 显存 | 当前使用量、总容量、使用率、触发阈值 | WARNING 或 ERROR |
| GPU 温度/功耗 | 当前值、阈值、触发时间 | WARNING |
| CPU 使用率 | 当前使用率、进程级 CPU 占用、触发阈值 | WARNING |
| 内存使用 | 当前使用量、总容量、使用率、触发阈值 | WARNING 或 ERROR |
| 磁盘空间 | 当前剩余空间、总容量、触发阈值 | WARNING |
| 网卡吞吐/带宽 | 当前吞吐、触发阈值、丢包率 | WARNING |

#### 1.4 告警事件

触发系统级告警的事件必须记录，便于事后分析和告警收敛。

| 告警类型 | 记录内容 | 日志级别 |
|---------|---------|---------|
| 业务告警触发 | 告警名称、触发条件、当前值、阈值 | WARNING |
| 业务告警恢复 | 告警名称、恢复时间、持续时长 | INFO |
| 告警抑制/合并 | 被抑制的告警列表、抑制原因 | DEBUG |
| 告警通知发送 | 通知渠道、接收人、告警内容 | INFO |

#### 1.5 维护操作

人工维护操作必须记录，确保可审计。

| 维护操作 | 记录内容 | 日志级别 |
|---------|---------|---------|
| 手动重启服务 | 操作人、操作时间、原因、影响范围 | INFO |
| 手动切换主备 | 操作人、操作时间、原因 | INFO |
| 配置文件修改 | 操作人、操作时间、修改前后的配置项 | INFO |
| 人工上下线实例 | 操作人、操作时间、实例标识、原因 | INFO |
| 数据清理/迁移 | 操作人、操作时间、影响范围、清理数据量 | INFO |

---

### 标准 2：运维管理

**原则：日志需要自动转储和老化，确保可追溯性的同时控制存储成本。**

#### 2.1 自动转储

日志在产生后需要自动转储到指定存储。

| 要求 | 说明 |
|------|------|
| 转储时机 | 日志产生后尽快转储，延迟不超过 5 分钟 |
| 转储格式 | 结构化日志（JSON）优先，便于检索和分析 |
| 转储路径 | 按组件、日期分目录，不建议所有日志混合存储 |
| 转储完整性 | 转储前校验日志条数，确保不丢日志 |
| 异常处理 | 转储失败时记录错误日志，本地保留直到转储成功 |

#### 2.2 老化规则

日志需要按策略老化，控制存储成本。

| 日志级别 | 保留时长 | 说明 |
|---------|---------|------|
| ERROR | 30 天 | 故障相关日志，保留较长时间便于复盘 |
| WARNING | 15 天 | 异常相关日志，保留适中时间 |
| INFO | 7 天 | 正常流程日志，仅保留必要周期 |
| DEBUG | 3 天 | 调试日志，尽快清理 |

#### 2.3 存储要求

| 要求 | 说明 |
|------|------|
| 存储容量 | 预估存储容量，确保有足够磁盘空间 |
| 存储路径 | 日志存储路径与应用路径分离，避免相互影响 |
| 磁盘监控 | 监控存储使用率，使用率达到 80% 时告警 |
| 紧急清理 | 磁盘不足时，按优先级（先 DEBUG 后 ERROR）紧急清理 |

#### 2.4 日志安全

| 要求 | 说明 |
|------|------|
| 访问权限 | 日志文件权限控制，禁止未授权访问 |
| 传输安全 | 日志传输使用加密通道（HTTPS/TLS） |
| 脱敏合规 | 转储前确保敏感信息已脱敏（见标准 8：隐私保护） |

---

### 标准 3：分级清晰

**原则：有问题的日志必须是 ERROR 或 WARNING，绝不允许是 INFO。正常流程打 INFO，出现异常才升级。**

| 级别 | 触发条件 | 示例 |
|------|---------|------|
| ERROR | 业务流程中断、需要人工介入才能恢复、影响部分或全部推理请求 | 链路建连失败、下游 RPC 超时、模型加载失败、显存不足 |
| WARNING | 局部异常、不影响主流程但需要关注、可自动恢复 | 连接池接近满、某次重试成功、超时阈值临近、配置项使用默认值 |
| INFO | 正常业务流程的关键节点 | 服务启动、请求开始/结束、链路建立成功 |
| DEBUG | 开发调试信息 | 循环内每次迭代、详细参数打印 |

**判断树：**

```
这条日志要打吗？
├── 正常流程？
│   ├── YES → INFO（仅关键节点，非每个函数入口）
│   └── NO  → 继续
├── 影响推理请求或业务流程？
│   ├── YES → ERROR
│   └── NO  → 继续
├── 可自动恢复且偶发？
│   ├── YES → WARNING
│   └── NO  → ERROR
```

**常见误用：**
- 循环重试中每次失败都打 ERROR → 应改为 WARNING + 计数，最后全部失败才打 ERROR
- 入参校验失败打 ERROR 但业务没中断 → 应改为 WARNING
- "服务启动中"打 ERROR → 应改为 INFO 或 WARNING

---

### 标准 4：描述充分

**原则：错误日志必须回答三个问题：什么错 + 为什么 + 查哪里。**

#### 4.1 必有字段

每条 ERROR/WARNING 日志必须包含：

| 字段 | 要求 | 示例 |
|------|------|------|
| 错误描述 | 清晰描述故障本身，不含动态值 | "KVCache 链路建连失败" |
| 根因 | 如果知道根因必须写，否则写"原因待查" | "原因：上游 P 节点网络不可达" |
| 参数快照 | 与故障相关的关键入参/配置值 | `group_id=g_123, peer=10.0.0.5:8080` |
| 对比值 | 当问题是"值不符合预期"时，必须打印期望值和实际值 | `expected: timeout>0, actual: timeout=-1` |
| 进一步分析方案 | 告诉运维下一步查什么 | "可检查：该 P 节点网络连通性、端口是否监听、对端进程是否存活" |

#### 4.2 参数快照规则

- **必须打印的参数**：与故障直接相关的配置、ID、地址、状态码、数值
- **禁止打印的参数**：用户原始输入（prompt、messages）、PII 内容
- **对比值获取**：期望值从代码常量、配置默认值、协议规范中获取，禁止硬编码无法追溯的魔法数字

#### 4.3 分析方案模板

```
可检查：
1. 网络连通性：ping/telnet {peer_ip} {port}
2. 端口监听：ss -tlnp | grep {port}
3. 进程存活：ps -ef | grep {process_name}
4. 配置一致性：检查 {config_key} 是否一致
```

#### 4.4 示例

**好日志：**
```
[ERROR] [llm/router] KVCache link establishment failed.
  group_id: g_123, local_role: P, peer: 10.0.0.5:8080
  error: connection timeout (errno=110)
  expected: timeout > 0, actual: timeout = -1
  可检查：1. ping 10.0.0.5 2. telnet 10.0.0.5 8080 3. 对端 P 节点进程是否存活
  trace_id: req_abc123
```

**坏日志（不要这样写）：**
```
[ERROR] link failed              ← 缺少组件、参数、根因
[ERROR] connect failed: 110      ← errno 没解释，不知道查哪里
[WARN] param error               ← 什么参数、期望什么值，都没有
```

---

### 标准 5：组件归属明确

**原则：每条日志必须能让人一眼看出是哪个组件打的，且能找到对应负责人。**

#### 5.1 组件标识格式

```
[{仓库缩写}/{子模块}] 日志内容
```

| 仓库 | 缩写 | 示例 |
|------|------|------|
| MindIE-PyMotor | pymotor | `[pymotor/coordinator]` |
| cmotor | cmotor | `[cmotor/router]` |
| MindIE-LLM | llm | `[llm/model]` |
| vLLM-Ascend | vllm | `[vllm/scheduler]` |

#### 5.2 子模块粒度

子模块按代码目录/类名前缀划分，粒度以"能定位到具体 team 或负责人"为准，不要过粗（如整个 llm）或过细（如单个函数）。

**正确：** `[pymotor/coordinator]`、`[cmotor/router]`、`[vllm/scheduler]`  
**错误：** `[llm]`（太粗）、`[pymotor/CoordinatorClass/handleRequest_inner_helper]`（太细）

#### 5.3 运维找人路径

日志系统收集 → 按组件过滤 → 告警通知到对应 team

---

### 标准 6：防刷屏

**原则：同一错误在短时内重复出现，必须合并为一条带计数，不允许逐条打印。**

#### 6.1 合并规则

| 场景 | 处理方式 |
|------|---------|
| 循环内同一 error 重复出现 | 最后统一打一条，带 `count=N` |
| 指数退避重试 | 只打最终失败那次，或最终失败 + 重试总次数 |
| 短时间内相同错误（窗口内） | 合并为一条，注明 `last {window}s saw {count} occurrences` |

#### 6.2 实现建议

```python
# 伪代码示例
from collections import defaultdict
import time

error_counter = defaultdict(lambda: {"count": 0, "first_seen": 0, "last_msg": ""})
WINDOW_SECONDS = 10

def log_error_once(key: str, msg: str, level: str = "ERROR"):
    now = time.time()
    if error_counter[key]["count"] == 0:
        error_counter[key]["first_seen"] = now
    error_counter[key]["count"] += 1
    error_counter[key]["last_msg"] = msg

    if now - error_counter[key]["first_seen"] >= WINDOW_SECONDS:
        final_msg = f"{msg} (last {WINDOW_SECONDS}s saw {error_counter[key]['count']} occurrences)"
        # 实际打印
        print(f"[{level}] {final_msg}")
        error_counter[key] = defaultdict(int)
```

#### 6.3 计数阈值建议

| 错误类型 | 建议窗口 | 建议阈值 |
|---------|---------|---------|
| 网络连接失败 | 10s | count >= 3 |
| RPC 超时 | 30s | count >= 2 |
| 模型加载失败 | 不合并 | 立即打 |

---

### 标准 7：链路追踪

**原则：所有 ERROR/WARNING 日志必须携带 trace_id / request_id，方便跨进程串联全链路日志。**

#### 7.1 必有字段

```
trace_id: {uuid 或自定义 ID}
```

#### 7.2 来源

- 请求入口（API 网关 / HTTP 入口）生成 trace_id，贯穿整个请求生命周期
- 如果请求未携带 trace_id，在第一个日志点生成，后续日志沿用

#### 7.3 格式要求

- trace_id 必须是全局唯一 ID（UUID 或 Snowflake）
- 必须出现在日志的固定位置（建议在组件标识之后、描述之前）
- P 节点和 D 节点各自打印日志时，使用同一个 trace_id

#### 7.4 示例

```
[pymotor/coordinator] trace_id=req_8f3a9c, KVCache link established. group_id=g_123 -> 10.0.0.5:8080
[cmotor/router] trace_id=req_8f3a9c, Routing request to group g_123
[vllm/scheduler] trace_id=req_8f3a9c, Inference completed. prefill_tokens=1024, decode_tokens=128
```

#### 7.5 例外

- 服务启动/停止、后台定时任务等没有请求上下文的日志，不需要 trace_id，但必须带 `component` 和时间戳
- 上述例外场景的日志，打印 `job_id` 或 `task_name` 作为替代标识

---

### 标准 8：隐私保护

**原则：错误日志要带够上下文帮助排查，但不能包含用户原始输入或任何 PII。**

#### 8.1 允许打印的上下文

| 类型 | 示例 |
|------|------|
| 配置/参数 ID | `model_id=qwen3-235b, tensor_parallel=4` |
| 请求维度 | `input_len=1024, output_len=512, batch_size=8` |
| 系统资源 | `gpu_mem_used=28GB, gpu_mem_total=80GB` |
| 内部状态 | `kvcache_miss_rate=0.15, num_active_requests=32` |

#### 8.2 禁止打印的内容

| 禁止类型 | 示例 |
|---------|------|
| 用户原始输入 | 用户 prompt、messages 内容、system prompt |
| PII | 手机号、邮箱、身份证、用户名 |
| 完整 API Key/Token | `api_key=sk-xxxxxx` 应改为 `api_key=sk-****` |
| 敏感文件内容 | 文件里的密钥、证书、私钥 |

#### 8.3 脱敏规则

如果确需打印某参数值用于调试，先脱敏：
- 手机号/邮箱：只保留前3后4位 `138****1234`
- Token：只留前4后4 `sk-xxxx****xxxx`
- 完整文件路径中的用户名：替换为 `{user}`

---

## 附录

### 仓库/组件速查表

| 仓库 | 组件 | 说明 |
|------|------|------|
| pymotor | coordinator | 调度协调器 |
| pymotor | config_manager | 配置管理 |
| cmotor | router | 路由 |
| cmotor | load_balancer | 负载均衡 |
| llm | model | 模型推理 |
| llm | kv_cache | KVCache 管理 |
| vllm | scheduler | vLLM 调度器 |
| vllm | worker | vLLM Worker |
| vllm-ascend | ascend backend | Ascend 后端 |

### ERROR 和 WARNING 的边界判断

| 场景 | 错误等级 | 判断理由 |
|------|---------|---------|
| 链路建连失败 | ERROR | 推理请求无法继续 |
| 单次 RPC 超时但有重试最终成功 | WARNING | 业务没断，最终通了 |
| 连续 N 次 RPC 超时最终失败 | ERROR | 业务中断 |
| 配置项使用默认值（与预期不符） | WARNING | 可能导致性能问题，但不阻断 |
| 入参校验失败 | WARNING 或 ERROR | 取决于是否影响后续处理 |
| 连接池满 | ERROR 或 WARNING | 接近满 → WARNING；完全满 → ERROR |
| GPU 显存使用 > 90% | WARNING | 接近瓶颈 |
| GPU 显存完全耗尽 | ERROR | 无法继续分配 |

### 好日志自检清单

打印之前逐条检查：

```
[ ] 这条日志属于 ERROR / WARNING / INFO / DEBUG 中的哪一级？
[ ] 是否属于"事件必记"中要求的必须记录的场景？（状态改变/关键动作/硬件资源/告警/维护）
[ ] 级别判断是否符合"分级清晰"原则？（有问题的不能是 INFO）
[ ] ERROR/WARNING 日志是否回答了"什么错 + 为什么 + 查哪里"？
[ ] 参数错误场景是否打印了期望值和实际值？
[ ] 是否带 trace_id / request_id？（ERROR/WARNING 必须带）
[ ] 是否在组件标识中写明了仓库和子模块？
[ ] 是否包含任何用户输入、prompt 或 PII？（禁止出现）
[ ] 如果在循环内，是否考虑了防刷屏合并？
```
