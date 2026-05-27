---
name: log-diagnosis-shrink-p-reserve-d
description: 缩P保D流程异常的日志诊断。当PD分离架构中D实例故障后触发缩P保D，流程卡住或失败时，通过日志定位根因。触发词：缩P保D失败、缩P保D卡住、缩P保D异常、shrink P reserve D、缩容失败、D实例故障恢复失败。
parent: log-diagnosis-framework
---

# 缩P保D问题日志诊断

## 问题描述

**缩P保D**是指在PD分离架构中，当Decode(D)实例出现故障时，释放一个Prefill(P)实例为D实例腾出NPU资源，然后调度新D实例恢复服务的过程（如2P1D→1P1D）。

**涉及仓库/组件**：
- mindie-motor/controller: FaultManager、FaultHandler、ScaleInTimer
- mindie-llm: KVCache链路管理、建链/解链操作

**典型异常表现**：
- D实例故障后长时间未恢复服务
- P实例被释放但新D实例未加入
- 缩P保D流程卡在某个阶段不推进
- 建链反复失败，角色状态不变为READY

**关联问题**：建链失败可能是缩P保D的下游问题，如果诊断到建链阶段异常，转 `log-diagnosis-pd-link-establishment`。

---

## 诊断入口日志

| 入口日志 | 组件 | 含义 | grep 命令 |
|----------|------|------|-----------|
| `[FaultManager] Record hardware fault for node` | mindie-motor | D实例故障被记录 | `grep "Record hardware fault" $LOG_DIR/*.log` |
| `[FaultHandler] Unhealthy state handler` | mindie-motor | 缩容策略被触发 | `grep "Unhealthy state handler" $LOG_DIR/*.log` |
| `[FaultManager] Timer started with timeout` | mindie-motor | 缩容定时器启动 | `grep "Timer started with timeout" $LOG_DIR/*.log` |

**确认问题类型**：如果日志中出现上述入口日志，说明进入了缩P保D流程，继续诊断。

---

## 诊断决策树

```
[入口] D实例故障被记录
├── 故障检测阶段
│   ├── 有 "Unhealthy state handler" → 缩容策略已触发 ✓ → 继续定时器阶段
│   └── 无 "Unhealthy state handler" → 故障处理器未启动 ✗ → 故障1: FaultHandler未触发
│
├── 定时器阶段
│   ├── 有 "Timer started" → 定时器启动 ✓ → 继续等待
│   │   ├── 有 "Timer expired" → 定时器超时 ✓ → 继续缩P决策
│   │   ├── 有 "Timer stopped successfully" → 定时器被停止 → 说明有新D实例 → 继续扩容阶段
│   │   └── 无超时/停止日志 → 定时器卡住 ✗ → 故障2: 定时器未超时
│   └── 无 "Timer started" → 定时器未创建 ✗ → 故障3: 缩容策略生成失败
│
├── 缩P决策阶段
│   ├── 有 "Get static elastic pd cnt" → 正在获取期望实例数 ✓
│   │   ├── 有 "Current D instance number less" → D实例不足确认 ✓ → 继续释放P
│   │   └── 无上述日志 → D实例数量满足 → 不需要缩P（可能是误触发）
│   └── 无 "Get static elastic pd cnt" → 策略未执行 ✗ → 故障4: 缩P策略未执行
│
├── 释放P实例阶段
│   ├── 有 "Releasing prefill instances" → 开始释放 ✓
│   │   ├── 有 "Release.*successfully" → 释放成功 ✓ → 继续组更新
│   │   ├── 有 "Release.*failed" → 释放失败 ✗ → 故障5: P实例释放失败
│   │   └── 有 "All groups do not has enough" → 所有组P实例不足 ✗ → 故障6: 无可释放P实例
│   └── 无 "Releasing prefill instances" → 未尝试释放 ✗ → 故障7: 释放流程未启动
│
├── 组信息更新阶段
│   ├── 有 "Updating group info" → 组信息更新中 ✓
│   │   ├── 有 "send new peers info to new instances" → 消息下发中 ✓
│   │   └── 有 "send new peers info to old instances" → 旧实例通知中 ✓
│   └── 无 "Updating group info" → 组信息未更新 ✗ → 故障8: 组更新未触发
│
├── 扩容阶段（新D实例加入）
│   ├── 有 "Scale out operation: detected" → 检测到新节点 ✓
│   │   ├── 有 "has enough new nodes" → 节点充足 ✓ → 停定时器+建链
│   │   └── 有 "does not have enough new nodes" → 节点不足 → 定时器继续
│   └── 无 "Scale out operation" → 未检测到新节点 ✗ → 故障9: 新节点未加入
│
└── 建链阶段
    ├── 有 "start to process DMI link scenario" → 建链流程启动 ✓
    │   → 转 log-diagnosis-pd-link-establishment 继续诊断
    └── 无建链相关日志 → 建链未触发 ✗ → 故障10: 建链未触发
```

---

## 分阶段诊断流程

### 阶段1：故障检测

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| 1.1 | `grep "Record hardware fault" $LOG_DIR/*.log` | 出现故障记录，含node id和type | 无记录→故障未被检测，检查FaultManager是否正常运行 | 阶段2 |
| 1.2 | `grep "Handle hardware fault" $LOG_DIR/*.log` | 出现处理入口日志 | 无→故障未传递到处理器 | 阶段2 |
| 1.3 | `grep "Unhealthy state handler" $LOG_DIR/*.log` | 出现缩容策略触发日志 | 无→FaultHandler未识别为unhealthy状态，检查故障类型 | 阶段2 |

### 阶段2：定时器与策略

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| 2.1 | `grep "create a scaling timer" $LOG_DIR/*.log` | 创建缩容定时器 | 无→策略未生成，检查GenerateStrategy是否被调用 | 阶段3 |
| 2.2 | `grep "Timer started with timeout" $LOG_DIR/*.log` | 定时器启动，超时30秒 | 无→定时器启动失败 | 阶段3 |
| 2.3 | `grep "Timer expired\|Timer stopped" $LOG_DIR/*.log` | 超时或被停止 | 超时→走缩P流程；被停→有新实例走扩容 | 阶段3 |
| 2.4 | `grep "non-redundant scale in strategy" $LOG_DIR/*.log` | 策略触发检查 | "but time is not over"→还在定时等待中，正常 | 阶段3 |

### 阶段3：缩P决策与释放

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| 3.1 | `grep "Get static elastic pd cnt\|Get GRT PD cnt" $LOG_DIR/*.log` | 获取期望和实际P/D实例数 | 无→策略未执行，回到阶段2检查 | 阶段4 |
| 3.2 | `grep "Current D instance number less" $LOG_DIR/*.log` | 确认D实例不足 | 无→D实例数满足期望，不需要缩P | 阶段4 |
| 3.3 | `grep "Releasing prefill instances" $LOG_DIR/*.log` | 开始释放P实例 | 无→释放未启动，检查是否有组可选 | 阶段4 |
| 3.4 | `grep "Release.*successfully\|Release.*failed" $LOG_DIR/*.log` | 释放成功或失败 | 失败→检查P实例状态和NPU资源 | 阶段4 |
| 3.5 | `grep "does not have more than 1 prefill\|All groups do not has enough" $LOG_DIR/*.log` | P实例不足 | 出现→无可释放P实例，缩P保D无法执行 | 阶段4 |

### 阶段4：组更新与消息下发

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| 4.1 | `grep "Updating group info" $LOG_DIR/*.log` | 组信息开始更新 | 无→组更新未触发，检查ReleasePrefillInstances是否完成 | 阶段5 |
| 4.2 | `grep "send new peers info to new instances\|send new peers info to old instances" $LOG_DIR/*.log` | 对端信息下发 | 缺失某一步→检查postRole是否正常 | 阶段5 |
| 4.3 | `grep "instance group.*has.*prefill nodes and.*decode nodes" $LOG_DIR/*.log` | 组信息更新完成，新的P/D数量 | 数量不对→回查释放和扩容日志 | 阶段5 |

### 阶段5：扩容与建链

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| 5.1 | `grep "Scale out operation.*detected" $LOG_DIR/*.log` | 检测到新节点 | 无→新节点未加入集群，检查deployer和灵衢 | 阶段6 |
| 5.2 | `grep "available node size\|has enough new nodes\|does not have enough" $LOG_DIR/*.log` | 节点充足或不足 | 不足→定时器继续运行，等待更多节点 | 阶段6 |
| 5.3 | `grep "start to process DMI link scenario\|enter process pdRole" $LOG_DIR/*.log` | 建链流程启动 | 无→postRole未触发建链，检查mindie-llm侧 | 建链诊断 |

### 阶段6：服务恢复确认

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| 6.1 | `grep "Setting role status to READY" $LOG_DIR/*.log` | 角色就绪 | 无→建链未完成，转建链诊断 | 完成 |
| 6.2 | `grep "Engine started and ready to accept requests" $LOG_DIR/*.log` | 服务恢复 | 无→引擎启动异常 | 完成 |

---

## 快速诊断表

| 日志现象 | 根因 | 修复方法 | 优先级 |
|----------|------|----------|--------|
| 只有"Record hardware fault"无后续 | FaultHandler未触发缩容策略 | 检查故障类型是否为UNHEALTHY，确认FaultHandler是否正常注册 | P0 |
| "Timer started"后无超时/停止 | 定时器卡住不推进 | 检查ScaleInTimer线程是否存活，是否有死锁 | P0 |
| "All groups do not has enough prefill instance" | 所有组P实例≤1，无法释放 | 需要外部添加新节点资源，或调整静态弹性模板 | P1 |
| "Release.*prefill instance node.*failed" | P实例释放失败 | 检查NPU资源占用、P实例进程是否可终止 | P1 |
| 有"send new peers info"但无"Scale out operation" | 新节点未加入集群 | 检查deployer是否调度了新D实例，灵衢是否分配了NPU | P1 |
| 有"Scale out operation"但无"process DMI link" | postRole未触发建链 | 检查mindie-llm的pdRole接口是否正常响应 | P0 |
| 建链阶段反复失败 | 底层LLMDataDist或网络问题 | 转 log-diagnosis-pd-link-establishment | P0 |

---

## 故障清单详解

### 故障1: FaultHandler未触发

**现象**：有"Record hardware fault"但无"Unhealthy state handler"
**根因**：故障类型不被FaultHandler识别，或FaultHandler未注册该故障类型
**排查**：
1. `grep "Handle hardware fault" $LOG_DIR/*.log` — 确认是否有处理入口
2. 检查故障type字段，是否在FaultHandler的支持列表中
3. 检查FaultHandler是否正常初始化

### 故障2: 定时器未超时

**现象**：有"Timer started"但无"Timer expired"或"Timer stopped"
**根因**：ScaleInTimer线程挂起或死锁
**排查**：
1. 确认定时器超时时间（默认30秒）
2. 检查controller进程线程状态
3. 查看是否有锁竞争的日志

### 故障3: 缩容策略生成失败

**现象**：有"Unhealthy state handler"但无"create a scaling timer"
**根因**：GenerateStrategy未能生成缩P保D策略
**排查**：
1. `grep "GenerateStrategy\|AbortInstanceNPURecovery" $LOG_DIR/*.log`
2. 检查灵衢恢复流程是否被正确退出

### 故障4: 缩P策略未执行

**现象**：定时器超时后无"Get static elastic pd cnt"
**根因**：InstanceLevelNonRedundantScaleIn未被调用
**排查**：
1. `grep "non-redundant scale in strategy" $LOG_DIR/*.log`
2. 检查是否被"but time is not over"拦截

### 故障5: P实例释放失败

**现象**：有"Release.*failed"
**根因**：P实例进程终止失败或NPU资源释放失败
**排查**：
1. 确认P实例进程是否仍在运行
2. 检查NPU设备状态：`npu-smi info`
3. 检查是否有资源竞争

### 故障6: 无可释放P实例

**现象**：有"All groups do not has enough prefill instance"
**根因**：所有组内P实例数量≤1，释放后将无P实例可用
**修复**：需要外部添加新节点，或调整静态弹性模板降低期望D实例数

### 故障7: 释放流程未启动

**现象**：有"Current D instance number less"但无"Releasing prefill instances"
**根因**：SelectGroup2ReleaseInstance未找到可释放的组
**排查**：
1. `grep "active prefill instances\|does not have more than 1" $LOG_DIR/*.log`
2. 检查各组P实例数量

### 故障8: 组更新未触发

**现象**：P实例释放成功但无"Updating group info"
**根因**：ReleasePrefillInstances后的回调未执行
**排查**：
1. 检查ReleasePrefillInstances返回值
2. 确认ScalingUpdateAllGroups是否被调用

### 故障9: 新节点未加入

**现象**：有"send new peers info"但无"Scale out operation"
**根因**：deployer未调度新D实例，或灵衢未分配NPU
**排查**：
1. 检查deployer日志：是否有调度记录
2. 检查灵衢日志：是否有NPU分配记录
3. 检查集群资源池：是否有可用NPU

### 故障10: 建链未触发

**现象**：有"send new peers info to new instances"但无"enter process pdRole"
**根因**：postRole请求未到达mindie-llm，或llm未响应
**排查**：
1. 检查mindie-llm日志是否收到POST /v2/role请求
2. 检查网络连通性
3. 检查mindie-llm进程是否正常运行

---

## 关联问题

| 关联问题 | 关联方向 | 对应 Skill |
|----------|----------|------------|
| PD实例建链失败 | 下游 | log-diagnosis-pd-link-establishment |
| NPU硬件故障 | 上游 | - |
| 灵衢资源调度失败 | 上游 | - |
