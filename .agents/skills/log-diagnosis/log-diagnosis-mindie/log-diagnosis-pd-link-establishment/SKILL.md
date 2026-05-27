---
name: log-diagnosis-pd-link-establishment
description: PD实例建链失败的日志诊断。当P节点与D节点之间的KVCache链路建立失败、卡住或反复重试时，通过日志定位根因。触发词：建链失败、建链卡住、link failed、建链超时、Link exception、内存注册失败、PD链路异常。
parent: log-diagnosis-framework
---

# PD实例建链问题日志诊断

## 问题描述

**PD实例建链**是指Prefill(P)节点与Decode(D)节点之间建立KVCache传输链路的过程。建链由RouterImpl发起，经Config模块配置，由SeparateDeploymentWorker执行，底层调用LLMDataDist完成。

**涉及仓库/组件**：
- mindie-llm/RouterImpl: 建链流程编排（销毁旧链→创建新链→状态轮询）
- mindie-llm/SeparateDeploymentWorker: 建链执行（窗口线程+内存注册轮询）
- mindie-llm/SeparateDeploymentEngine: 底层LLMDataDist调用
- mindie-llm/Config: PD链路配置下发
- mindie-server: 角色设置与链路状态汇总

**典型异常表现**：
- 角色设置后长时间不进入READY状态
- query_link_status显示failed_links>0
- 建链卡在running状态不推进
- 推理请求报Pull KV失败

**关联问题**：建链是缩P保D的下游环节，缩P保D流程中新D实例加入后需要建链。如果是缩P保D场景触发的建链失败，先查 `log-diagnosis-shrink-p-reserve-d`。

---

## 诊断入口日志

| 入口日志 | 组件 | 含义 | grep 命令 |
|----------|------|------|-----------|
| `Link failed, error code is` | mindie-llm | 底层建链返回失败 | `grep "Link failed, error code" $LOG_DIR/*.log` |
| `Link exception from` | mindie-llm | 建链过程网络异常 | `grep "Link exception" $LOG_DIR/*.log` |
| `Mem status query failed` | mindie-llm | 内存注册失败 | `grep "Mem status query failed" $LOG_DIR/*.log` |
| `Max query attempts reached` | mindie-llm | 内存注册轮询超上限 | `grep "Max query attempts reached" $LOG_DIR/*.log` |
| `query link status failed` | mindie-llm | 链路状态查询失败 | `grep "query link status failed" $LOG_DIR/*.log` |
| `failed_links=[1-9]` | mindie-server | 有失败链路 | `grep "failed_links=[1-9]" $LOG_DIR/*.log` |

**确认问题类型**：如果入口日志命中，说明建链阶段出问题，继续诊断。

---

## 诊断决策树

```
[入口] 建链相关错误日志出现
│
├── 阶段A: 配置与角色设置
│   ├── 有 "start to set PD link/unlink info" → 配置下发正常 ✓
│   │   ├── 有 "PD switch is True" → PD开关开启 ✓ → 继续建链流程
│   │   ├── 有 "PD switch is False" → PD开关关闭 → 检查是否本就应该关闭
│   │   └── 有 "Do not need to link and unlink" → 无需建链 → 确认角色和配置
│   └── 无配置下发日志 → 配置阶段异常 ✗ → 故障A1: 配置未下发
│
├── 阶段B: 解链（旧链路清理）
│   ├── 有 "Destroy all clusters kvcache link start" → 开始解链 ✓
│   │   ├── 有 "Destroy all clusters kvcache link finish" → 解链完成 ✓ → 继续建链
│   │   ├── 有 "Batch unlink results" 含失败项 → 批量解链部分失败 → 故障B1: 解链部分失败
│   │   └── 有 "process DMI unlink failed" → DMI解链失败 ✗ → 故障B2: DMI解链失败
│   └── 无解链日志 → 跳过解链（可能无旧链路）→ 继续建链
│
├── 阶段C: 底层建链调用
│   ├── 有 "Link succeeded with status: SUCCESS" → 建链成功 ✓ → 继续内存注册
│   ├── 有 "Link failed, error code is" → 建链失败 ✗ → 故障C1: 底层建链返回错误码
│   ├── 有 "Link exception from" → 建链异常 ✗ → 故障C2: 建链网络异常
│   ├── 有 "Link already exists" → 链路已存在 → 故障C3: 重复建链
│   └── 无任何Link结果日志 → 建链未执行 ✗ → 故障C4: 建链未触发
│
├── 阶段D: 内存注册轮询
│   ├── 有 "Query completed in" → 内存注册成功 ✓ → 继续状态确认
│   ├── 有 "Mem status query failed" → 内存注册失败 ✗ → 故障D1: 内存注册失败
│   ├── 有 "Max query attempts reached" → 轮询超限 → 故障D2: 内存注册超时
│   └── 有 "Querying mem status" 但无完成/失败 → 卡在轮询 ✗ → 故障D3: 内存注册卡住
│
├── 阶段E: 状态确认
│   ├── 有 "add_to_success" → 建链成功 ✓ → 继续角色就绪
│   ├── 有 "add_to_failed" → 建链失败 → 回查阶段C/D
│   ├── 有 "query link status result" 含waitting/running → 链路未完成 → 回查卡在哪
│   └── 有 "query link status failed" → 查询接口异常 ✗ → 故障E1: 状态查询失败
│
└── 阶段F: 角色就绪
    ├── 有 "Setting role status to READY" → 建链全部完成 ✓ → 诊断结束
    ├── 有 "Processing link status" 但无READY → 有链路未完成 → 回查
    └── 无 "Processing link status" → 状态轮询未启动 ✗ → 故障F1: 状态轮询未启动
```

---

## 分阶段诊断流程

### 阶段A：配置与角色设置

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| A.1 | `grep "start to set PD link/unlink info" $LOG_DIR/*.log` | 配置下发开始 | 无→检查postRole是否成功发送到llm侧 | 阶段B |
| A.2 | `grep "PD switch is\|PD remote link/unlink info\|PD remote attr info\|PD remote policy info" $LOG_DIR/*.log` | PD开关和配置信息 | switch=False→检查角色配置是否正确 | 阶段B |
| A.3 | `grep "Do not need to link and unlink\|do not link with remote" $LOG_DIR/*.log` | 无需建链时出现 | 如果不应出现→检查角色和rank配置 | 阶段B |
| A.4 | `grep "Reset PD role finish\|Is remote dp group across\|role:.*tp_p\|using link map" $LOG_DIR/*.log` | 角色和并行配置 | link map异常→检查配置文件 | 阶段B |

### 阶段B：解链（旧链路清理）

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| B.1 | `grep "start to process DMI link scenario\|Destroy all clusters kvcache link start" $LOG_DIR/*.log` | DMI建链流程启动，先解链 | 无解链→可能无旧链路，正常跳过 | 阶段C |
| B.2 | `grep "Batch destroy clusters start\|Batch unlink results" $LOG_DIR/*.log` | 批量解链执行和结果 | 结果含失败项→检查失败的cluster_id | 阶段C |
| B.3 | `grep "Destroy all clusters kvcache link finish\|process DMI unlink failed" $LOG_DIR/*.log` | 解链完成或失败 | DMI解链失败→旧链路残留，可能影响新建链 | 阶段C |

### 阶段C：底层建链调用

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| C.1 | `grep "Create all clusters kvcache links start" $LOG_DIR/*.log` | 新建链流程启动 | 无→解链阶段异常未完成，回查阶段B | 阶段D |
| C.2 | `grep "Link params cluster_rank_info" $LOG_DIR/*.log` | 建链参数准备完成 | 无→建链参数生成失败 | 阶段D |
| C.3 | `grep "Link succeeded\|Link failed\|Link exception\|Link already exists" $LOG_DIR/*.log` | 建链结果 | 见决策树各分支 | 阶段D |
| C.4 | `grep "add_to_waitting\|add_to_running\|add_to_failed" $LOG_DIR/*.log` | 建链状态流转 | 直接failed→底层问题 | 阶段D |

### 阶段D：内存注册轮询

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| D.1 | `grep "Querying mem status" $LOG_DIR/*.log` | 有轮询记录 | 无→建链可能未到running阶段 | 阶段E |
| D.2 | `grep "Query completed in\|Mem status query failed\|Max query attempts" $LOG_DIR/*.log` | 轮询结果 | 失败→见故障D1/D2 | 阶段E |
| D.3 | `grep "add_to_success\|add_to_failed" $LOG_DIR/*.log` | 最终状态 | 全failed→确认是内存注册问题还是底层link问题 | 阶段E |
| D.4 | `grep "Failed to unlink\|Unlink failed" $LOG_DIR/*.log` | 解链清理结果 | 解链失败→资源可能泄漏 | 阶段E |

### 阶段E：状态确认

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| E.1 | `grep "global rank.*enter query link status" $LOG_DIR/*.log` | 状态查询启动 | 无→查询未启动，检查RouterImpl | 阶段F |
| E.2 | `grep "query link status result\|QueryPDLinkStatus" $LOG_DIR/*.log` | 查询结果 | "failed"→回查阶段C/D | 阶段F |
| E.3 | `grep "Processing link status" $LOG_DIR/*.log` | 链路状态汇总 | running/waitting>0→仍在进行中，等待 | 阶段F |

### 阶段F：角色就绪

| 步骤 | 检查日志 | 预期结果 | 异常处理 | 下一阶段 |
|------|----------|----------|----------|----------|
| F.1 | `grep "All links completed successfully\|Setting role status to READY" $LOG_DIR/*.log` | 角色就绪 | 无→有链路未完成，回查 | 完成 |
| F.2 | `grep "Engine started and ready to accept requests" $LOG_DIR/*.log` | 服务可用 | 无→引擎未启动，检查LlmEngine | 完成 |

---

## 快速诊断表

| 日志现象 | 根因 | 修复方法 | 优先级 |
|----------|------|----------|--------|
| `Link failed, error code is {code}` | LLMDataDist底层返回错误码 | 根据错误码查表：检查rank_table、IP映射、对端进程 | P0 |
| `Link exception from {local} to {remote}` | 网络异常（连接中断/超时） | 检查RDMA状态、网络连通性、防火墙 | P0 |
| `Mem status query failed` 无retry | 对端NPU内存注册失败 | 登录对端检查NPU状态、内存使用 | P0 |
| `Max query attempts reached` 后requeueing | 对端响应慢 | 检查对端负载、网络延迟、kv_link_timeout配置 | P1 |
| `Link already exists` | 重复建链 | 检查是否解链不彻底，先手动解链再重建 | P2 |
| `process DMI unlink failed` | 旧链路清理失败 | 检查底层资源状态，可能需重启 | P1 |
| `query link status failed` | 状态查询接口异常 | 检查LLMManager和底层连接 | P1 |
| `Batch unlink results` 含失败项 | 批量解链部分失败 | 逐个检查失败cluster_id对应的链路 | P1 |
| 无"Create all clusters kvcache links start" | 解链阶段未完成或配置错误 | 回查解链阶段和PD配置 | P0 |
| `add_to_waitting` 后无 `add_to_running` | 窗口线程未处理 | 检查_fill_window_worker线程是否存活 | P0 |
| 有`add_to_running`但无`Querying mem status` | 处理窗口线程未启动 | 检查_process_window_worker线程是否存活 | P0 |

---

## 故障清单详解

### 故障A1: 配置未下发

**现象**：无"start to set PD link/unlink info"
**根因**：postRole请求未到达Config模块，或Config模块未解析
**排查**：
1. `grep "enter process pdRole" $LOG_DIR/*.log` — 确认pdRole入口是否到达
2. `grep "POST /v2/role" $LOG_DIR/*.log` — 确认HTTP请求是否到达
3. 检查mindie-llm进程是否正常运行

### 故障C1: 底层建链返回错误码

**现象**：`Link failed, error code is {code}`
**根因**：LLMDataDist底层建链失败，具体原因看错误码
**排查**：
1. 记录error code，查阅LLMDataDist错误码表
2. 检查rank_table配置：`cat /path/to/rank_table.json`
3. 检查cluster_rank_info参数是否与实际拓扑一致
4. 检查对端节点：`ps aux | grep mindie`，`npu-smi info`
5. 检查/etc/hosts中IP映射是否正确
6. 检查LLMDataDist日志：`/var/log/llmdatadist/*.log`

### 故障C2: 建链网络异常

**现象**：`Link exception from {local_ip} to {remote_ip}: {exception}`
**根因**：网络连接中断、超时或RDMA异常
**排查**：
1. 确认异常IP对：local_ip → remote_ip
2. 检查网络连通性：`ping {remote_ip}`、`traceroute {remote_ip}`
3. 检查RDMA状态：`ibv_devinfo`、`rdma link show`
4. 检查防火墙：`iptables -L -n`
5. 检查网络带宽：`iperf3 -c {remote_ip}`
6. 检查系统日志：`dmesg | grep -i network`

### 故障C3: 重复建链

**现象**：`Link already exists with status: ALREADY_LINK`
**根因**：链路已存在，可能是上次建链未解链
**排查**：
1. 检查是否有对应的解链操作
2. 确认是否需要先解链再建链
3. 检查cluster_comm_map中是否已有该链路记录

### 故障C4: 建链未触发

**现象**：有"Create all clusters kvcache links start"但无Link相关日志
**根因**：SeparateDeploymentWorker未执行建链，或队列未处理
**排查**：
1. 检查`add_to_waitting`是否有记录
2. 检查_fill_window_worker线程是否存活
3. 检查link_queue是否有积压

### 故障D1: 内存注册失败

**现象**：`Mem status query failed for {remote_ip}, no retry.`
**根因**：对端NPU内存状态异常，无法完成内存注册
**排查**：
1. 登录对端节点：`npu-smi info`
2. 检查NPU内存使用：`npu-smi det -t usages -i 0`
3. 检查是否有OOM：`dmesg | grep "Out of memory"`
4. 检查NPU驱动状态：`cat /proc/driver/npu/info`
5. 检查内存注册日志：`dmesg | grep -i "memory\|register"`

### 故障D2: 内存注册超时

**现象**：`Max query attempts reached for {remote_ip}, requeueing`
**根因**：对端响应慢，内存注册未在预期时间内完成
**排查**：
1. 检查对端节点负载：`top`、`iostat -x 1`
2. 检查网络延迟：`ping -c 10 {remote_ip}`
3. 检查kv_link_timeout配置是否合理
4. 检查内存注册进度：`grep "Querying mem status" $LOG_DIR/*.log`

### 故障D3: 内存注册卡住

**现象**：有"Querying mem status"持续输出但无完成/失败
**根因**：_process_window_worker卡在轮询循环
**排查**：
1. 统计"Querying mem status"出现次数
2. 检查是否一直返回PENDING状态
3. 检查_process_window_worker线程状态

### 故障E1: 状态查询失败

**现象**：`query link status failed`
**根因**：QueryPDLinkStatus接口异常
**排查**：
1. `grep "QueryPDLinkStatus" $LOG_DIR/*.log` — 检查查询接口
2. 检查LLMManager状态
3. 检查底层资源：`lsof | grep mindie`

### 故障F1: 状态轮询未启动

**现象**：建链成功后无"Processing link status"汇总
**根因**：RouterImpl的定时查询未启动
**排查**：
1. `grep "global rank.*enter query link status" $LOG_DIR/*.log`
2. 检查RouterImpl是否正常运行
3. 检查10秒定时查询是否被启动

---

## 常用检查命令汇总

```bash
# 变量设置
LOG_DIR=/var/log/mindie    # 根据实际环境修改

# 1. 建链整体状态
grep -E "Link (succeeded|failed|exception|already exists)|add_to_(success|failed|running|waitting)" $LOG_DIR/*.log

# 2. 内存注册状态
grep -E "Querying mem status|Query completed|Mem status query failed|Max query attempts" $LOG_DIR/*.log

# 3. 链路状态查询
grep -E "query link status|Processing link status|QueryPDLinkStatus" $LOG_DIR/*.log

# 4. 解链状态
grep -E "Destroy all clusters|Batch (destroy|unlink)|process DMI unlink" $LOG_DIR/*.log

# 5. PD配置
grep -E "PD (switch|remote|link/unlink)|start to set PD" $LOG_DIR/*.log

# 6. 角色就绪
grep -E "Setting role status to READY|Engine started and ready" $LOG_DIR/*.log

# 7. 快速统计建链成功/失败数
grep -c "add_to_success" $LOG_DIR/*.log
grep -c "add_to_failed" $LOG_DIR/*.log
```

---

## 关联问题

| 关联问题 | 关联方向 | 对应 Skill |
|----------|----------|------------|
| 缩P保D流程异常 | 上游 | log-diagnosis-shrink-p-reserve-d |
| Pull KV失败 | 下游 | - |
| NPU硬件故障 | 上游 | - |
| RDMA网络异常 | 并发 | - |
