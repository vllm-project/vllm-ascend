---
name: log-quality-write
description: 从零为新功能编写符合 log-quality-standard 的日志语句。在不改变业务逻辑的前提下，给出每条日志的建议内容、级别、格式。触发词：写日志、编写日志、新增日志、日志设计。
---

# 日志编写器

从零为新功能编写符合 `log-quality-standard` 的日志语句。输入功能描述和代码，输出完整的日志设计方案。

---

## 触发词

写日志、编写日志、新增日志、日志设计

---

## 输入

| 输入 | 必需 | 说明 |
|------|------|------|
| 功能描述 | 必需 | 要打日志的功能，如"链路保活心跳"、"请求超时处理" |
| 代码片段 | 必需 | 要新增日志的代码（包含业务逻辑） |
| 涉及组件 | 必需 | 如 cmotor/router、vllm/scheduler |
| 已有日志 | 可选 | 该功能已有哪些日志（避免重复） |

---

## 输出

| 输出 | 格式 | 说明 |
|------|------|------|
| 日志设计方案 | Markdown | 每条日志的：位置、内容、级别、理由 |
| 代码模板 | 伪代码/代码片段 | 可直接粘贴到代码中的日志语句 |

---

## 工作流程

### 第一步：分析代码逻辑

理解代码的：
1. **执行路径**：有哪些分支（成功/失败/异常/边界）
2. **上下文**：有哪些变量可用（error_code、peer_id、timeout 等）
3. **调用关系**：是入口/中间/出口

### 第二步：确定打日志的位置

根据 `log-quality-standard` 标准 1（何时打日志），确定日志位置：

| 场景 | 建议 | 级别 |
|------|------|------|
| 进入关键函数 | 可选，仅关键入口打 INFO | INFO |
| 参数校验失败 | 必须打 | WARNING 或 ERROR（取决于是否阻断） |
| 外部调用（RPC/网络）前 | 可选 | DEBUG 或 INFO |
| 外部调用成功后 | 必须打 | INFO |
| 外部调用失败 | 必须打 | ERROR |
| 重试成功 | 必须打 | WARNING |
| 循环内失败 | 最后统一打 ERROR | ERROR（防刷屏） |
| 退出函数（成功/失败） | 可选，仅关键路径打 INFO | INFO |

### 第三步：按 `log-quality-standard` 中所有标准编写每条日志

#### 示例：为"链路建连"功能写日志

**代码片段：**
```cpp
int establish_link(const string& group_id, const string& peer) {
    if (group_id.empty() || peer.empty()) {
        return -1;  // 早返回
    }

    int ret = connect_to_peer(peer);
    if (ret != 0) {
        // 重试 3 次
        for (int i = 0; i < 3; i++) {
            ret = connect_to_peer(peer);
            if (ret == 0) break;
        }
        if (ret != 0) return -1;
    }

    return register_link(group_id, peer);
}
```

**日志设计方案：**

```markdown
## 日志设计方案：链路建连

| # | 代码位置 | 日志级别 | 日志内容 | 编写理由 |
|---|---------|---------|---------|----------|
| 1 | 参数校验后 | WARNING | `[cmotor/router] Link establish validation failed. group_id={group_id}, peer={peer}, reason=empty_id_or_peer` | 入参校验失败是异常，但不一定阻断，所以 WARNING |
| 2 | 首次连接失败 | DEBUG | `[cmotor/router] First connection attempt failed. peer={peer}, err={errno}` | 首次失败正常重试，DEBUG 足够 |
| 3 | 重试循环内 | （不打） | — | 防刷屏，不在循环内逐条打 |
| 4 | 重试全部失败 | ERROR | `[cmotor/router] trace_id={trace_id}, Link establish failed after 3 retries. group_id={group_id}, peer={peer}, err={errno}, 可检查：1. ping {peer_ip} 2. telnet {peer_ip} {port} 3. 对端进程是否存活` | 标准2+3+5，ERROR 级别，必须带 trace_id 和分析方案 |
| 5 | 建连成功 | INFO | `[cmotor/router] trace_id={trace_id}, Link established. group_id={group_id}, peer={peer}` | 成功路径打 INFO，带 trace_id 方便关联 |
```

### 第四步：输出代码模板

```cpp
// === 链路建连日志（按 log-quality-standard 编写） ===

// 位置1：参数校验后
if (group_id.empty() || peer.empty()) {
    logger_warning("[cmotor/router] Link establish validation failed. "
                   "group_id={}, peer={}, reason=empty_id_or_peer",
                   group_id.c_str(), peer.c_str());
    return -1;
}

// 位置2：首次连接失败（DEBUG）
int ret = connect_to_peer(peer);
if (ret != 0) {
    logger_debug("[cmotor/router] First connection attempt failed. "
                 "peer={}, err={}", peer.c_str(), ret);
}

// 位置3：重试（防刷屏，不打日志，计数器在循环外）
int retry_count = 0;
for (int i = 0; i < 3; i++) {
    ret = connect_to_peer(peer);
    if (ret == 0) break;
    retry_count++;
}

// 位置4：重试全部失败
if (ret != 0) {
    logger_error("[cmotor/router] trace_id=%s, Link establish failed after 3 retries. "
                 "group_id=%s, peer=%s, err=%d, "
                 "可检查：1. ping %s 2. telnet %s %s 3. 对端进程是否存活",
                 trace_id, group_id.c_str(), peer.c_str(), ret,
                 peer_ip.c_str(), peer_ip.c_str(), port.c_str());
    return -1;
}

// 位置5：建连成功
logger_info("[cmotor/router] trace_id=%s, Link established. group_id=%s, peer=%s",
             trace_id, group_id.c_str(), peer.c_str());
```

### 第五步：自检清单

写完后对照 `log-quality-standard` 自检：

```
[ ] 每条日志级别是否正确？（异常不是 INFO，成功不是 ERROR）
[ ] ERROR/WARNING 是否回答了"什么错+为什么+查哪里"？
[ ] 是否带了组件标识（[仓库/子模块]）？
[ ] 如果在循环中，是否用了防刷屏计数？
[ ] ERROR/WARNING 是否带了 trace_id？
[ ] 是否包含任何用户输入、prompt、PII？
[ ] 占位符是否完整（{id}、%s 等）？
```

---

## 注意事项

- 只提供日志方案，不改变业务逻辑
- 如果代码中已有日志，先评估是否合规，不合规的用 rewrite 方案替代
- 防刷屏计数器的维护方式需要在代码注释中说明
- trace_id 如果代码中没有，需要在函数签名中新增参数或从上下文获取
- 输出可以直接粘贴，但需注意语言（C++/Python）与代码一致
