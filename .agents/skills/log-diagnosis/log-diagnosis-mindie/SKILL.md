---
name: log-diagnosis-mindie
description: MindIE-PyMotor 日志诊断调度器。接收 MindIE/PyMotor 相关日志，自动判断问题方向并调度对应子skill进行诊断。触发词：日志诊断、日志分析、日志定位、故障定位、MindIE日志、PyMotor日志。
---

# MindIE-PyMotor 日志诊断调度器

接收 MindIE-PyMotor 相关日志，自动判断方向，分发给对应子 skill 进行诊断，并汇总输出结论。

---

## 使用方式

**方式一：直接调度（用户已告知方向）**
```
/skill log-diagnosis-mindie
[粘贴日志]
> 我觉得可能是建链问题
```

**方式二：全自动（用户不知道什么方向）**
```
/skill log-diagnosis-mindie
[粘贴日志]
> 帮我看看是什么问题
```

---

## 诊断流程

```
Step 1：解析输入
Step 2：判断方向
Step 3：加载子 skill 并执行
Step 4：汇总输出
```

### Step 1：解析输入

同时解析两个输入：
- **原始日志**：用户粘贴的完整或部分日志
- **用户提示**（可选）：用户对可能方向的猜测

### Step 2：判断方向

三种路径，优先级从高到低：

| 路径 | 条件 | 处理方式 |
|------|------|----------|
| **直接调度** | 用户主动提示方向 | 根据提示直接加载对应子 skill |
| **错误特征分析** | 日志中有 error/warn 日志 | 分析错误语义，映射到对应方向 |
| **零样本分类** | 以上都没有命中 | LLM 根据日志特征判断最可能的方向 |

**错误特征 → 方向 映射表：**

| 错误特征 | 映射方向 | 对应子 skill |
|----------|----------|--------------|
| gRPC 超时/连接失败/链路超时 | PD建链 | `log-diagnosis-pd-link-establishment` |
| 建链失败/link failed/链路异常 | PD建链 | `log-diagnosis-pd-link-establishment` |
| 缩容/缩P/释放P实例/保D | 缩P保D | `log-diagnosis-shrink-p-reserve-d` |
| D实例故障/D节点异常/fault D | 缩P保D | `log-diagnosis-shrink-p-reserve-d` |

**零样本分类 prompt（备用）：**

```
你是一个 MindIE-PyMotor 推理服务的日志诊断专家。以下是一段服务日志，请判断最可能属于哪类问题：

日志：
---
{raw_logs}
---

选项：
A. PD建链问题（P节点与D节点之间的KVCache链路建立失败）
B. 缩P保D问题（PD分离架构中D实例故障后的缩容恢复流程异常）
C. 其他 MindIE 异常

请只输出选项字母和简短理由（1-2句话）。
```

### Step 3：加载子 skill 并执行

根据 Step 2 的判断结果，加载对应子 skill：

```
if 方向 == "PD建链":
    skill_view(name="log-diagnosis-pd-link-establishment")
    执行子 skill 的诊断流程
elif 方向 == "缩P保D":
    skill_view(name="log-diagnosis-shrink-p-reserve-d")
    执行子 skill 的诊断流程
elif 方向 == "无法判断" or 命中多个方向:
    所有方向都跑一遍，汇总各自发现
```

### Step 4：汇总输出

统一格式输出：

```
## 诊断结论

**判断方向**：xxx
**诊断依据**：...
**根因定位**：...
**修复建议**：...

---
## 详细诊断过程

[子 skill 的完整输出]

---
## 下一步

1. ...
2. ...
```

---

## 子 skill 清单

| 子 skill | 状态 | 说明 |
|----------|------|------|
| `log-diagnosis-pd-link-establishment` | ✅ 已就绪 | PD建链失败诊断 |
| `log-diagnosis-shrink-p-reserve-d` | ✅ 已就绪 | 缩P保D流程异常诊断 |

---

## 扩展方式

新增诊断方向时：

1. 在 `log-diagnosis-mindie/` 下创建子 skill 目录
2. 在上方「错误特征 → 方向映射表」中添加对应条目
3. 在「子 skill 清单」中更新状态

详见 `log-diagnosis-framework` 框架规范。
