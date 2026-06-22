# Framework Integration Baseline

本文档定义 **vLLM 框架层集成** 的当前能力基线，覆盖 scheduler、worker、sampler、attention backend 接线、KV cache group、feature flag 等不属于单个模型文件的部分。

## 1. 这一层解决什么问题

当模型 adapter 本身已经存在或基本正确，但运行时仍失败时，经常需要判断问题是不是来自：

- scheduler 假设变化；
- worker / model runner metadata 变化；
- sampler / logits path 变化；
- kv cache group / backend selector 假设变化；
- vllm-ascend 现有 patch 与新 upstream 代码漂移。

## 2. 当前能力基线

当前 skill 默认承认以下事实：

- `vllm-ascend` 已经覆盖了一部分通用 vLLM framework 模块；
- 新模型上游合入后，common module 行为可能变化，但不代表一定要新增 Ascend backend；
- 更常见的是已有 patch / override 需要重新对齐 upstream 改动。

## 3. 当前判断原则

### 3.1 先问“这个模块是不是已有 Ascend 覆盖”

如果某个变更发生在：

- scheduler
- worker / model_runner
- sampler
- attention backend selector
- kv cache interface

先搜索 `vllm_ascend/` 是否已有 patch 或依赖。

### 3.2 先修漂移，再新增 override

如果已有 patch，只是 upstream 改了接口或行为：

- 优先更新现有 patch；
- 不要立刻加第二套平行 override。

### 3.3 保持范围最小

framework 层改动很容易扩大影响面，必须：

- 只改触发不兼容的路径；
- 避免为单模型引入泛化过度的分支。

## 4. 典型输入证据

- 上游模型支持 commit 改了哪些非 model 文件
- `git diff` 的 framework 侧文件
- `vllm_ascend` 中是否已有对应 patch / override
- metadata shape / field mismatch
- scheduler 或 graph capture 相关报错

## 5. 固定输出模板

```markdown
## Framework Integration Gap Analysis

### 1. Current Capability
- Existing vllm-ascend coverage:
- Existing patch/override path:
- Existing framework assumptions:

### 2. Model Requirement
- Upstream framework modules touched:
- Runtime path exercised by this model:
- Required framework behavior:

### 3. Gap
- Upstream drift:
- Missing override:
- Metadata / interface mismatch:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Existing patch to update vs new override:
- Validation focus:
- Stop / escalate condition:
```

## 6. 最常见的适配动作

- 更新已有 patch；
- 只在必要时加最小 override；
- 调整 metadata / graph / scheduler 相关接线；
- 修 backend selector 与模型 attention 类型的连接关系。
