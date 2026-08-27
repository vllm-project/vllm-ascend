# 全局预扫描 — 条例级匹配（大型 PR 检视）

派发为子 Agent 执行，**每个 file_group 派发 1 个子 Agent**，各组并行。

## 派发

```
Agent({
  subagent_type: "general",
  model: "haiku",
  description: "预扫描：" + group_name,
  prompt: "条例级匹配预扫描

【输入】
- 文件组名：{group_name}
- 本组文件列表：{group_file_list}
- 完整源码路径：{repo_path}

【执行要求】
严格按本文件「执行流程」中定义的 4 步执行，产出本组的 matched_rules 清单。禁止生成报告文件。"
})
```

---

## 执行流程（子 Agent 执行指南）

### Step 1 — 收集领域触发关键词

1. 执行 `sed -n '/<适用>/,/<\/适用>/p' references/*.md` 一次性提取所有文件的 <适用> 区块
2. 对 `领域: true` 的文件，提取 `触发:` 字段中的关键词列表
3. 合并去重，得到「全局领域关键词清单」

### Step 2 — 逐文件组 Grep 匹配

对 file-split 产出的每个 file_group：

```
a. 从文件扩展名判定语言:
     .cpp/.h/.hpp → C++
     .py → Python
     CMakeLists.txt/.cmake/Makefile → Build

b. 从文件路径判定侧别:
     op_kernel/ → Kernel
     op_host/   → Tiling
     其他       → 混合

c. Grep 领域关键词:
     在本组文件列表中 Grep 全局领域关键词
     命中任一关键词 → 激活对应的领域规则文件

d. 通用规则（领域=false）:
     语言匹配 + 侧别匹配 → 始终激活
```

### Step 3 — 逐条例内容筛查

对每个 file_group 的每个激活规则文件：

```
1. Read 该规则文件的「快速索引」表
2. 对表中每条条例:
   a. Read 条例的标题和「问题描述」章节（前 5-10 行）
   b. 提取该条例关注的关键代码模式:
      示例:
        SEC-2.1（溢出）      → grep 乘法*、类型转换(int32_t/int64_t/uint32_t)
        SEC-2.3（除零）      → grep 除法/、取余%
        SEC-3.1（未初始化）   → grep 变量声明未赋初值
        API-3（DataCopy对齐） → grep DataCopy、DataCopyPad
        PERF-1（流水线）     → grep EnQue、DeQue、pipe.InitBuffer
        TOPK-8（GM偏移）     → grep SetGlobalBuffer、GM_ADDR、offset.*=、uint32_t.*offset
        MC2-*（通信）        → grep hccl_、AlltoAll、AllGather
        SIMT-*（线程）       → grep Simt::、GetThreadNum、SetThreadNum
   c. Grep 本组文件中是否出现该关键模式
   d. 出现 → 纳入本组的 matched_rules 清单
   e. 未出现 → 跳过（该条例与本组代码无关）

3. 对纳入 matched_rules 的每条条例，Grep `^{条例ID}` 在对应规则文件中定位起始行号，记录到输出中
```

若无法从标题推断关键模式，保守保留该条例（宁可多检不少检）。

**硬约束：SEC 和 TOPK 条例永不过滤**

`cpp-secure.md`（全部 SEC-* 条例）和 `ascendc-topk.md`（全部 TOPK-* 条例）属于最高优先级安全规则，**内容筛查对其无效**——只要该规则文件被激活（语言+侧别匹配），其中所有条例全部纳入 matched_rules，不执行 Grep 跳过。

理由：安全条例的覆盖范围往往超出简单的关键词匹配（例如 SEC-2.1 溢出检查涉及隐式类型转换，不能仅靠 grep '*' 来判断）。遗漏一条安全条例的代价远高于多检一条。

### Step 4 — 输出

```
文件组匹配清单:

kernel_G1（Kernel侧，5文件）:
  激活文件: cpp-secure, ascendc-api, ascendc-perf, ascendc-topk
  匹配条例: SEC-2.1(cpp-secure:99), SEC-2.3(cpp-secure:115), SEC-3.1(cpp-secure:150), ...
  跳过条例: SEC-4.*（无外部输入）, SEC-5.*（无资源申请）, MC2-*（无hccl_）, SIMT-*（无Simt::）

kernel_G2（Kernel侧，3文件）:
  激活文件: cpp-secure, ascendc-api, ascendc-perf
  匹配条例: SEC-2.1, SEC-3.1, API-1, PERF-2, PERF-3
  跳过条例: ...

host_G1（Tiling侧，6文件）:
  激活文件: cpp-secure, cpp-general, ascendc-perf, ascendc-topk
  匹配条例: SEC-2.1, SEC-4.1, GEN-12, PERF-4, TOPK-8
  跳过条例: ...

shared（混合，2文件）:
  激活文件: compile-secure, cpp-general
  匹配条例: CMP-1, CMP-3, GEN-5
  跳过条例: ...
```

## 约束

- 仅 Grep 模式匹配，不读完整代码（保持极轻量）
- 内容筛查无法确定时保守保留（宁可多检不少检）
- 不生成报告文件，输出直接传给 clause-grouping
- 若派发为 haiku 子 Agent 执行，主 Agent 等待返回后继续
