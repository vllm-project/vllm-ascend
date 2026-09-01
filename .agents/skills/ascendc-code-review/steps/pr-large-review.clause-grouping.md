# 条例分组 — 小组策略 + 负载感知波次规划（大型 PR 检视）

派发为子 Agent 执行。基于 global-pre-scan 产出的 per-group matched_rules，按小组策略分组，再按负载感知算法构建波次。

## 派发

```
Agent({
  subagent_type: "general",
  model: "haiku",
  description: "条例分组 + 波次规划",
  prompt: "条例分组 + 负载感知波次规划

【输入】
- per-group matched_rules: {matched_rules}
- 各组文件数: {group_file_counts}
- 检视类型: {review_type}（pr）
- 检视标识: {review_id}（PR 号）

【执行要求】
先 Read `core/review-load-balance.md` 获取规则（每组容量从文件 `<检视负载>` 头读取、合并取最小值、每波上限）。
严格按本文件「执行流程」中定义的步骤执行，产出全局波次规划表 + yaml_dir。禁止生成报告文件。"
})
```

---

## 执行流程（子 Agent 执行指南）

## 前置输入

global-pre-scan 产出的 per-group matched_rules，每组已知：file_group 名、文件数、激活的规则文件、匹配的条例ID 列表。

## 执行流程

### Step 0 — 创建 yaml 输出目录

调用 `{skill_base}/scripts/workflow.create_review_dir.py` 创建本次检视的结构化 yaml 输出目录：
- 命令：`python3 {skill_base}/scripts/workflow.create_review_dir.py --type {review_type} --id {review_id}`
- 捕获 stdout 输出作为 `yaml_dir`（绝对路径，如 `/tmp/pr1234_a3b7x9`）
- 将 `yaml_dir` 作为返回值之一回传主 Agent。**主 Agent 保留 yaml_dir 用于启动 collector 服务和阶段5 脚本调用，不传给 clause-review / design-check 子 Agent**（子 Agent 通过 collector HTTP 端点提交 yaml，不接触目录路径）
- 目录创建失败（exit code 非 0）则终止，报错返回

### Step 1 — 条例归类

将每个 file_group 的 matched_rules 按类别归类：

| 条例ID 前缀 | 类别 | 优先级 |
|------------|------|--------|
| SEC（cpp-secure）中的数值安全/内存安全/输入验证 | 高危安全 | 1 |
| TOPK-8 等 TOPK 安全类 | 高危安全 | 1 |
| API（ascendc-api） | API使用 | 2 |
| SEC（cpp-secure）中的资源管理/并发安全/类型安全 | 一般安全 | 2 |
| GEN（cpp-general） | 通用规范 | 4 |
| SIMT（simt-api-analysis） | 领域规则 | 3 |
| MC2（mc2-specific） | 领域规则 | 3 |
| PERF / PREC / TIL（ascendc-perf） | 性能 | 4 |
| CMP（compile-secure） | 编译 | 5 |
| PY（python-secure） | Python | 5 |
| cpp-style 全部 19 条 | 代码风格 | 1（专项，见 Step 2.5） |

### Step 2 — 小组策略打组

按类别分组，每组上限按各条款所属文件 `<检视负载>` 头的 `通用检视子 agent 检视条款容量上限` 字段（合并组取最小值）。具体容量值由各文件自声明，不在此重复。

每组的文件列表维持 file_group 的原始文件（≤5 文件，由 file-split 保证）。

每组打标：
```
{
  group_id: "kernel_G1_安全_01",
  file_group: "kernel_G1",
  file_list: ["file1.cpp", "file2.h", ...],  // ≤5 文件
  rule_ids: ["SEC-2.1", "SEC-2.3"],          // 按容量打包
  priority: 1,
  file_count: 5,
  rule_count: 2,
  capacity: 10,                               // 来自 cpp-secure <检视负载> 头，合并组取最小值
  estimated_load: 10                          // file_count × rule_count（无加权，仅用于排序粗排）
}
```

### Step 2.5 — 代码风格专项分组

cpp-style 的 19 条条例**不按 file_group 拆分**，合并为 1 个跨文件组的全局组：

```
{
  group_id: "style_global",
  file_group: "ALL",                      // 跨所有文件组
  file_list: [所有变更的 C++ 文件],
  rule_ids: [cpp-style 全部 19 条],
  priority: 1,                            // 并入第一波
  file_count: {C++ 文件总数},
  rule_count: 19,
  estimated_load: file_count × 19,             // cpp-style 专项，不参与容量打包
  style: true                                  // 标记：读取方式与输出格式特殊
}
```

- 读取方式：直接 Read `references/cpp-style.md` 全文（其专属检视方法已声明不走 `Grep ^{条例ID}` 逐条定位）
- 输出标记：所有结果以 `[STYLE]` 前缀输出，供报告单独归入「代码风格」章节，不进 PASS/FAIL/SUSPICIOUS 统计
- 派发位置：并入第一波（与高危安全条例同波并行）

### Step 3 — 负载感知波次构建

```
1. 将所有 rule_group 按 priority 升序排列（priority 1 先）

2. 同 priority 内按 estimated_load（file_count × rule_count）降序粗排
   （重负载组先派发，避免轻负载组全跑完了重负载还在等）

3. 贪心构建波次:
   wave = []
   for group in sorted_groups:
     if len(wave) < 6:
       wave.append(group)
     else:
       开始新 wave

4. 组间均衡检查（每波构建完成后）:
   统计本波中各 file_group 的占比
   若某 file_group 在单波中 >5 组（即 >50%）:
     将其超出 5 组的 group 推迟到下一波
     （避免某文件组独占一波，其他组的检视饿死）

5. 输出波次规划表
```

### 输出格式

```
yaml_dir: /tmp/{目录名}

全局波次规划:

Wave 1（10组，优先级1-2）:
  kernel_G1_安全_01: [SEC-2.1, SEC-2.3] | 5文件 | load=10
  kernel_G2_安全_01: [SEC-2.1, SEC-3.1] | 3文件 | load=6
  host_G1_安全_01:   [SEC-2.1, SEC-4.1] | 6文件 | load=12
  kernel_G1_API_01:  [API-3, API-7]      | 5文件 | load=10
  ...

Wave 2（8组，优先级2-3）:
  ...

Wave 3（7组，优先级3-4）:
  ...

共 {G} 组，{W} 波
```

## 约束

- 本步骤由子 Agent 执行，Step 0（调用 workflow.create_review_dir.py）由该子 Agent 负责调用，将 `yaml_dir` 作为返回值回传主 Agent
- 每文件组单波占比 ≤50%（均衡硬约束）
- 不生成报告文件，波次规划表直接用于 Stage 2
