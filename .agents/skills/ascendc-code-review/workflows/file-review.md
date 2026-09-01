# 文件检视场景

## 触发
检视代码、审核代码、检查规范、代码审查、帮我检视 xxx

---

## 编排

### 任务清单

启动时创建 4 个固定任务（全部 pending）：

| 任务 | 阶段 | 内容 |
|------|------|------|
| 任务0 | 代码概要 + API 预研 + 设计文档探测 + 检视计划设计 | 并行派发 code-summarize + api-prestudy（仅 Kernel 侧）+ docs-detect → 检视计划设计 |
| 任务1 | 逐条检视 | 按波次派发通用检视子 agent |
| 任务2 | 行号校对 | steps/common.line-verify.md |
| 任务3 | 撰写报告 | steps/common.report-write.md |

### 输入解析

从用户输入提取代码文件（支持：单文件路径、多文件路径、目录路径 find 枚举），统一为 `file_input`（可以是单个路径，也可以是多个路径）。

### 阶段0：代码概要 + API 预研 + 设计文档探测 + 检视计划设计

1. 将任务0 标记为 in_progress
2. 从 file_input 提取算子名，确认文件存在。统计 file_input 代码总行数（排除注释空行），执行 `python3 {skill_base}/scripts/workflow.review_mode.py --lines {代码总行数}`（无 --files，文件检视不触发大型），解析返回 JSON 取 mode/guidance
3. **在单个消息中并行派发子 Agent**（代码概要、设计文档探测总是派发，API 预研仅当 file_input 含 `op_kernel/` 或判定为 Kernel/混合侧时派发）：

- **代码概要子 Agent**：Read+执行 `steps/file-review.code-summarize.md`，传入 file_input + 概要输出路径 `./operators/{operator_name}/code_summary.md`
- **API 预研子 Agent**（仅当 file_input 含 `op_kernel/` 或判定为 Kernel/混合侧）：Read+执行 `steps/common.api-prestudy.md`，传入 file_input（仅 Kernel 侧）+ 输出路径 `./operators/{operator_name}/api_prestudy.md`
- **设计文档探测子 Agent**：Read+执行 `steps/common.docs-detect.md`，传入 file_input + 用户已指明的文档路径（明确给出则传，否则为空）→ 返回 docs_input（路径/目录或空）

4. 等待三个子 Agent 全部返回，收集：代码概要→侧别+概要路径；API 预研→预研报告路径（若已派发）；设计文档探测→docs_input（路径/目录或空）
5. **侧别回填**：若 API 预研子 Agent 未派发（纯 Tiling 侧），跳过 API 预研路径
6. Read + 执行 `steps/common.plan-design.md`，派发子 Agent 产出检视计划（通用分组 + 专项清单 + 跳过清单 + 仅核对清单）。传入 file_input + 概要路径 + API 预研路径（若存在）+ docs_input + scope_hint + 检视类型 `file` + 检视标识留空（文件检视无 PR 号）+ mode + guidance（mode/guidance 由 file-review 入口调 review_mode.py 判定，plan-design 按该 mode 的 guidance 编排）。**plan-design 负责调用 `scripts/workflow.create_review_dir.py` 创建 yaml 输出目录**，返回值含 `yaml_dir` 路径
7. 将任务0 标记为 done

### 阶段1：逐条检视 + 设计一致性检查

1. 将任务1 标记为 in_progress
2. **启动 yaml collector 服务**（子 Agent 通过 HTTP 提交 yaml，不接触 yaml 输出目录路径）：
   - 选可用端口：`PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")`
   - 后台启动（分离 stdio 避免阻塞 shell）：`setsid python3 {skill_base}/scripts/workflow.submit_server.py {yaml_dir} {PORT} > /tmp/collector_{PORT}.log 2>&1 < /dev/null &`（保存 PID）
   - 就绪检查（最多重试 2 次）：`sleep 1 && curl -s http://127.0.0.1:{PORT}/health || (sleep 2 && curl -s http://127.0.0.1:{PORT}/health)`；若仍失败，检查 `/tmp/collector_{PORT}.log` 排查原因并尝试重启
3. Read `steps/file-review.clause-review.md` 获取 prompt 模板
4. 按阶段0 检视计划的通用分组，逐波派发：
   - 每波在单个消息中并行调用 ≤6 个 `Agent` 工具（上限见 `core/review-load-balance.md`）
   - `subagent_type` 使用 `"general"`
   - 每组用 prompt 模板填入：侧别 + 条例ID和标题 + file_input + 代码概要路径 + API 预研路径（若存在）+ **collector_port（=步骤2 选定的 PORT）**，**不传 yaml_output_dir**
   - 波次内并行，波次间串行
   - 子 Agent 通过 `curl -X POST http://127.0.0.1:{PORT}/submit` 提交 yaml 结果，不再以文本消息返回结果
5. **🆕design-check 与波次1 同消息并行**：派发**波次1 的那一条消息时**，若检视计划专项清单含 design-check，在同一消息里额外加入 1 个专项检视子 agent（design-check，`subagent_type: "general"`，不进 plan-design 通用分组，独立输出）。填入 docs_input + file_input + 代码概要路径 + API 预研路径（若存在）+ **collector_port**。子 Agent 内部读设计文档 + 建立设计映射 + 复用概要/API预研做 S1-S7 + D8 对照，通过 curl 提交 design.yaml 结果。**禁止把 design-check 排到所有波次之后单独派发**——它必须与波次1 的通用检视子 agent 在同一条消息里发出，以实现真正并行
6. 波次2 及之后：仅派发通用检视子 agent（design-check 已在波次1 并行发出，无需重复）
7. 每波完成后输出进度，所有波次完成后汇总（子 Agent 结果已通过 collector 落盘 yaml，无需收集文本返回值）
8. **波次完整性校验**：每波所有子 Agent 返回后，执行 `ls {yaml_dir}/*.yaml | wc -l` 检查落盘数。若与已派发的子 Agent 数量不符，执行 `ls {yaml_dir}/ | sort` 逐组核对，对缺失结果的 group 自动补派一次
9. **关闭 collector**：`kill {COLLECTOR_PID}`（所有波次 + design-check 完成后）
10. 将任务1 标记为 done

### 阶段2：行号校对

1. 将任务2 标记为 in_progress
2. Read + 执行 `steps/common.line-verify.md` 薄壳：传入 yaml_dir（=阶段0 plan-design 返回值）+ file_input（或源码根目录，用于行号定位）。薄壳内部调用 `scripts/workflow.line_verify.py` 扫描 yaml 目录、原地修正行号
3. 将任务2 标记为 done

### 阶段3：撰写报告

1. 将任务3 标记为 in_progress
2. Read + 执行 `steps/common.report-write.md` 薄壳：传入 yaml_dir + 报告输出路径 + 头部元信息（侧别/文档列表/总条例数/docs_input/时间戳）。薄壳内部调用 `scripts/workflow.assemble_report.py` 组装报告正文，主 Agent 替换头部占位符。**替换完成后检查报告行数（`wc -l`），若 >5000 行则派发 `steps/common.report-filter.md` 过滤子 Agent 原地精简报告**
3. 报告输出路径 `./operators/{operator_name}/{source_file}_review_summary.md`
4. 将任务3 标记为 done

---

## 上下文传递链

```
                 ┌─ code-summarize → 侧别 + 概要路径 + 跨文件关系
阶段0（并行） ───┤─ api-prestudy → API 预研报告路径（仅 Kernel 侧）
                 └─ docs-detect → docs_input（设计文档路径/目录或空）
                         ↓ 三个子 Agent 全部返回后
                  plan-design → 检视计划（通用分组 + 专项清单 + 跳过/仅核对清单）+ yaml_dir
                         ↓
阶段1 → 子 Agent 将结果写入 yaml_dir（clause 每条例一个 yaml + design.yaml）
          ↓
阶段2 → workflow.line_verify.py 扫描 yaml_dir、原地修正行号（文件检视无 diff 红线）
          ↓
阶段3 → workflow.assemble_report.py 读 yaml_dir 组装报告正文 + 主 Agent 补头部元信息
```

## 约束

- 严格按阶段顺序执行，禁止跳步
- 阶段0 的子 Agent 必须在单个消息中并行派发（代码概要 + 设计文档探测总是，API 预研仅 Kernel 侧）；plan-design 在三个子 Agent 全部返回后单独派发
- design-check 发射与否由 plan-design 专项清单决定，workflow 阶段1 只按专项清单执行派发（含 design-check 则与波次1 同消息并行，不进通用分组）；专项清单不含 design-check 时报告退化为纯条例检视
- 禁止提前 Read 未执行阶段的 step 文件
