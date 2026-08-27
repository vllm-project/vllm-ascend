# 大型 PR 检视场景

## 触发
由 `workflows/pr-review.md` Stage 0 调 `scripts/workflow.review_mode.py` 判定 mode=large（阈值见 `scripts/workflow.review-thresholds.yaml`）后跳转进入。不单独暴露给用户。

## 编排

### 任务清单

启动时创建 6 个固定任务（全部 pending，若从 pr-review 跳转则先清理旧任务）：

| 任务 | 阶段 | 执行者 |
|------|------|--------|
| 任务0 | 文件分组 + 预扫描 + 设计文档探测 | file-split（子Agent）→ global-pre-scan（子Agent × N 并行）∥ docs-detect（子Agent × 1，与 file-split 并行） |
| 任务1 | 摘要 + 分组 + API 预研 | summarize（子Agent × N）∥ clause-grouping（子Agent × 1）∥ api-prestudy（子Agent × 1，仅 Kernel 侧） |
| 任务2 | 负载感知波次检视 | 逐波派发通用检视子 agent |
| 任务3 | 共享文件检视 + 综合研判 | shared 检视（子Agent）→ synthesize（主Agent） |
| 任务4 | 合并结果 + 设计一致性检查 | merge（主Agent）→ design-check（专项检视子 agent × 1，仅 docs_input 非空时） |
| 任务5 | 行号校验 + 报告 | line-verify（拆分路由）→ report-write（主Agent） |

### 阶段0：文件分组 + 预扫描 + 设计文档探测

1. 将任务0 标记为 in_progress
2. 若 diff_path 和 repo_path 已由上游传入 → 跳过 code-fetch
3. 主 Agent Read diff 前 200 行，提取变更文件路径列表
4. **并行派发**：① 1 个子 Agent 执行 `steps/pr-large-review.file-split.md`（传入文件路径列表，产出 file_groups）；② 1 个子 Agent 执行 `steps/common.docs-detect.md`（传入 repo_path + 用户已指明文档路径，产出 docs_input）。若上游 pr-review 已传入 docs_input 则跳过 docs-detect
5. 对每个 file_group **并行派发子 Agent** 执行 `steps/pr-large-review.global-pre-scan.md`：
   - 传入：group_file_list + repo_path
   - 产出：该组的 matched_rules（条例级匹配清单）
   - 每波 ≤6 Agent（上限见 `core/review-load-balance.md`），超过 6 组分批
6. **预扫描完整性校验**：每波子 Agent 返回后，检查是否每组均返回非空 matched_rules。对空返回的 file_group 自动补派一次
7. 收集 per-group matched_rules + docs_input，将任务0 标记为 done

### 阶段1：摘要 + 分组 + API 预研（并行派发）

1. 将任务1 标记为 in_progress
2. 在单个消息中并行派发子 Agent：
   - **summarize × N**：对每个 file_group 派发，Read `steps/pr-large-review.code-summarize.md`，每波 ≤6 Agent（见 `core/review-load-balance.md`）
    - **clause-grouping × 1**：派发 1 个子 Agent，Read `steps/pr-large-review.clause-grouping.md`，传入 per-group matched_rules + 检视类型 `pr` + 检视标识 `{pr_number}`。**该子 Agent 负责调用 `scripts/workflow.create_review_dir.py --type pr --id {pr_number}` 创建 yaml 输出目录**，返回值含 `yaml_dir` 路径
    - **api-prestudy × 1**（条件派发：仅当 diff 含 `op_kernel/` 路径或代码特征判定为 Kernel/混合侧时）：Read `steps/common.api-prestudy.md`，传入 Kernel 侧文件列表 + 预研报告路径 `./operators/pr-{pr_number}/api_prestudy.md`
3. **摘要完整性校验**：所有 summarize 子 Agent 返回后，执行 `ls {概要输出目录}/` 检查概要文件是否均存在且非空（`[ -s ]` 校验）。缺失或空文件的组自动补派一次
4. 收集 per-group summary_path + 全局波次规划表 + **yaml_dir** + API 预研路径（若已派发），将任务1 标记为 done

### 阶段2：负载感知波次逐条检视

1. 将任务2 标记为 in_progress
2. **启动 yaml collector 服务**（子 Agent 通过 HTTP 提交 yaml，不接触 yaml 输出目录路径）：
   - 选可用端口：`PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")`
   - 后台启动（分离 stdio 避免阻塞 shell）：`setsid python3 {skill_base}/scripts/workflow.submit_server.py {yaml_dir} {PORT} > /tmp/collector_{PORT}.log 2>&1 < /dev/null &`（保存 PID）
   - 就绪检查（最多重试 2 次）：`sleep 1 && curl -s http://127.0.0.1:{PORT}/health || (sleep 2 && curl -s http://127.0.0.1:{PORT}/health)`；若仍失败，检查 `/tmp/collector_{PORT}.log` 排查原因并尝试重启
3. Read `steps/pr-large-review.clause-review.md` 获取 prompt 模板
4. 使用波次规划表逐波派发：每波 ≤6 组（见 `core/review-load-balance.md`），每组按各文件 `<检视负载>` 头的 `通用检视子 agent 检视条款容量上限` 打包（合并组取最小值）+ ≤5 文件，波内并行波间串行。每组 prompt 填入 **collector_port（=步骤2 选定的 PORT）**，**不传 yaml_output_dir**。子 Agent 通过 `curl -X POST http://127.0.0.1:{PORT}/submit` 提交 yaml 结果，不再以文本消息返回结果
5. **波次完整性校验**：每波所有子 Agent 返回后，执行 `ls {yaml_dir}/*.yaml | wc -l` 检查落盘数。若与该波派发的子 Agent 数量不符，执行 `ls {yaml_dir}/ | sort` 逐组核对，对缺失结果的 group 自动补派一次
6. 收集全部结果（子 Agent 结果已通过 collector 落盘 yaml，无需收集文本返回值），将任务2 标记为 done

### 阶段3：共享文件检视 + 综合研判

1. 将任务3 标记为 in_progress
2. 若 shared_bucket 非空，派发 shared 检视（≤1 波），prompt 填入 **collector_port**（collector 服务仍在运行）
3. 主 Agent Read + 执行 `steps/pr-large-review.synthesize.md`：跨文件组模式识别、冲突解决、置信度过滤
4. 将任务3 标记为 done

### 阶段4：合并结果 + 设计一致性检查

1. 将任务4 标记为 in_progress
2. 主 Agent Read + 执行 `steps/pr-large-review.merge.md`
3. **设计一致性检查**：若阶段0 的 docs_input 非空，派发 1 个专项检视子 agent（design-check，`subagent_type: "general"`），填入 docs_input + diff路径 + repo_path + 合并后摘要路径 + API 预研路径（若存在）+ **collector_port**。子 Agent 通过 curl 提交 design.yaml 结果。子 Agent 内部读设计文档 + 建立设计映射 + 复用合并摘要/API预研做 S1-S7 + D8 整体对照（避免按文件组碎片化）
4. **关闭 collector**：`kill {COLLECTOR_PID}`（design-check 子 Agent 完成后，所有 yaml 提交结束）
5. 将任务4 标记为 done

### 阶段5：行号校验 + 报告

1. 将任务5 标记为 in_progress
2. Read + 执行 `steps/pr-review.line-verify.md` 薄壳，传入 yaml_dir + diff路径 + repo_path。薄壳内部调用 `scripts/workflow.line_verify.py` 扫描 yaml 目录、原地修正行号 + diff 红线校验（clause 类走 diff 红线、design 类无红线，拆分逻辑在脚本内实现）
3. Read + 执行 `steps/common.report-write.md` 薄壳：传入 yaml_dir + 报告输出路径 + 头部元信息。薄壳内部调用 `scripts/workflow.assemble_report.py` 组装报告正文，主 Agent 替换头部占位符。**替换完成后检查报告行数（`wc -l`），若 >5000 行则派发 `steps/common.report-filter.md` 过滤子 Agent 原地精简报告**
4. 输出 `./operators/pr-{N}/{N}_review_summary.md`，将任务5 标记为 done

---

## 约束

- 严格按阶段顺序执行，禁止跳步
- code-fetch 失败则终止流程
- 禁止提前 Read 未执行阶段的 step 文件
- 每波 ≤6 Agent（见 `core/review-load-balance.md`），>4 文件组分批
- **主 Agent 只做编排派发**——file-split、global-pre-scan、summarize、clause-grouping 全部由子 Agent 执行
- design-check 置于 Stage4 merge 之后：复用合并后的全局摘要做整体对照，避免按文件组碎片化；属独立轨道，不进 clause 波次规划
- docs_input 为空时不派发 design-check，报告退化为纯条例检视
- 阶段5 走 `pr-review.line-verify.md` 薄壳：脚本内部对 clause 类 yaml 做 diff 红线校验、对 design 类 yaml 无红线（拆分路由在脚本内实现）
